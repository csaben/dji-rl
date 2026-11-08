"""Race maker backend: train a racing agent on a gate course in a splat scene, and record its ghost.

The agent is state-based (no camera): it sees the next two gates in its body frame, its velocity,
and ESDF clearance probes around it, and flies the same stick dynamics as the viewer (Mini 3 Normal
mode limits, 200 ms command delay). Collisions use the scene ESDF, as everywhere else. So its lap
times are directly comparable with a human time trial in the sandbox.

Course (scene frame, metres; same as the planner's /api/lib/races/<slug>):
  {"name", "scene", "laps": 1, "gates": [{"x","y","z","yaw_deg","diameter_m"}, ...],
   "start": {"x","y","z","yaw_deg"} (optional; default 4 m before gate 1, facing through it, hovering)}
Timing (= the sandbox time trial): the clock starts when gate 1 is passed; splits are the later gates'
pass times from there, time_s the last one. Ghost traj t is on that clock (the run-up has t < 0).
yaw_deg: math angle (0 = +x east, CCW), the direction you fly THROUGH the gate.
Pass rule (same as the client): the segment between consecutive control-step positions crosses the
gate plane in the forward direction, and the crossing point is within diameter/2 of the centre.

    .venv/bin/djirl race-train --course course.json --minutes 10 --out runs/race/<job>
writes  status.json (live), policy.pt, ghost.json {time_s, splits, traj [[t,x,y,z,yaw_rad],...] at 10 Hz}
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .dynamics import StickDynamics, StickParams
from .scene import ESDF

REPO = Path(__file__).resolve().parents[2]
DRONE_R = 0.25
HZ = 10.0
GATE_TIMEOUT_S = 10.0      # no pass for this long -> episode fails
START_BEHIND_M = 4.0       # default spawn: this far before gate 1 on its axis, facing through it (= the sandbox)


def course_hash(c: dict) -> str:
    key = {"scene": c["scene"], "laps": int(c.get("laps", 1)),
           "gates": [[round(float(g[k]), 2) for k in ("x", "y", "z", "yaw_deg", "diameter_m")] for g in c["gates"]],
           "start": c.get("start")}
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def load_esdf(scene: str, device) -> ESDF:
    z = np.load(REPO / "data" / "splats" / scene / "esdf.npz")
    return ESDF(torch.from_numpy(z["sdf"].astype(np.float32)), torch.from_numpy(z["origin"]).float(), float(z["voxel"])).to(device)


class RaceEnv:
    OBS_DIM = 12 + 4 + 19 + 4 + 1

    def __init__(self, course: dict, n: int, device, dr=0.1, wind=0.1, random_start_frac=0.5, seed=0):
        self.dev, self.n = torch.device(device), n
        self.esdf = load_esdf(course["scene"], self.dev)
        laps = int(course.get("laps", 1))
        g = course["gates"] * laps
        self.G = len(g)
        self.n_course = len(course["gates"])
        self.gc = torch.tensor([[float(q["x"]), float(q["y"]), float(q["z"])] for q in g], device=self.dev)
        yaw = torch.tensor([math.radians(float(q["yaw_deg"])) for q in g], device=self.dev)
        self.gyaw = yaw
        self.gn = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], -1)
        self.grad = torch.tensor([float(q.get("diameter_m", 2.5)) / 2 for q in g], device=self.dev)
        st = course.get("start")
        if st:
            self.start = torch.tensor([float(st["x"]), float(st["y"]), float(st["z"])], device=self.dev)
            self.start_yaw = math.radians(float(st.get("yaw_deg", math.degrees(float(yaw[0])))))
        else:
            self.start = self.gc[0] - self.gn[0] * START_BEHIND_M
            self.start_yaw = float(yaw[0])
        if float(self.esdf.query(self.start[None])[0]) < DRONE_R + 0.2:
            raise ValueError("start position is inside an obstacle (move gate 0 or set course.start)")
        self.P = StickParams(dr=dr, wind_std=wind, delay_steps=2)
        self.dyn = StickDynamics(n, self.P, self.dev, control_dt=1 / HZ, substeps=5)
        self.rsf = random_start_frac
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        a = torch.arange(8, device=self.dev) * (2 * math.pi / 8)
        ring = torch.stack([torch.cos(a), torch.sin(a), torch.zeros_like(a)], -1)
        self.probe = torch.cat([ring * 1.5, ring * 3.0, torch.tensor([[0, 0, 1.5], [0, 0, -1.5]], device=self.dev)])  # body FLU
        self.nxt = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.t = torch.zeros(n, device=self.dev)
        self.t_pass = torch.zeros(n, device=self.dev)
        self.prev_d = torch.zeros(n, device=self.dev)
        self.last_a = torch.zeros(n, 4, device=self.dev)
        self.ep_ret = torch.zeros(n, device=self.dev)
        self.reset_idx(torch.arange(n, device=self.dev), force_start=True)

    def _rand(self, m):
        return torch.rand(m, device=self.dev, generator=self.gen)

    def reset_idx(self, idx, force_start=False):
        m = len(idx)
        pos = self.start.expand(m, 3).clone()
        yaw = torch.full((m,), self.start_yaw, device=self.dev)
        vel = torch.zeros(m, 3, device=self.dev)
        nxt = torch.zeros(m, dtype=torch.long, device=self.dev)
        if not force_start and self.rsf > 0:
            rnd = self._rand(m) < self.rsf
            k = (self._rand(m) * self.G).long().clamp(max=self.G - 1)
            n_, c = self.gn[k], self.gc[k]
            lat = torch.stack([-n_[:, 1], n_[:, 0], torch.zeros(m, device=self.dev)], -1)
            p = c - n_ * (2 + 6 * self._rand(m))[:, None] + lat * ((self._rand(m) - 0.5) * 2)[:, None]
            p[:, 2] += (self._rand(m) - 0.5) * 1.0
            ok = rnd & (self.esdf.query(p) > DRONE_R + 0.4) & self.esdf.in_bounds(p)
            pos = torch.where(ok[:, None], p, pos)
            yaw = torch.where(ok, self.gyaw[k] + (self._rand(m) - 0.5) * 0.6, yaw)
            vel = torch.where(ok[:, None], n_ * (6 * self._rand(m))[:, None], vel)
            nxt = torch.where(ok, k, nxt)
        self.dyn.reset(idx, pos, yaw, vel)
        self.nxt[idx] = nxt
        self.t[idx] = 0
        self.t_pass[idx] = 0
        self.prev_d[idx] = (self.gc[nxt] - pos).norm(dim=-1)
        self.last_a[idx] = 0
        self.ep_ret[idx] = 0

    def _body(self, v):
        c, s = torch.cos(self.dyn.yaw), torch.sin(self.dyn.yaw)
        return torch.stack([c * v[:, 0] + s * v[:, 1], s * v[:, 0] - c * v[:, 1], v[:, 2]], -1)  # FRU

    def obs(self):
        pos, yaw = self.dyn.pos, self.dyn.yaw
        feats = []
        for j in (0, 1):
            k = (self.nxt + j).clamp(max=self.G - 1)
            rel = self._body(self.gc[k] - pos) / 10
            dy = self.gyaw[k] - yaw
            feats += [rel, torch.cos(dy)[:, None], torch.sin(dy)[:, None], (self.grad[k] / 2.5)[:, None]]
        c, s = torch.cos(yaw), torch.sin(yaw)
        pr = self.probe
        off = torch.stack([c[:, None] * pr[:, 0] - s[:, None] * pr[:, 1], s[:, None] * pr[:, 0] + c[:, None] * pr[:, 1],
                           pr[:, 2].expand(self.n, -1)], -1)
        pts = pos[:, None] + off
        sd = self.esdf.query(pts.reshape(-1, 3)).reshape(self.n, -1)
        sd = torch.where(self.esdf.in_bounds(pts.reshape(-1, 3)).reshape(self.n, -1), sd, torch.zeros_like(sd))
        sd0 = self.esdf.query(pos)[:, None]
        feats += [self._body(self.dyn.vel) / 8, self.dyn.yaw_rate[:, None] / 1.6,
                  sd.clamp(-1, 5) / 5, sd0.clamp(-1, 5) / 5, self.last_a, ((self.G - self.nxt) / self.G)[:, None]]
        # feats: 2x(3+1+1+1)=12 | vel 3 + yaw rate 1 | probes 18 + 1 | last action 4 | remaining 1
        return torch.cat(feats, -1)

    def step(self, a):
        a = a.clamp(-1, 1)
        p0 = self.dyn.pos.clone()
        subs = self.dyn.step(a)
        p1 = self.dyn.pos
        self.t += 1 / HZ
        sw = torch.stack(subs, 1)                                       # [n,S,3]
        sdf = self.esdf.query(sw.reshape(-1, 3)).reshape(self.n, -1)
        oob = ~self.esdf.in_bounds(sw.reshape(-1, 3)).reshape(self.n, -1)
        coll = ((sdf < DRONE_R) | oob).any(1)
        k = self.nxt.clamp(max=self.G - 1)
        c, nrm = self.gc[k], self.gn[k]
        s0, s1 = ((p0 - c) * nrm).sum(-1), ((p1 - c) * nrm).sum(-1)
        frac = (s0 / (s0 - s1).clamp_min(1e-6)).clamp(0, 1)
        x = p0 + (p1 - p0) * frac[:, None]
        passed = (s0 < 0) & (s1 >= 0) & ((x - c).norm(dim=-1) <= self.grad[k])
        d = (c - p1).norm(dim=-1)
        r = (self.prev_d - d).clamp(-1.5, 1.5) - 0.02
        r = r + passed.float() * 10.0
        self.nxt = self.nxt + passed.long()
        self.t_pass = torch.where(passed, self.t, self.t_pass)
        finished = self.nxt >= self.G
        k2 = self.nxt.clamp(max=self.G - 1)
        self.prev_d = torch.where(passed, (self.gc[k2] - p1).norm(dim=-1), d)
        stuck = (self.t - self.t_pass) > GATE_TIMEOUT_S
        r = r - coll.float() * 10.0 - 0.02 * (a - self.last_a).pow(2).sum(-1)
        self.last_a = a
        done = coll | finished | stuck
        self.ep_ret += r
        info = {"passed": passed, "finished": finished & done, "collided": coll & done, "t": self.t.clone(),
                "ret": self.ep_ret.clone(), "x": x}
        if done.any():
            self.reset_idx(done.nonzero()[:, 0])
        return self.obs(), r, done, info


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim=4, h=256):
        super().__init__()
        f = lambda: nn.Sequential(nn.Linear(obs_dim, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, 128), nn.ELU())
        self.pi, self.v = f(), f()
        self.mu, self.vh = nn.Linear(128, act_dim), nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        nn.init.zeros_(self.mu.bias)
        self.mu.weight.data.mul_(0.1)

    def dist(self, o):
        return torch.distributions.Normal(self.mu(self.pi(o)), self.log_std.exp())

    def value(self, o):
        return self.vh(self.v(o))[:, 0]


@torch.no_grad()
def evaluate(course, model, device, max_s=None):
    """Deterministic run from the start, no randomisation -> (success, time_s, splits, traj)."""
    env = RaceEnv(course, 1, device, dr=0.0, wind=0.0, random_start_frac=0.0)
    max_s = max_s or (env.G * GATE_TIMEOUT_S + 5)
    o = env.obs()
    traj, splits = [], []
    p, y = env.dyn.pos[0], float(env.dyn.yaw[0])
    traj.append([0.0, *[round(float(v), 3) for v in p], round(y, 4)])
    t = 0.0
    while t < max_s:
        a = model.dist(o).mean
        o, _, done, info = env.step(a)
        t += 1 / HZ
        if bool(info["passed"][0]):
            frac_t = t  # crossing time at control-step resolution
            splits.append(round(frac_t, 2))
        if bool(done[0]):
            ok = bool(info["finished"][0])
            # the env auto-reset: log the final pre-reset pose from the crossing point
            if ok:
                xx = info["x"][0]
                traj.append([round(t, 2), *[round(float(v), 3) for v in xx], traj[-1][-1]])
            t0 = splits[0] if splits else 0.0
            traj = [[round(r[0] - t0, 2), *r[1:]] for r in traj]
            sp = [round(x - t0, 2) for x in splits]
            return ok, (sp[-1] if ok else None), sp, traj, bool(info["collided"][0])
        p, y = env.dyn.pos[0], float(env.dyn.yaw[0])
        traj.append([round(t, 2), *[round(float(v), 3) for v in p], round(y, 4)])
    return False, None, splits, traj, False


def train(course: dict, out: Path, minutes=10.0, device="cuda:1", envs=2048, horizon=64, epochs=4, mb=4,
          gamma=0.99, lam=0.95, lr=3e-4, clip=0.2, ent=0.002, eval_every=10):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "course.json").write_text(json.dumps(course))
    status = {"status": "running", "course_hash": course_hash(course), "started": time.time(), "iter": 0, "steps": 0,
              "episodes": 0, "train_success": None, "best_time_s": None, "best_splits": None, "minutes": minutes}

    def write_status(**kw):
        status.update(kw, updated=time.time())
        tmp = out / "status.json.tmp"
        tmp.write_text(json.dumps(status))
        tmp.replace(out / "status.json")

    try:
        env = RaceEnv(course, envs, device)
        dev = env.dev
        model = MLPPolicy(env.OBS_DIM).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        o = env.obs()
        t0, it, n_ep, best = time.time(), 0, 0, None
        succ_hist = []
        write_status()
        while time.time() - t0 < minutes * 60:
            it += 1
            O, A, LP, R, D, V = [], [], [], [], [], []
            with torch.no_grad():
                for _ in range(horizon):
                    d = model.dist(o)
                    a = d.sample()
                    v = model.value(o)
                    o2, r, done, info = env.step(a)
                    O.append(o); A.append(a); LP.append(d.log_prob(a).sum(-1)); R.append(r); D.append(done.float()); V.append(v)
                    if done.any():
                        n_ep += int(done.sum())
                        succ_hist += info["finished"][done].float().tolist()
                    o = o2
                v_last = model.value(o)
                adv = torch.zeros(horizon, env.n, device=dev)
                g = torch.zeros(env.n, device=dev)
                for i in reversed(range(horizon)):
                    nv = v_last if i == horizon - 1 else V[i + 1]
                    delta = R[i] + gamma * nv * (1 - D[i]) - V[i]
                    g = delta + gamma * lam * (1 - D[i]) * g
                    adv[i] = g
                Vt = torch.stack(V)
                ret = adv + Vt
            Ob, Ab, LPb = torch.stack(O).flatten(0, 1), torch.stack(A).flatten(0, 1), torch.stack(LP).flatten(0, 1)
            advb, retb = adv.flatten(), ret.flatten()
            advb = (advb - advb.mean()) / (advb.std() + 1e-8)
            N = len(Ob)
            for _ in range(epochs):
                perm = torch.randperm(N, device=dev)
                for j in range(mb):
                    ix = perm[j * N // mb:(j + 1) * N // mb]
                    d = model.dist(Ob[ix])
                    lp = d.log_prob(Ab[ix]).sum(-1)
                    ratio = (lp - LPb[ix]).exp()
                    pg = -torch.min(ratio * advb[ix], ratio.clamp(1 - clip, 1 + clip) * advb[ix]).mean()
                    vl = (model.value(Ob[ix]) - retb[ix]).pow(2).mean()
                    loss = pg + 0.5 * vl - ent * d.entropy().sum(-1).mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            succ_hist = succ_hist[-2000:]
            if it % eval_every == 0:
                ok, t_lap, splits, traj, _ = evaluate(course, model, device)
                if ok and (best is None or t_lap < best):
                    best = t_lap
                    ghost = {"time_s": t_lap, "splits": splits, "traj": traj, "course_hash": course_hash(course),
                             "job": out.name, "iter": it, "hz": HZ}
                    (out / "ghost.json").write_text(json.dumps(ghost))
                    torch.save({"model": model.state_dict(), "obs_dim": env.OBS_DIM}, out / "policy.pt")
                write_status(iter=it, steps=it * horizon * env.n, episodes=n_ep,
                             train_success=round(float(np.mean(succ_hist)), 3) if succ_hist else None,
                             best_time_s=best, best_splits=(json.loads((out / "ghost.json").read_text())["splits"]
                                                            if best is not None else None),
                             last_eval={"ok": ok, "time_s": t_lap, "gates": len(splits)},
                             elapsed_min=round((time.time() - t0) / 60, 2))
                print(json.dumps({k: status[k] for k in ("iter", "steps", "episodes", "train_success", "best_time_s")}
                                 | {"eval_gates": len(splits)}), flush=True)
        write_status(status="done", iter=it, episodes=n_ep, elapsed_min=round((time.time() - t0) / 60, 2))
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        write_status(status="failed", error=f"{type(e).__name__}: {e}")
        raise
