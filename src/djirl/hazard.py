"""Hazard model: from camera latents + velocity, predict how much room there is along the flight
path, and whether the drone must brake *now* given its speed.

Labels come free from the scene's ESDF (no need to actually crash):
  clear_vel   free distance (m) along the current velocity direction (body forward when ~hovering),
              for a sphere of the drone's radius, capped at RAY_MAX
  clear_fwd   same along the camera/body forward axis (what the camera actually looks at)
  hit_T       1 if flying on at constant velocity hits something within T seconds (T in HORIZONS)
  must_brake  1 if clear_vel < stopping distance(v) + margin, where
              stopping distance = v * latency + v^2 / (2 * a_brake)   (DJI brakes for us, but momentum)

On the phone the stop rule is physics + the learned clearance (velocity enters twice: as an input
to the net, and in the stopping distance), so faster flight stops earlier:
    brake if  pred_clear_vel  <  v*latency + v^2/(2*a_brake) + margin   or   p(must_brake) > thr
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HORIZONS = (0.5, 1.0, 2.0)
RAY_MAX = 20.0
RAY_STEP = 0.2


@dataclass
class BrakeModel:
    latency: float = 0.30      # s, video + decision + command latency (fit from Flight Lab later)
    a_brake: float = 5.0       # m/s^2 effective braking decel (fit from Flight Lab step-to-zero)
    margin: float = 1.5        # m extra buffer beyond the drone radius

    def stopping_distance(self, speed: torch.Tensor) -> torch.Tensor:
        return speed * self.latency + speed ** 2 / (2 * self.a_brake) + self.margin


@torch.no_grad()
def ray_clearance(esdf, origin: torch.Tensor, direction: torch.Tensor, radius: float,
                  max_dist: float = RAY_MAX, step: float = RAY_STEP) -> torch.Tensor:
    """Free distance along unit `direction` from `origin` for a sphere of `radius` ([N,3] -> [N])."""
    ts = torch.arange(0, max_dist + 1e-6, step, device=origin.device)
    pts = origin[:, None, :] + direction[:, None, :] * ts[None, :, None]
    sdf = esdf.query(pts.reshape(-1, 3)).reshape(len(origin), -1)
    hit = sdf < radius
    first = torch.where(hit.any(1), hit.float().argmax(1), torch.full((len(origin),), len(ts) - 1, device=origin.device))
    d = ts[first]
    return torch.where(hit.any(1), d, torch.full_like(d, max_dist))


@torch.no_grad()
def hazard_labels(env, brake: BrakeModel) -> dict:
    """Labels for the env's current state (call after a step, before reading the next obs)."""
    dyn = env.dyn
    vel = dyn.vel
    speed = vel.norm(dim=-1)
    fwd = torch.stack([torch.cos(dyn.yaw), torch.sin(dyn.yaw), torch.zeros_like(dyn.yaw)], -1)
    vdir = torch.where((speed > 0.3)[:, None], vel / speed.clamp_min(1e-6)[:, None], fwd)
    r = env.cfg.drone_radius
    clear_vel = ray_clearance(env.s.esdf, dyn.pos, vdir, r)
    clear_fwd = ray_clearance(env.s.esdf, dyn.pos, fwd, r)
    hits = torch.stack([(clear_vel < speed * T).float() for T in HORIZONS], -1)
    must_brake = (clear_vel < brake.stopping_distance(speed)).float()
    return {"clear_vel": clear_vel, "clear_fwd": clear_fwd, "hit": hits, "must_brake": must_brake, "speed": speed}


# ------------------------------------------------------------------ scripted data-collection pilot

class ScriptedPilot:
    """Flies toward the current waypoint with per-episode random cruise speed and heading bias, plus
    episodes that deliberately aim at nearby obstacles, so the data has plenty of near-misses and
    hits at every speed. Never brakes for obstacles itself (it is the thing we are labelling)."""

    def __init__(self, env, v_range=(1.0, 8.0), aim_frac=0.35, seed=0):
        self.env, self.v_range, self.aim_frac = env, v_range, aim_frac
        n, dev = env.n, env.dev
        self.g = torch.Generator(device=dev).manual_seed(seed)
        self.cruise = torch.zeros(n, device=dev)
        self.bias = torch.zeros(n, device=dev)
        self.aim = torch.zeros(n, dtype=torch.bool, device=dev)
        self.reroll(torch.ones(n, dtype=torch.bool, device=dev))

    def reroll(self, mask):
        m = int(mask.sum())
        if m == 0:
            return
        dev = self.env.dev
        lo, hi = self.v_range
        self.cruise[mask] = lo + (hi - lo) * torch.rand(m, device=dev, generator=self.g)
        self.bias[mask] = torch.randn(m, device=dev, generator=self.g) * 0.35
        self.aim[mask] = torch.rand(m, device=dev, generator=self.g) < self.aim_frac

    def act(self):
        env, dyn = self.env, self.env.dyn
        goal = env.wps[env.target] - dyn.pos
        if self.aim.any():
            # aim at the closest obstacle direction among 16 horizontal probes at 6 m
            ang = torch.linspace(-math.pi, math.pi, 17, device=env.dev)[:-1]
            probes = dyn.pos[:, None, :] + 6 * torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], -1)[None]
            sd = env.s.esdf.query(probes.reshape(-1, 3)).reshape(env.n, -1)
            best = ang[sd.argmin(1)]
            tgt = torch.stack([torch.cos(best), torch.sin(best), torch.zeros_like(best)], -1) * 6
            goal = torch.where(self.aim[:, None], tgt, goal)
        want_yaw = torch.atan2(goal[:, 1], goal[:, 0]) + self.bias
        err = torch.remainder(want_yaw - dyn.yaw + math.pi, 2 * math.pi) - math.pi
        a = torch.zeros(env.n, 4, device=env.dev)
        a[:, 0] = (self.cruise / env.cfg.stick.v_h_max) * (1 - err.abs() / math.pi).clamp(0.2, 1)
        a[:, 1] = 0.15 * torch.randn(env.n, device=env.dev, generator=self.g)
        a[:, 2] = (goal[:, 2] / 3).clamp(-1, 1) * 0.5
        a[:, 3] = (-err * 1.5).clamp(-1, 1)  # yaw right = clockwise = negative ENU error
        return a.clamp(-1, 1)


def collect(env, n_frames: int, out: Path, brake: BrakeModel, seed: int = 0, log_every: int = 50) -> Path:
    """Run the scripted pilot and save (latent stack, proprio, labels) for hazard training."""
    pilot = ScriptedPilot(env, seed=seed)
    obs = env.reset()
    steps = math.ceil(n_frames / env.n)
    keys = ("latent", "vec")
    buf = {k: [] for k in keys}
    lab = {k: [] for k in ("clear_vel", "clear_fwd", "hit", "must_brake", "speed")}
    t0 = time.time()
    for i in range(steps):
        lb = hazard_labels(env, brake)  # labels for the state the current obs shows
        buf["latent"].append(obs["latent"].half().cpu())
        buf["vec"].append(obs["vec"].half().cpu())
        for k in lab:
            lab[k].append(lb[k].half().cpu())
        obs, r, done, info = env.step(pilot.act())
        pilot.reroll(done)
        if i % log_every == 0:
            mb = torch.cat(lab["must_brake"]).float().mean().item()
            print(f"collect {i}/{steps} frames {(i + 1) * env.n} must_brake {mb:.3f} {time.time() - t0:.0f}s", flush=True)
    data = {k: torch.cat(v) for k, v in buf.items()} | {k: torch.cat(v) for k, v in lab.items()}
    data["meta"] = {"scene": env.s.name, "brake": asdict(brake), "horizons": HORIZONS, "frames": len(data["vec"])}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out)
    rates = {"hit": [round(x, 3) for x in data["hit"].float().mean(0).tolist()],
             "must_brake": round(data["must_brake"].float().mean().item(), 3)}
    print(f"saved {len(data['vec'])} frames -> {out}  positive rates {rates}", flush=True)
    return out


# ------------------------------------------------------------------ model

class HazardNet(nn.Module):
    """latent stack [B,S,4,h,w] + proprio vec [B,13] -> clearance (m) along velocity & forward,
    and logits for hit@HORIZONS + must_brake. Velocity is an explicit input (vec[:,4:7])."""

    def __init__(self, stack: int = 2, vec_dim: int = 13):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(stack * 4, 32, 3, padding=1), nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ELU(), nn.Flatten())
        self.vel = nn.Sequential(nn.Linear(vec_dim + 2, 64), nn.ELU())
        self.head = nn.Sequential(nn.LazyLinear(256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
                                  nn.Linear(128, 2 + len(HORIZONS) + 1))

    def forward(self, latent, vec):
        speed = (vec[:, 4:7] * 8).norm(dim=-1, keepdim=True)  # vec stores body velocity / 8
        v = torch.cat([vec, speed / 8, speed ** 2 / 64], -1)
        z = torch.cat([self.conv(latent.flatten(1, 2)), self.vel(v)], -1)
        o = self.head(z)
        clear = F.softplus(o[:, :2])            # metres along velocity, along forward
        return clear, o[:, 2:]                  # logits: hit@HORIZONS..., must_brake


def train_hazard(data_path: Path, run_dir: Path, device="cuda", epochs: int = 8, bs: int = 2048, lr: float = 1e-3):
    d = torch.load(data_path)
    n = len(d["vec"])
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    n_val = max(1000, n // 20)
    val, tr = perm[:n_val], perm[n_val:]
    model = HazardNet(d["latent"].shape[1], d["vec"].shape[1]).to(device)
    model(d["latent"][:2].float().to(device), d["vec"][:2].float().to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tgt_logit = torch.cat([d["hit"].float(), d["must_brake"].float()[:, None]], -1)
    pos = tgt_logit[tr].mean(0).clamp(1e-3, 1 - 1e-3)
    pos_weight = ((1 - pos) / pos).clamp(max=20).to(device)
    run_dir.mkdir(parents=True, exist_ok=True)

    def batch(idx):
        return (d["latent"][idx].float().to(device), d["vec"][idx].float().to(device),
                torch.stack([d["clear_vel"][idx], d["clear_fwd"][idx]], -1).float().to(device),
                tgt_logit[idx].to(device))

    def loss_fn(clear, logit, c_t, l_t):
        reg = F.smooth_l1_loss(torch.log1p(clear), torch.log1p(c_t))
        cls = F.binary_cross_entropy_with_logits(logit, l_t, pos_weight=pos_weight)
        return reg + cls, reg, cls

    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * math.ceil(len(tr) / bs))
    for ep in range(epochs):
        model.train()
        sh = tr[torch.randperm(len(tr))]
        for i in range(0, len(sh), bs):
            lat, vec, c_t, l_t = batch(sh[i:i + bs])
            l, _, _ = loss_fn(*model(lat, vec), c_t, l_t)
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
            sched.step()
        m = evaluate(model, d, val, device, tgt_logit)
        print(json.dumps({"epoch": ep, **m}), flush=True)
    torch.save({"model": model.state_dict(), "stack": d["latent"].shape[1], "vec_dim": d["vec"].shape[1],
                "meta": d["meta"]}, run_dir / "hazard.pt")
    (run_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    return model


@torch.no_grad()
def evaluate(model, d, idx, device, tgt_logit):
    model.eval()
    cs, ls = [], []
    for i in range(0, len(idx), 4096):
        j = idx[i:i + 4096]
        c, l = model(d["latent"][j].float().to(device), d["vec"][j].float().to(device))
        cs.append(c.cpu())
        ls.append(l.cpu())
    c, l = torch.cat(cs), torch.cat(ls)
    t = tgt_logit[idx]
    out = {"clear_vel_mae_m": round((c[:, 0] - d["clear_vel"][idx].float()).abs().mean().item(), 3)}
    names = [f"hit@{T}s" for T in HORIZONS] + ["must_brake"]
    for k, name in enumerate(names):
        p = l[:, k] > 0
        y = t[:, k] > 0.5
        tp = (p & y).sum().item()
        out[name] = {"recall": round(tp / max(1, y.sum().item()), 3), "precision": round(tp / max(1, p.sum().item()), 3),
                     "pos_rate": round(y.float().mean().item(), 3)}
    return out
