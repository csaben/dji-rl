"""Live pilot server: the phone (Flight Lab) flies the planned mission with its GPS mission runner and
streams camera frames + telemetry here; this server runs the hazard classifier on every frame and,
when something unmapped is in the way, tells the phone to hand the sticks to the learned policy,
streams the policy's stick commands, and tells it when (and where on the route) to resume the mission.

    .venv/bin/djirl pilot-serve --ckpt runs/route-wlref-2/policy.pt --hazard-ckpt runs/hazard-wlref/hazard.pt

Protocol: docs/pilot-protocol.md. The decision logic is the one evaluated in the sim (handoff.py,
net trigger): take over when p(must_brake) > thr or the predicted clearance along velocity is under
stopping distance + lookahead; hand back after >= 1 s under the policy, 1 s with no hazard, the
camera facing the route carrot with predicted forward clearance, and the map line to the carrot free.

Frames: mission scene coordinates (sim ENU) <-> real metres relative to the takeoff spot, with the
mission's alignment: real = R(yaw_trim) * (scene - A) * scale_corr (same as the planner's toReal).
"""
from __future__ import annotations

import asyncio
import io
import json
import math
import struct
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .hazard import BrakeModel, HazardNet

REPO = Path(__file__).resolve().parents[2]
PLANNER_MISSIONS = REPO.parent / "dji-planner" / "missions"
CAM_W, CAM_H, CAM_ASPECT = 128, 96, 4 / 3       # policy camera: 128x96, 82 deg hfov, gimbal -10 deg
CARROT_M, STEP_M = 6.0, 0.25
V_H_MAX, V_UP_MAX, V_DOWN_MAX, YAW_RATE_MAX = 8.0, 3.0, 2.5, 1.6   # StickParams (normalised stick -> units)


def _yaw_enu(compass_deg: float) -> float:
    """DJI compass yaw (0 = north, clockwise) -> ENU yaw (0 = east, counter-clockwise), radians."""
    return math.radians(90.0 - compass_deg)


class MissionFrame:
    """Scene <-> real conversion and the route polyline (scene coords) the policy's carrot follows."""

    def __init__(self, m: dict):
        self.m = m
        sr = m.get("sim_request") or {}
        self.scene = (m.get("sim_preview") or {}).get("scene") or sr.get("scene")
        if not self.scene:
            raise ValueError("mission has no sim_preview.scene / sim_request.scene (launch it from the sim planner)")
        al = m.get("alignment") or {}
        self.k = float(al.get("scale_corr", 1.0))
        self.t = math.radians(float(al.get("yaw_trim_deg", 0.0)))
        self.A = np.array([float(sr["a"]["x"]), float(sr["a"]["y"])])
        route, rel = m["route"], m["route_rel"]
        if len(route) != len(rel):
            raise ValueError("route and route_rel lengths differ")
        # scene path: xy from route_rel (inverse of toReal); z = the planned scene altitude (route[i].alt_m)
        xy = np.array([self.to_scene_xy(r["e_m"], r["n_m"]) for r in rel])
        z = np.array([float(r["alt_m"]) for r in route])
        self.ground_a = float(np.median(z - np.array([float(r["up_m"]) for r in rel]) / self.k))
        self.verts = np.concatenate([xy, z[:, None]], 1)
        pts, vidx = [self.verts[0]], [0]
        for i in range(1, len(self.verts)):
            a, b = self.verts[i - 1], self.verts[i]
            n = max(1, int(math.ceil(np.linalg.norm(b[:2] - a[:2]) / STEP_M)))
            for j in range(1, n + 1):
                pts.append(a + (b - a) * j / n)
            vidx.append(len(pts) - 1)
        self.path = np.array(pts)                       # dense [P,3], STEP_M apart (horizontally)
        self.vert_at = np.array(vidx)                   # dense index of each route vertex
        self.idx = 0

    def to_scene_xy(self, e, n):
        c, s = math.cos(-self.t), math.sin(-self.t)
        return self.A + np.array([c * e - s * n, s * e + c * n]) / self.k

    def to_real(self, p):
        d = (np.asarray(p[:2]) - self.A) * self.k
        c, s = math.cos(self.t), math.sin(self.t)
        return {"e_m": round(float(c * d[0] - s * d[1]), 2), "n_m": round(float(s * d[0] + c * d[1]), 2),
                "up_m": round(float((p[2] - self.ground_a) * self.k), 2)}

    def pose(self, h: dict):
        """header -> scene pos [3], ENU yaw (scene frame), velocity [3] ENU m/s (metric, real)."""
        r = h["pos_rel"]
        xy = self.to_scene_xy(float(r["e"]), float(r["n"]))
        pos = np.array([xy[0], xy[1], float(r["up"]) / self.k + self.ground_a])
        yaw = _yaw_enu(float(h["yaw_deg"])) - self.t
        vn, ve, vd = (float(v) for v in h["vel_ned"])
        c, s = math.cos(-self.t), math.sin(-self.t)
        vel = np.array([c * ve - s * vn, s * ve + c * vn, -vd])
        return pos, yaw, vel

    def progress(self, pos):
        """Monotone nearest dense index (forward window); returns (idx, xtrack_m)."""
        lo, hi = max(0, self.idx - 8), min(len(self.path), self.idx + 200)
        d = np.linalg.norm(self.path[lo:hi, :2] - pos[:2], axis=1)
        j = int(d.argmin())
        self.idx = max(self.idx, lo + j)
        return self.idx, float(d[j])

    def carrot(self):
        ci = min(self.idx + int(CARROT_M / STEP_M), len(self.path) - 1)
        return self.path[ci], ci

    def rejoin(self, ci):
        """Next route vertex at/after the carrot: where the mission runner should resume."""
        k = int(np.searchsorted(self.vert_at, ci))
        k = min(k, len(self.verts) - 1)
        return {"route_idx": k, **self.to_real(self.verts[k])}


class PilotCore:
    def __init__(self, ckpt, hazard_ckpt, device="cuda:1", thr=0.5, lookahead=4.0, cruise=None,
                 min_policy_s=1.0, quiet_s=1.0, max_policy_s=20.0, speed_cap=4.0, log_dir=None):
        from .encoder import LatentEncoder
        from .ppo import ActorCritic
        self.dev = torch.device(device)
        self.enc = LatentEncoder(self.dev)
        d = torch.load(hazard_ckpt, map_location=self.dev)
        self.haz = HazardNet(d["stack"], d["vec_dim"]).to(self.dev)
        self.stack, self.vec_dim = d["stack"], d["vec_dim"]
        z0 = torch.zeros(2, self.stack, 4, CAM_H // 8, CAM_W // 8, device=self.dev)
        v0 = torch.zeros(2, self.vec_dim, device=self.dev)
        self.haz(z0, v0)
        self.haz.load_state_dict(d["model"])
        self.haz.eval()
        sd = torch.load(ckpt, map_location=self.dev)["model"]
        priv_dim = sd["critic.0.weight"].shape[1] - 256 - self.vec_dim
        self.pol = ActorCritic(self.stack, self.vec_dim, priv_dim).to(self.dev)
        o = {"latent": z0, "vec": v0, "priv": torch.zeros(2, priv_dim, device=self.dev)}
        self.pol.value(o), self.pol.dist(o)
        self.pol.load_state_dict(sd)
        self.pol.eval()
        self.brake = BrakeModel()
        self.thr, self.look, self.cruise = thr, lookahead, cruise
        self.min_policy_s, self.quiet_s, self.max_policy_s, self.speed_cap = min_policy_s, quiet_s, max_policy_s, speed_cap
        self.scenes = {}
        self.log_dir = Path(log_dir) if log_dir else None
        self.lock = threading.Lock()
        self.mf = None

    # ---------------------------------------------------------------- session
    def start(self, mission: dict, mode: str = "shadow"):
        from .scene import load_scene
        mf = MissionFrame(mission)
        if mf.scene not in self.scenes:
            s = load_scene(REPO / "data" / "splats" / mf.scene)
            self.scenes[mf.scene] = s.esdf.to(self.dev)
        self.esdf = self.scenes[mf.scene]
        self.mf, self.mode = mf, mode
        self.cruise_now = self.cruise or float((mission.get("limits") or {}).get("cruise_speed_mps", 3.0))
        self.latents = None
        self.flying = "mission"
        self.t_mode = self.quiet = 0.0
        self.last_t = None
        self.last_yaw = None
        self.yaw_rate = 0.0
        self.last_action = np.zeros(4, dtype=np.float32)
        self.takeovers = 0
        self.sess = None
        if self.log_dir:
            self.sess = self.log_dir / time.strftime("%Y%m%d_%H%M%S")
            (self.sess / "frames").mkdir(parents=True, exist_ok=True)
            (self.sess / "mission.json").write_text(json.dumps(mission))
            self.logf = (self.sess / "log.jsonl").open("a")
        warn = []
        if abs(float(mission.get("fpv_gimbal_deg", -10)) + 10) > 5 and mission.get("flight_style") == "fpv":
            warn.append("policy was trained with the gimbal at -10 deg")
        if mf.k != 1.0:
            warn.append(f"scale_corr {mf.k:g}: scene is scaled to real metres for the route; velocities stay metric")
        return {"type": "ready", "scene": mf.scene, "route_n": len(mf.verts), "path_m": round(len(mf.path) * STEP_M, 1),
                "mode": mode, "lookahead_m": self.look, "thr": self.thr, "warnings": warn,
                "log": str(self.sess) if self.sess else None}

    # ---------------------------------------------------------------- per frame
    def _image(self, jpeg: bytes):
        im = Image.open(io.BytesIO(jpeg)).convert("RGB")
        w, h = im.size
        if abs(w / h - CAM_ASPECT) > 0.02:              # centre-crop to 4:3 (narrows the hfov on 16:9 streams)
            if w / h > CAM_ASPECT:
                cw = int(round(h * CAM_ASPECT)); im = im.crop(((w - cw) // 2, 0, (w - cw) // 2 + cw, h))
            else:
                ch = int(round(w / CAM_ASPECT)); im = im.crop((0, (h - ch) // 2, w, (h - ch) // 2 + ch))
        im = im.resize((CAM_W, CAM_H), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).to(self.dev)
        return x[None], (w, h)

    def _map_los(self, a, b, clear):
        n = int(max(2, math.ceil(np.linalg.norm(b - a) / 0.25) + 1))
        pts = torch.tensor(np.linspace(a, b, n), dtype=torch.float32, device=self.dev)
        return bool((self.esdf.query(pts) > clear).all())

    @torch.no_grad()
    def frame(self, h: dict, jpeg: bytes) -> dict:
        t0 = time.time()
        mf = self.mf
        r = h.get("pos_rel") or {}
        if any(h.get(k) is None for k in ("vel_ned", "yaw_deg")) or any(r.get(k) is None for k in ("e", "n", "up")):
            return {"type": "cmd", "seq": h.get("seq"), "t_ms": h.get("t_ms"), "idle": True, "want": "mission",
                    "hazard": False, "reason": "no pose (no active mission)"}
        pos, yaw, vel = mf.pose(h)
        t = float(h["t_ms"]) / 1000.0
        if self.last_t is not None and t > self.last_t:
            dy = (yaw - self.last_yaw + math.pi) % (2 * math.pi) - math.pi
            self.yaw_rate = 0.5 * self.yaw_rate + 0.5 * dy / (t - self.last_t)
        dt = (t - self.last_t) if self.last_t is not None else 0.1
        self.last_t, self.last_yaw = t, yaw
        rgb, wh = self._image(jpeg)
        z = self.enc(rgb)
        self.latents = z[:, None].repeat(1, self.stack, 1, 1, 1) if self.latents is None else \
            torch.cat([self.latents[:, 1:], z[:, None]], 1)
        idx, xtrack = mf.progress(pos)
        goal, ci = mf.carrot()
        g = goal - pos
        dist = float(np.linalg.norm(g))
        c, s = math.cos(yaw), math.sin(yaw)
        body = lambda v: np.array([c * v[0] + s * v[1], s * v[0] - c * v[1], v[2]])     # ENU -> body FRU
        gh = h.get("ground_h_m")
        height = float(gh) if gh is not None and float(gh) >= 0 else -1.0
        cv = h.get("cmd_vel")
        if cv is not None and self.flying != "policy":       # MSDK velocity-mode command -> normalised sticks
            up = float(cv.get("up_mps", 0))
            self.last_action = np.array([float(cv.get("fwd_mps", 0)) / V_H_MAX, float(cv.get("right_mps", 0)) / V_H_MAX,
                                         up / (V_UP_MAX if up > 0 else V_DOWN_MAX),
                                         math.radians(float(cv.get("yaw_rate_dps", 0))) / YAW_RATE_MAX],
                                        dtype=np.float32).clip(-1, 1)
        elif h.get("sticks_now") is not None and self.flying != "policy":
            self.last_action = np.asarray(h["sticks_now"], dtype=np.float32).clip(-1, 1)
        elif self.flying != "policy":                       # mission runner: estimate its sticks from velocity
            bv = body(vel)
            self.last_action = np.array([bv[0] / V_H_MAX, bv[1] / V_H_MAX, bv[2] / V_UP_MAX,
                                         -self.yaw_rate / YAW_RATE_MAX], dtype=np.float32).clip(-1, 1)
        vec = np.concatenate([body(g / max(dist, 1e-3)), [min(dist / 20, 2)], body(vel) / 8,
                              [self.yaw_rate / 1.6], [height / 10.0], self.last_action]).astype(np.float32)
        vec_t = torch.tensor(vec, device=self.dev)[None]
        clear, logit = self.haz(self.latents, vec_t)
        p_brake = float(torch.sigmoid(logit[0, -1]))
        clear_v, clear_f = float(clear[0, 0]), float(clear[0, 1])
        speed = float(np.linalg.norm(vel))
        stop = float(self.brake.stopping_distance(torch.tensor(speed)))
        hazard = p_brake > self.thr or clear_v < stop + self.look
        # mode machine (same rules as handoff.run)
        self.quiet = 0.0 if hazard else self.quiet + dt
        self.t_mode += dt
        fwd = np.array([c, s, 0.0])
        to_c = g / max(dist, 1e-3)
        need_c = float(self.brake.stopping_distance(torch.tensor(self.cruise_now))) + 1.0
        clear_c = float(fwd @ to_c) > math.cos(math.radians(25)) and clear_f >= need_c
        reported = h.get("flying_by", self.flying)
        if reported in ("rc", "hover", "manual") and self.flying == "policy":
            self.flying, self.t_mode = "mission", 0.0            # phone stopped following us: stop driving
        reason = None
        if self.flying == "mission" and hazard and reported == "mission":
            self.flying, self.t_mode, reason = "policy", 0.0, ("p_brake" if p_brake > self.thr else "clearance")
            self.takeovers += 1
        elif self.flying == "policy" and self.mode == "shadow":
            # the phone keeps flying the mission and never takes our sticks: "policy" here only means
            # "would be avoiding", so release once the hazard has been quiet for quiet_s
            if self.quiet >= self.quiet_s:
                self.flying, self.t_mode, reason = "mission", 0.0, "clear"
        elif self.flying == "policy":
            if (self.t_mode >= self.min_policy_s and self.quiet >= self.quiet_s and clear_c
                    and self._map_los(pos, goal, 0.25 + 0.5)):
                self.flying, self.t_mode, reason = "mission", 0.0, "clear"
            elif self.t_mode > self.max_policy_s:
                self.flying, self.t_mode, reason = "hover", 0.0, "policy_timeout"
        out = {"type": "cmd", "seq": h.get("seq"), "t_ms": h["t_ms"], "hazard": bool(hazard), "p_brake": round(p_brake, 3),
               "clear_m": round(clear_v, 2), "clear_fwd_m": round(clear_f, 2), "stop_m": round(stop + self.look, 2),
               "want": self.flying, "route_progress_m": round(idx * STEP_M, 1), "xtrack_m": round(xtrack, 2)}
        if reason:
            out["reason"] = reason
        if self.flying == "policy":
            obs = {"latent": self.latents, "vec": vec_t}
            a = self.pol.act(obs, deterministic=True)[0].clamp(-1, 1).cpu().numpy()
            self.last_action = a.astype(np.float32)
            f, r = a[0] * V_H_MAX, a[1] * V_H_MAX
            n = math.hypot(f, r)
            if n > self.speed_cap:
                f, r = f * self.speed_cap / n, r * self.speed_cap / n
            out["sticks"] = {"fwd_mps": round(float(f), 3), "right_mps": round(float(r), 3),
                             "up_mps": round(float(a[2] * (V_UP_MAX if a[2] > 0 else V_DOWN_MAX)), 3),
                             "yaw_rate_dps": round(float(math.degrees(a[3] * YAW_RATE_MAX)), 2),   # + = clockwise
                             "norm": [round(float(v), 4) for v in a]}
        if reason == "clear":
            out["rejoin"] = mf.rejoin(ci)
        out["server_ms"] = round((time.time() - t0) * 1000, 1)
        if self.sess:
            seq = h.get("seq", 0)
            if seq is not None and (self.flying != "mission" or hazard or int(seq) % 5 == 0):
                (self.sess / "frames" / f"{int(seq):06d}.jpg").write_bytes(jpeg)
            self.logf.write(json.dumps({"h": h, "out": out, "pos_scene": [round(float(v), 3) for v in pos],
                                        "yaw_scene": round(yaw, 4), "img": list(wh)}) + "\n")
            self.logf.flush()
        return out


def load_mission(spec: str) -> dict:
    p = PLANNER_MISSIONS / "latest.json" if spec in (None, "", "latest") else Path(spec)
    if not p.exists() and (PLANNER_MISSIONS / spec).exists():
        p = PLANNER_MISSIONS / spec
    return json.loads(p.read_text())


def serve(core: PilotCore, host="100.80.98.4", port=8971):
    """WebSocket server, one active session; newest frame wins (stale frames are dropped)."""
    import websockets

    async def handler(ws):
        latest = {"msg": None}
        ev = asyncio.Event()
        loop = asyncio.get_running_loop()

        async def worker():
            while True:
                await ev.wait()
                ev.clear()
                msg, latest["msg"] = latest["msg"], None
                if msg is None:
                    continue
                n = struct.unpack(">I", msg[:4])[0]
                h = json.loads(msg[4:4 + n])
                try:
                    out = await loop.run_in_executor(None, lambda: _locked(core.frame, h, msg[4 + n:]))
                except Exception as e:  # noqa: BLE001
                    out = {"type": "error", "seq": h.get("seq"), "error": f"{type(e).__name__}: {e}", "want": "hover"}
                await ws.send(json.dumps(out))

        def _locked(f, *a):
            with core.lock:
                return f(*a)

        task = None
        try:
            async for msg in ws:
                if isinstance(msg, str):
                    m = json.loads(msg)
                    if m.get("type") == "hello":
                        try:
                            mission = m["mission"] if isinstance(m.get("mission"), dict) else load_mission(m.get("mission"))
                            r = await loop.run_in_executor(None, lambda: _locked(core.start, mission, m.get("mode", "shadow")))
                        except Exception as e:  # noqa: BLE001
                            r = {"type": "error", "error": f"{type(e).__name__}: {e}"}
                        await ws.send(json.dumps(r))
                        if task is None and r.get("type") == "ready":
                            task = asyncio.create_task(worker())
                    elif m.get("type") == "ping":
                        await ws.send(json.dumps({"type": "pong", "t_ms": m.get("t_ms"), "server_t": time.time()}))
                    continue
                if core.mf is None:
                    await ws.send(json.dumps({"type": "error", "error": "send hello first", "want": "hover"}))
                    continue
                latest["msg"] = msg
                ev.set()
        finally:
            if task:
                task.cancel()

    async def main():
        async with websockets.serve(handler, host, port, max_size=8 * 1024 * 1024, compression=None):
            print(f"pilot server on ws://{host}:{port}/", flush=True)
            await asyncio.Future()

    asyncio.run(main())
