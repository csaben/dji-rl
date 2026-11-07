"""Fly a scene from the browser with the sim's own dynamics, collisions and gsplat camera.

    djirl serve --scene matrixcity-block --port 8965 --device cuda:1

One shared drone per server. The page polls /frame with the current stick values; each request
advances the sim by the wall-clock time since the last one (capped), renders, and returns a JPEG
plus the drone state in the X-State header. Modes: manual (your sticks), policy (a checkpoint
flies at the training rate, 10 Hz), random (raw untrained-policy-like noise)."""
from __future__ import annotations

import io
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch
from PIL import Image

from .dynamics import StickDynamics, StickParams
from .render import Renderer
from .scene import Scene

REPO = Path(__file__).resolve().parents[2]
PAGE = Path(__file__).with_name("viewer.html")


class Session:
    def __init__(self, scene: Scene, device: str):
        self.dev = torch.device(device)
        self.lock = threading.Lock()
        self.load_scene(scene)
        self.mode = "manual"
        self.policy = None
        self.policy_name = None
        self.env = None

    # ---------------------------------------------------------------- scene / reset
    def load_scene(self, scene: Scene):
        self.scene = scene.to(self.dev)
        g = self.scene.gaussians
        self.hd = Renderer(g, 960, 540, 82.0, chunk=1)
        self.sim = Renderer(g, 128, 96, 82.0, chunk=1)
        self.params = StickParams(dr=0.0, wind_std=0.0, delay_steps=6)  # 30 Hz manual: 6 steps = 200 ms
        self.dyn = StickDynamics(1, self.params, self.dev, control_dt=1 / 30, substeps=2)
        self.pitch_deg = -10.0
        self.collisions = 0
        self.in_contact = False
        self.flash = 0.0
        self.wp = 0
        self.reset()

    def reset(self):
        w = self.scene.waypoints
        p0 = w[0].clone()
        d = w[1 % len(w)] - p0 if len(w) > 1 else torch.tensor([1.0, 0, 0], device=self.dev)
        yaw = torch.atan2(d[1], d[0])
        self.dyn.reset(torch.tensor([0], device=self.dev), p0[None], yaw[None])
        self.wp = 1 % len(w)
        self.last_t = time.time()
        self.env = None

    # ---------------------------------------------------------------- policy
    def list_policies(self):
        return sorted(str(p.relative_to(REPO)) for p in (REPO / "runs").glob("*/policy.pt"))

    def set_mode(self, mode: str, ckpt: str | None):
        self.mode = mode
        self.env = None
        if mode == "policy" and ckpt:
            from .ppo import ActorCritic
            self.policy_name = ckpt
            self.policy = ckpt  # built lazily once the env exists (lazy layers need an obs)

    def _ensure_env(self):
        if self.env is not None:
            return
        from .env import BatchEnv, EnvConfig
        cfg = EnvConfig(num_envs=1, render_chunk=1, height_available=1.0)
        cfg.stick.dr, cfg.stick.wind_std = 0.0, 0.0
        self.env = BatchEnv(self.scene, cfg, device=str(self.dev))
        self.obs = self.env.reset()
        # start from where the manual drone is
        self.env.dyn.reset(torch.tensor([0], device=self.dev), self.dyn.pos.clone(), self.dyn.yaw.clone())
        self.env.target[0] = self.wp
        self.env.prev_dist[0] = (self.scene.waypoints[self.wp] - self.dyn.pos[0]).norm()
        self.env.latents = None
        self.obs = self.env._obs()
        if isinstance(self.policy, str):
            from .ppo import ActorCritic
            m = ActorCritic(cfg.frame_stack, self.env.VEC_DIM, self.env.PRIV_DIM).to(self.dev)
            m.value(self.obs), m.dist(self.obs)
            m.load_state_dict(torch.load(REPO / self.policy, map_location=self.dev)["model"])
            self.policy = m.eval()
        self.acc = 0.0

    # ---------------------------------------------------------------- stepping
    def step_manual(self, sticks, dt, noclip=False):
        """Same contact test as BatchEnv (swept sub-steps vs drone radius 0.25 m against the scene ESDF).
        Normal mode blocks and bounces; explore (noclip) passes through but still counts and flashes.
        A hit is counted once per contact (edge), not every frame spent touching."""
        n = max(1, min(6, round(dt * 30)))
        a = torch.tensor([sticks], device=self.dev, dtype=torch.float32)
        touching = False
        for _ in range(n):
            prev = self.dyn.pos.clone()
            subs = self.dyn.step(a)
            sdf = self.scene.esdf.query(torch.cat(subs)).min()
            hit = bool(sdf < 0.25) or not self.scene.esdf.in_bounds(self.dyn.pos).item()
            if hit:
                touching = True
                if not self.in_contact:
                    self.collisions += 1
                    self.in_contact = True
                self.flash = time.time()
                if not noclip:
                    self.dyn.pos = prev
                    self.dyn.vel = -0.3 * self.dyn.vel  # bounce back a little
                    break
        if not touching:
            self.in_contact = False

    def step_auto(self, dt):
        self._ensure_env()
        self.acc += dt
        steps = 0
        while self.acc >= 0.1 and steps < 3:  # policy/env rate is 10 Hz
            self.acc -= 0.1
            steps += 1
            if self.mode == "policy" and not isinstance(self.policy, (str, type(None))):
                a = self.policy.act(self.obs, deterministic=True)
            else:
                a = (torch.randn(1, 4, device=self.dev) * 0.6).clamp(-1, 1)
                a[:, 0] = a[:, 0].abs()
            self.obs, r, done, info = self.env.step(a)
            if info["collided"].any():
                self.collisions += 1
                self.flash = time.time()
        d = self.env.dyn
        self.dyn.pos, self.dyn.vel, self.dyn.yaw, self.dyn.yaw_rate = d.pos.clone(), d.vel.clone(), d.yaw.clone(), d.yaw_rate.clone()
        self.wp = int(self.env.target[0])

    def start_replay(self, traj: dict):
        if traj["scene"] != self.scene.name:
            from .scene import load_scene
            self.load_scene(load_scene(REPO / "data" / "splats" / traj["scene"]))
        self.replay = traj
        self.replay_t0 = time.time()
        self.replay_fixed = None
        self.mode = "replay"
        self.env = None

    def _replay_pose(self, t_override):
        tr = self.replay["traj"]
        dur = tr[-1]["t"]
        if t_override is not None:
            self.replay_fixed = t_override
        t = self.replay_fixed if self.replay_fixed is not None else (time.time() - self.replay_t0) % (dur + 2.0)
        t = min(max(t, 0.0), dur)
        i = min(int(t / 0.1), len(tr) - 1)
        j = min(i + 1, len(tr) - 1)
        w = (t - tr[i]["t"]) / 0.1 if j > i else 0.0
        lerp = lambda k: tr[i][k] * (1 - w) + tr[j][k] * w
        dy = (tr[j]["yaw"] - tr[i]["yaw"] + math.pi) % (2 * math.pi) - math.pi
        pos = torch.tensor([[lerp("x"), lerp("y"), lerp("z")]], device=self.dev)
        vel = torch.tensor([[lerp("vx"), lerp("vy"), lerp("vz")]], device=self.dev)
        yaw = torch.tensor([tr[i]["yaw"] + w * dy], device=self.dev)
        return pos, vel, yaw, t, dur

    def frame(self, sticks, view: str, pitch_deg: float, noclip: bool = False, cam: str = "fpv", t_override=None):
        with self.lock:
            now = time.time()
            dt = min(now - self.last_t, 0.2)
            self.last_t = now
            self.pitch_deg = pitch_deg
            replay_info = None
            if self.mode == "replay":
                pos, vel, yaw, t, dur = self._replay_pose(t_override)
                self.dyn.pos, self.dyn.vel, self.dyn.yaw = pos, vel, yaw
                replay_info = {"id": self.replay["id"], "t": round(t, 2), "duration": dur,
                               "paused": self.replay_fixed is not None}
            elif self.mode == "manual":
                self.step_manual(sticks, dt, noclip)
                w = self.scene.waypoints
                if (w[self.wp] - self.dyn.pos[0]).norm() < 1.5:
                    self.wp = (self.wp + 1) % len(w)
            else:
                self.step_auto(dt)
            pos, yaw = self.dyn.pos, self.dyn.yaw
            pitch = torch.full_like(yaw, math.radians(pitch_deg))
            r = self.hd if view == "hd" else self.sim
            cam_pos = pos
            if cam == "chase":  # behind and above, looking at the drone
                back = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], -1)
                cam_pos = pos - 6.0 * back + torch.tensor([0.0, 0.0, 3.0], device=self.dev)
                pitch = torch.full_like(yaw, -math.atan2(3.0, 6.0))
            img = r.render(cam_pos, yaw, pitch)[0, ..., :3]
            if time.time() - self.flash < 0.25:
                img = img * 0.5 + torch.tensor([0.5, 0, 0], device=self.dev)
            im = Image.fromarray((img.clamp(0, 1).cpu().numpy() * 255).astype("uint8"))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=85 if view == "hd" else 95)
            p = pos[0]
            clearance = self.scene.esdf.query(pos).item()
            down = torch.linspace(0, 30, 121, device=self.dev)
            col = self.scene.esdf.query(p[None] - torch.stack([torch.zeros_like(down)] * 2 + [down], -1))
            hit = (col < 0.1).nonzero()
            agl = down[hit[0, 0]].item() if len(hit) else None
            wpt = self.scene.waypoints[self.wp]
            state = {
                "pos": [round(x, 2) for x in p.tolist()], "yaw_deg": round(math.degrees(yaw.item()), 1),
                "speed": round(self.dyn.vel[0].norm().item(), 2),
                "vel_body": [round(x, 2) for x in self.dyn.body_velocity()[0].tolist()],
                "agl": None if agl is None else round(agl, 2), "clearance": round(clearance, 2),
                "collisions": self.collisions, "mode": self.mode, "waypoint": self.wp,
                "wp_dist": round((wpt - p).norm().item(), 1), "scene": self.scene.name,
                "replay": replay_info, "cam": cam,
            }
            return buf.getvalue(), state


def serve(scene: Scene, host="0.0.0.0", port=8965, device="cuda:1"):
    from .sim_api import SimAPI
    sess = Session(scene, device)
    api = SimAPI(device)
    splats = REPO / "data" / "splats"

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype: str, headers=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/":
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if u.path.startswith("/api/sim/"):
                return self._api_get(u.path[len("/api/sim/"):], q)
            if u.path == "/replay":
                tid = q.get("id", "")
                if tid not in api.trajs:
                    return self._send(404, b"unknown trajectory id", "text/plain")
                with sess.lock:
                    if q.get("t") is not None and sess.mode == "replay" and sess.replay["id"] == tid:
                        sess.replay_fixed = None if q["t"] == "play" else float(q["t"])
                        if q["t"] == "play":
                            sess.replay_t0 = time.time() - float(q.get("from", 0))
                    else:
                        sess.start_replay(api.trajs[tid])
                return self._send(200, b"ok", "text/plain")
            if u.path == "/frame":
                sticks = [float(q.get(k, 0)) for k in ("f", "r", "u", "y")]
                jpg, st = sess.frame(sticks, q.get("view", "hd"), float(q.get("pitch", -10)), q.get("noclip") == "1",
                                     cam=q.get("cam", "fpv"))
                return self._send(200, jpg, "image/jpeg", {"X-State": json.dumps(st)})
            if u.path == "/info":
                scenes = sorted(d.name for d in splats.iterdir() if (d / "scene.json").exists())
                return self._send(200, json.dumps({"scenes": scenes, "scene": sess.scene.name,
                                                   "policies": sess.list_policies(), "mode": sess.mode,
                                                   "waypoints": sess.scene.waypoints.tolist()}).encode(),
                                  "application/json")
            if u.path == "/reset":
                with sess.lock:
                    sess.reset()
                return self._send(200, b"ok", "text/plain")
            if u.path == "/mode":
                with sess.lock:
                    sess.reset() if q.get("reset") else None
                    sess.set_mode(q.get("mode", "manual"), q.get("ckpt"))
                return self._send(200, b"ok", "text/plain")
            if u.path == "/scene":
                from .scene import load_scene
                name = q.get("name", "")
                if not (splats / name / "scene.json").exists() or "/" in name:
                    return self._send(404, b"no such scene", "text/plain")
                with sess.lock:
                    sess.load_scene(load_scene(splats / name))
                    sess.set_mode("manual", None)
                return self._send(200, b"ok", "text/plain")
            return self._send(404, b"not found", "text/plain")

        def _json(self, code, obj):
            return self._send(code, json.dumps(obj).encode(), "application/json")

        def _api_get(self, what, q):
            try:
                if what == "scenes":
                    return self._json(200, api.list_scenes())
                name, px = q.get("scene", ""), int(q.get("px", 1024))
                if what == "topdown":
                    meta = api.topdown_meta(name, px)
                    return self._send(200, api.topdown_png(name, px), "image/png",
                                      {"X-Bounds": json.dumps(meta), "Access-Control-Expose-Headers": "X-Bounds"})
                if what == "topdown.json":
                    return self._json(200, api.topdown_meta(name, px))
                if what == "occupancy" and q.get("mode") == "column":
                    return self._send(200, api.column_png(name, float(q.get("alt", 5)), px), "image/png",
                                      {"X-Bounds": json.dumps(api.topdown_meta(name, px))})
                if what == "occupancy":
                    return self._send(200, api.occupancy_png(name, float(q.get("alt", 5)), float(q.get("clear", 1.5)), px),
                                      "image/png", {"X-Bounds": json.dumps(api.topdown_meta(name, px))})
                if what == "traj":
                    t = api.trajs.get(q.get("id", ""))
                    return self._json(200 if t else 404, t or {"error": "unknown id"})
                return self._json(404, {"error": f"unknown endpoint {what}"})
            except KeyError as e:
                return self._json(404, {"error": str(e)})

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != "/api/sim/fly":
                return self._json(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                with sess.lock:  # one GPU, keep fly and frame rendering from interleaving
                    out = api.fly(req)
                return self._json(200, out)
            except KeyError as e:
                return self._json(400, {"ok": False, "reason": f"missing/unknown field {e}"})
            except Exception as e:  # noqa: BLE001
                return self._json(500, {"ok": False, "reason": f"{type(e).__name__}: {e}"})

    srv = ThreadingHTTPServer((host, port), H)
    print(f"djirl viewer: http://{host}:{port}  scene={scene.name} device={device}", flush=True)
    srv.serve_forever()
