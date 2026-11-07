"""Classical (no learning) A->B flight in a scene: plan on the ESDF, track with a pure-pursuit
controller in the env's virtual-stick action space, simulate with StickDynamics + ESDF contacts.

Endpoints (wired up in viewer.serve):
  GET  /api/sim/scenes
  GET  /api/sim/topdown?scene=&px=1024         PNG, north up; sidecar /api/sim/topdown.json?scene=&px=
  GET  /api/sim/occupancy?scene=&alt=&clear=&px=  PNG mask (255 = blocked: ESDF < clear at alt)
  POST /api/sim/fly {scene, a:{x,y}, b:{x,y}, alt_m, speed_mps=3, clear_m=1.5, takeoff_land=true, wind=false}
All coordinates are scene ENU metres (x east, y north, z up, ground ~ z=0).
"""
from __future__ import annotations

import heapq
import io
import json
import math
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .dynamics import StickDynamics, StickParams
from .scene import Scene, load_scene

REPO = Path(__file__).resolve().parents[2]
SPLATS = REPO / "data" / "splats"
DRONE_R = 0.25
GROUND_R = 0.4      # m, neighbourhood for the local ground height (max of 3x3 samples)
GROUND_MAX = 2.0    # m; a takeoff/landing surface this far above the terrain map is a roof/canopy
TERRAIN_RES, TERRAIN_WIN, TERRAIN_PCT = 1.0, 15, 20  # terrain = 20th percentile of 1 m first-surface heights over 15 m
COLUMN_START = 1.0  # m above ground where the takeoff/landing column check starts (ignores grass / low clutter)


class SimAPI:
    def __init__(self, device):
        self.dev = torch.device(device)
        self.scenes: dict[str, Scene] = {}
        self.cache: dict = {}
        self.trajs: OrderedDict[str, dict] = OrderedDict()

    # ------------------------------------------------------------ scenes
    def scene(self, name: str) -> Scene:
        if "/" in name or not (SPLATS / name / "scene.json").exists():
            raise KeyError(f"no scene {name!r}")
        if name not in self.scenes:
            if len(self.scenes) >= 3:  # keep GPU memory bounded
                self.scenes.pop(next(iter(self.scenes)))
            self.scenes[name] = load_scene(SPLATS / name).to(self.dev)
        return self.scenes[name]

    def bounds(self, name: str) -> dict:
        e = self.scene(name).esdf
        lo, hi = e.origin.tolist(), e.upper.tolist()
        return {"x0": round(lo[0], 2), "y0": round(lo[1], 2), "x1": round(hi[0], 2), "y1": round(hi[1], 2),
                "zmax": round(hi[2], 2)}

    def list_scenes(self) -> list:
        out = []
        for d in sorted(p for p in SPLATS.iterdir() if (p / "scene.json").exists()):
            geo = json.loads((d / "geo.json").read_text()) if (d / "geo.json").exists() else None
            try:
                b = self.bounds(d.name) if d.name in self.scenes else self._bounds_from_disk(d)
            except Exception:  # noqa: BLE001
                b = None
            out.append({"name": d.name, "bounds": b, "geo": geo})
        return out

    @staticmethod
    def _bounds_from_disk(d: Path) -> dict:
        z = np.load(d / "esdf.npz")
        lo = z["origin"]
        hi = lo + float(z["voxel"]) * (np.array(z["sdf"].shape) - 1)
        return {"x0": round(float(lo[0]), 2), "y0": round(float(lo[1]), 2), "x1": round(float(hi[0]), 2),
                "y1": round(float(hi[1]), 2), "zmax": round(float(hi[2]), 2)}

    @staticmethod
    def q(s, P):
        """ESDF distance, treating anything above the collision grid's ceiling as free air
        (the grid is capped at --max-height when imported; above it there is only sky)."""
        d = s.esdf.query(P)
        top = float(s.esdf.upper[2])
        return torch.where(P[..., 2] > top - 0.05, torch.clamp(d, min=P[..., 2] - top + 5.0), d)

    # ------------------------------------------------------------ maps
    def _grid(self, name, px):
        b = self.bounds(name)
        W = int(px)
        H = max(1, round(W * (b["y1"] - b["y0"]) / (b["x1"] - b["x0"])))
        return b, W, H

    def topdown_meta(self, name, px=1024) -> dict:
        b, W, H = self._grid(name, px)
        return {"x0": b["x0"], "y0": b["y0"], "x1": b["x1"], "y1": b["y1"], "px_w": W, "px_h": H,
                "note": "pixel (u,v) centre -> x = x0 + (u+0.5)*(x1-x0)/px_w, y = y1 - (v+0.5)*(y1-y0)/px_h (north up)"}

    @torch.no_grad()
    def topdown_png(self, name, px=1024) -> bytes:
        key = ("top", name, px)
        if key in self.cache:
            return self.cache[key]
        from gsplat import rasterization
        s = self.scene(name)
        g = s.gaussians
        b, W, H = self._grid(name, px)
        # drop sky/cloud gaussians that would cover the map: high ones and very large soft ones
        keep = (g.means[:, 2] < min(b["zmax"], 12.0)) & (g.scales.max(-1).values < 1.0)
        m, q, sc, op, sh = (t[keep] for t in (g.means, g.quats, g.scales, g.opacities, g.sh))
        cx, cy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
        R = torch.tensor([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]], device=self.dev)  # right=E, down=S, fwd=-Up
        p = torch.tensor([cx, cy, 100.0], device=self.dev)
        V = torch.eye(4, device=self.dev)
        V[:3, :3], V[:3, 3] = R, -R @ p
        f = W / (b["x1"] - b["x0"])
        K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], device=self.dev)
        img, alpha, _ = rasterization(m, q, sc, op, sh, V[None], K[None], W, H, sh_degree=g.sh_degree,
                                      camera_model="ortho", near_plane=0.01, far_plane=1000.0)
        img = img[0, ..., :3] + (1 - alpha[0]) * torch.tensor([0.15, 0.15, 0.17], device=self.dev)
        buf = io.BytesIO()
        Image.fromarray((img.clamp(0, 1).cpu().numpy() * 255).astype("uint8")).save(buf, "PNG")
        self.cache[key] = buf.getvalue()
        return self.cache[key]

    def _pixel_xy(self, name, px):
        b, W, H = self._grid(name, px)
        xs = b["x0"] + (torch.arange(W, device=self.dev) + 0.5) * (b["x1"] - b["x0"]) / W
        ys = b["y1"] - (torch.arange(H, device=self.dev) + 0.5) * (b["y1"] - b["y0"]) / H
        Y, X = torch.meshgrid(ys, xs, indexing="ij")
        return X, Y

    @torch.no_grad()
    def occupancy_png(self, name, alt, clear, px=1024) -> bytes:
        s = self.scene(name)
        X, Y = self._pixel_xy(name, px)
        P = torch.stack([X, Y, torch.full_like(X, float(alt))], -1)
        blocked = self.q(s, P.reshape(-1, 3)).reshape(X.shape) < float(clear)
        buf = io.BytesIO()
        Image.fromarray(blocked.cpu().numpy().astype("uint8") * 255).save(buf, "PNG")
        return buf.getvalue()

    @torch.no_grad()
    def _ground(self, s, x, y, alt):
        """First ESDF<0.05 scanning down from alt (0 if nothing), per point. x,y: [n]."""
        zs = torch.arange(alt, -3.0, -0.1, device=self.dev)
        n = len(x)
        P = torch.stack([x[:, None].expand(n, len(zs)), y[:, None].expand(n, len(zs)), zs[None].expand(n, -1)], -1)
        hit = self.q(s, P.reshape(-1, 3)).reshape(n, -1) < 0.05
        return torch.where(hit.any(1), zs[hit.float().argmax(1)], torch.zeros(n, device=self.dev))

    @torch.no_grad()
    def _terrain_at(self, s, x, y):
        """Robust terrain height: first-surface heights on a 1 m grid (scanned from the grid top),
        20th-percentile filtered over 15 m and smoothed; bilinear lookup. Cached per scene."""
        key = ("terrain", s.name)
        if key not in self.cache:
            from scipy.ndimage import gaussian_filter, percentile_filter
            e = s.esdf
            lo, hi = e.origin.tolist(), e.upper.tolist()
            xs = torch.arange(lo[0], hi[0], TERRAIN_RES, device=self.dev)
            ys = torch.arange(lo[1], hi[1], TERRAIN_RES, device=self.dev)
            Xg, Yg = torch.meshgrid(xs, ys, indexing="ij")
            top = min(hi[2], 15.0)
            gz = torch.cat([self._ground(s, a, b, top) for a, b in zip(Xg.reshape(-1).split(32768), Yg.reshape(-1).split(32768))])
            g = gz.reshape(Xg.shape).cpu().numpy()
            t = gaussian_filter(percentile_filter(g, TERRAIN_PCT, size=TERRAIN_WIN), 2.0)
            self.cache[key] = (torch.tensor(t, device=self.dev, dtype=torch.float32), lo[0], lo[1], len(xs), len(ys))
        T, x0, y0, nx, ny = self.cache[key]
        u = ((x - x0) / TERRAIN_RES) / (nx - 1) * 2 - 1
        v = ((y - y0) / TERRAIN_RES) / (ny - 1) * 2 - 1
        import torch.nn.functional as F
        grid = torch.stack([v, u], -1)[None, :, None, :]  # grid_sample x->last dim (y index), y->first (x index)
        return F.grid_sample(T[None, None], grid, align_corners=True, padding_mode="border")[0, 0, :, 0]

    @torch.no_grad()
    def column_blocked(self, s, X, Y, alt):
        """Same test as fly(): local ground gz = the HIGHEST first-surface among a 3x3 neighbourhood
        (+-GROUND_R m) so grass/shrub fuzz doesn't sit 'above ground'; blocked if any z in
        [gz+COLUMN_START, alt] (0.2 m steps) has ESDF < DRONE_R+0.1, or the surface is a roof/canopy
        (gz more than GROUND_MAX above the terrain map, see _terrain_at). X,Y: same-shape tensors."""
        flat_x, flat_y = X.reshape(-1), Y.reshape(-1)
        out = torch.zeros(len(flat_x), dtype=torch.bool, device=self.dev)
        gz_all = torch.zeros(len(flat_x), device=self.dev)
        ter_all = torch.zeros(len(flat_x), device=self.dev)
        zs_col = torch.arange(-2.4, alt + 1e-6, 0.2, device=self.dev)
        offs = [(dx, dy) for dx in (-GROUND_R, 0.0, GROUND_R) for dy in (-GROUND_R, 0.0, GROUND_R)]
        for i in range(0, len(flat_x), 32768):
            x, y = flat_x[i:i + 32768], flat_y[i:i + 32768]
            n = len(x)
            gz = torch.stack([self._ground(s, x + dx, y + dy, alt) for dx, dy in offs], 0).max(0).values
            terrain = self._terrain_at(s, x, y)
            P = torch.stack([x[:, None].expand(n, len(zs_col)), y[:, None].expand(n, len(zs_col)),
                             zs_col[None].expand(n, -1)], -1)
            sd = self.q(s, P.reshape(-1, 3)).reshape(n, -1)
            in_col = (zs_col[None] >= gz[:, None] + COLUMN_START) & (zs_col[None] <= alt)
            out[i:i + n] = ((sd < DRONE_R + 0.1) & in_col).any(1) | (gz - terrain > GROUND_MAX)
            ter_all[i:i + n] = terrain
            gz_all[i:i + n] = gz
        self._last_terrain = ter_all.reshape(X.shape)
        return out.reshape(X.shape), gz_all.reshape(X.shape)

    @torch.no_grad()
    def column_png(self, name, alt, px=1024) -> bytes:
        key = ("col", name, round(float(alt), 2), px)
        if key in self.cache:
            return self.cache[key]
        s = self.scene(name)
        X, Y = self._pixel_xy(name, px)
        blocked, _ = self.column_blocked(s, X, Y, float(alt))
        buf = io.BytesIO()
        Image.fromarray(blocked.cpu().numpy().astype("uint8") * 255).save(buf, "PNG")
        if len(self.cache) > 64:
            self.cache.clear()
        self.cache[key] = buf.getvalue()
        return self.cache[key]

    # ------------------------------------------------------------ planning
    @torch.no_grad()
    def _ground_z(self, s, x, y, top):
        zs = torch.arange(top, -3.0, -0.1, device=self.dev)
        sd = self.q(s, torch.stack([torch.full_like(zs, x), torch.full_like(zs, y), zs], -1))
        hit = (sd < 0.05).nonzero()
        return float(zs[hit[0, 0]]) if len(hit) else 0.0

    @torch.no_grad()
    def plan(self, s, a, b, alt, clear, res=0.5):
        e = s.esdf
        lo, hi = e.origin.tolist(), e.upper.tolist()
        xs = np.arange(lo[0] + res / 2, hi[0], res)
        ys = np.arange(lo[1] + res / 2, hi[1], res)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        P = torch.tensor(np.stack([X, Y, np.full_like(X, alt)], -1).reshape(-1, 3), dtype=torch.float32, device=self.dev)
        free = (self.q(s, P).reshape(X.shape) >= clear).cpu().numpy()
        nx, ny = free.shape
        cell = lambda p: (int(round((p[0] - xs[0]) / res)), int(round((p[1] - ys[0]) / res)))
        sa, sb = cell(a), cell(b)
        for nm, c in (("A", sa), ("B", sb)):
            if not (0 <= c[0] < nx and 0 <= c[1] < ny):
                return None, f"{nm} is outside the scene bounds"
            if not free[c]:
                return None, f"{nm} is blocked at {alt} m with {clear} m clearance"
        # A* (8-connected)
        nbr = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        h = lambda c: math.hypot(c[0] - sb[0], c[1] - sb[1])
        openq = [(h(sa), 0.0, sa)]
        g = {sa: 0.0}
        parent = {}
        closed = set()
        while openq:
            _, gc, c = heapq.heappop(openq)
            if c == sb:
                break
            if c in closed:
                continue
            closed.add(c)
            for dx, dy, w in nbr:
                n = (c[0] + dx, c[1] + dy)
                if not (0 <= n[0] < nx and 0 <= n[1] < ny) or not free[n]:
                    continue
                ng = gc + w
                if ng < g.get(n, 1e18):
                    g[n], parent[n] = ng, c
                    heapq.heappush(openq, (ng + h(n), ng, n))
        if sb not in parent and sa != sb:
            return None, f"no path at {alt} m with {clear} m clearance"
        cells = [sb]
        while cells[-1] != sa:
            cells.append(parent[cells[-1]])
        cells.reverse()
        pts = [np.array([xs[c[0]], ys[c[1]], alt]) for c in cells]
        pts[0], pts[-1] = np.array([a[0], a[1], alt]), np.array([b[0], b[1], alt])
        # shortcut smoothing: keep the farthest visible point (ESDF line-of-sight with clearance)
        def visible(p, q):
            n = max(2, int(np.linalg.norm(q - p) / 0.2))
            seg = torch.tensor(np.linspace(p, q, n), dtype=torch.float32, device=self.dev)
            return bool((self.q(s, seg) >= clear * 0.9).all())
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not visible(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return np.array(out), None

    # ------------------------------------------------------------ fly
    @torch.no_grad()
    def fly(self, req: dict) -> dict:
        name = req["scene"]
        s = self.scene(name)
        a = (float(req["a"]["x"]), float(req["a"]["y"]))
        b = (float(req["b"]["x"]), float(req["b"]["y"]))
        alt = float(req.get("alt_m", 5.0))
        speed = float(req.get("speed_mps", 3.0))
        clear = float(req.get("clear_m", 1.5))
        tol = bool(req.get("takeoff_land", True))
        wind = bool(req.get("wind", False))
        path, why = self.plan(s, a, b, alt, clear)
        if path is None:
            return {"ok": False, "reason": why}
        gA = float(self.column_blocked(s, torch.tensor([[a[0]]], device=self.dev), torch.tensor([[a[1]]], device=self.dev), alt)[1][0, 0])
        gB = float(self.column_blocked(s, torch.tensor([[b[0]]], device=self.dev), torch.tensor([[b[1]]], device=self.dev), alt)[1][0, 0])
        # columns at A and B must be clear for the climb/descent (same test as the column occupancy mask)
        if tol:
            for nm, (x, y) in (("A", a), ("B", b)):
                blk, _ = self.column_blocked(s, torch.tensor([[x]], device=self.dev), torch.tensor([[y]], device=self.dev), alt)
                if bool(blk[0, 0]):
                    _, gz = self.column_blocked(s, torch.tensor([[x]], device=self.dev), torch.tensor([[y]], device=self.dev), alt)
                    above = float(gz[0, 0]) - float(self._last_terrain[0, 0])
                    why = (f"{nm} is on a roof/canopy ({above:.1f} m above the terrain)" if above > GROUND_MAX
                           else f"vertical column above {nm} is blocked (overhang between ground and {alt} m)")
                    return {"ok": False, "reason": why}
        P = StickParams(dr=0.0, wind_std=0.3 if wind else 0.0)
        dyn = StickDynamics(1, P, self.dev, control_dt=0.1, substeps=5)
        start = torch.tensor([[a[0], a[1], (gA + 0.35) if tol else alt]], device=self.dev)
        d0 = path[1] - path[0] if len(path) > 1 else np.array([1.0, 0, 0])
        dyn.reset(torch.tensor([0], device=self.dev), start, torch.tensor([math.atan2(d0[1], d0[0])], device=self.dev))
        pois = []
        for i, pi in enumerate(req.get("pois") or []):
            px_, py_ = float(pi["x"]), float(pi["y"])
            pz = pi.get("z")
            if pz is None:  # terrain height + 1 m
                pz = float(self._terrain_at(s, torch.tensor([px_], device=self.dev), torch.tensor([py_], device=self.dev))[0]) + 1.0
            pois.append({"x": px_, "y": py_, "z": round(float(pz), 3), "label": pi.get("label", f"poi{i}")})
        poi_mode = req.get("poi_mode", "nearest")
        if poi_mode not in ("nearest", "sequence"):
            return {"ok": False, "reason": f"poi_mode must be 'nearest' or 'sequence', got {poi_mode!r}"}
        ctl = PursuitController(path, alt, speed, P, gB + 0.35 if tol else None, (a[0], a[1]),
                                pois=pois, poi_mode=poi_mode)
        traj, collisions, contact, min_clear, touchdown = [], 0, False, 1e9, False
        t_max = sum(np.linalg.norm(np.diff(path, axis=0), axis=1)) / max(speed, 0.5) * 3 + 60
        t = 0.0
        reached = False
        while t < t_max:
            pos = dyn.pos[0].cpu().numpy()
            vel = dyn.vel[0].cpu().numpy()
            yaw = float(dyn.yaw[0])
            cmd, done = ctl.command(pos, vel, yaw)
            clear_now = float(self.q(s, dyn.pos)[0])
            traj.append({"t": round(t, 2), "x": round(float(pos[0]), 3), "y": round(float(pos[1]), 3), "z": round(float(pos[2]), 3),
                         "vx": round(float(vel[0]), 3), "vy": round(float(vel[1]), 3), "vz": round(float(vel[2]), 3),
                         "yaw": round(yaw, 4), "cmd": [round(float(c), 3) for c in cmd], "clear": round(clear_now, 3),
                         "gimbal_pitch_deg": round(ctl.gimbal_deg, 2), "poi": ctl.active})
            if done:
                reached = True
                break
            subs = dyn.step(torch.tensor([cmd], dtype=torch.float32, device=self.dev))
            sd = self.q(s, torch.cat(subs))
            if ctl.phase == "cruise":  # clearance stat = en-route only (takeoff/landing touch the ground)
                min_clear = min(min_clear, float(sd.min()))
            hit = bool(sd.min() < DRONE_R)
            landing = tol and ctl.phase in ("descend", "done") and float(dyn.pos[0, 2]) - gB < 0.8
            if hit and landing:
                touchdown = True
            elif hit and not contact:
                collisions += 1
            contact = hit and not landing
            t += 0.1
        final = np.array([traj[-1]["x"], traj[-1]["y"]])
        tid = uuid.uuid4().hex[:10]
        length = float(sum(math.dist((p["x"], p["y"], p["z"]), (q["x"], q["y"], q["z"])) for p, q in zip(traj, traj[1:])))
        out = {"ok": True, "id": tid, "scene": name, "path": [[round(float(v), 3) for v in p] for p in path], "traj": traj,
               "pois": pois, "poi_mode": poi_mode,
               "length_m": round(length, 2), "duration_s": round(t, 2), "min_clear_m": round(min_clear, 3),
               "collisions": collisions, "touchdown": touchdown, "reached": reached, "final_err_m": round(float(np.linalg.norm(final - np.array(b))), 3)}
        self.trajs[tid] = out
        while len(self.trajs) > 20:
            self.trajs.popitem(last=False)
        return out


class PursuitController:
    """Phases: climb at A -> pure pursuit along the path at alt -> descend at B. Outputs the env's
    action [fwd, right, up, yaw_right] in [-1,1]. Speed profile respects a braking budget for the
    endpoint and for corners (using a derated a_brake to cover lag and command delay)."""

    GIMBAL_MIN, GIMBAL_MAX, GIMBAL_RATE = -90.0, 20.0, 60.0  # Mini 3 gimbal pitch range (deg), rate (deg/s)
    NEAREST_HYST = 2.0  # m

    def __init__(self, path, alt, speed, P: StickParams, land_z, a_xy, pois=None, poi_mode="nearest", dt=0.1,
                 default_gimbal=-10.0):
        self.path, self.alt, self.speed, self.P = path, alt, speed, P
        self.pois = pois or []
        self.poi_mode, self.dt = poi_mode, dt
        self.active = 0 if self.pois else -1
        self.gimbal_deg = default_gimbal
        self.land_z, self.a_xy = land_z, np.array(a_xy)
        self.phase = "climb"
        self.seg = 0
        self.a_b = 0.6 * P.a_brake
        seglen = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
        self.cum = np.concatenate([[0], np.cumsum(seglen)])
        self.total = self.cum[-1]
        # corner speed limits at interior vertices
        self.v_corner = np.full(len(path), speed)
        for i in range(1, len(path) - 1):
            u, w = path[i, :2] - path[i - 1, :2], path[i + 1, :2] - path[i, :2]
            cth = float(u @ w / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-9))
            th = math.acos(max(-1, min(1, cth)))
            self.v_corner[i] = max(0.8, speed * (1 - th / math.pi) ** 2)
        self.v_corner[-1] = 0.0
        # sequence mode: arc length of each POI's projection onto the path
        self.poi_s = []
        for q in self.pois:
            best, bs = 1e18, 0.0
            for i in range(len(path) - 1):
                a, b = path[i, :2], path[i + 1, :2]
                ab = b - a
                t = float(np.clip((np.array([q["x"], q["y"]]) - a) @ ab / (ab @ ab + 1e-9), 0, 1))
                d = float(np.linalg.norm(a + t * ab - np.array([q["x"], q["y"]])))
                if d < best:
                    best, bs = d, self.cum[i] + t * (self.cum[i + 1] - self.cum[i])
            self.poi_s.append(bs)

    def _update_poi(self, pos, s_path):
        if not self.pois:
            return
        if self.phase == "climb":
            self.active = 0 if self.poi_mode == "sequence" else self.active
        if self.poi_mode == "sequence":
            while self.active + 1 < len(self.pois) and s_path is not None and s_path >= self.poi_s[self.active + 1]:
                self.active += 1
            if self.phase in ("descend", "done"):
                self.active = len(self.pois) - 1
        else:
            d = [math.hypot(q["x"] - pos[0], q["y"] - pos[1]) for q in self.pois]
            k = int(np.argmin(d))
            if self.active < 0 or d[k] < d[self.active] - self.NEAREST_HYST:
                self.active = k

    def _poi_aim(self, pos, yaw):
        """yaw target (rad) and gimbal pitch target (deg) for the active POI."""
        q = self.pois[self.active]
        dx, dy, dz = q["x"] - pos[0], q["y"] - pos[1], q["z"] - pos[2]
        h = math.hypot(dx, dy)
        want_yaw = math.atan2(dy, dx) if h > 0.3 else yaw
        pitch = math.degrees(math.atan2(dz, max(h, 1e-3)))
        return want_yaw, float(np.clip(pitch, self.GIMBAL_MIN, self.GIMBAL_MAX))

    def _project(self, p):
        best, bs, bt = 1e9, self.seg, 0.0
        for i in range(self.seg, min(self.seg + 3, len(self.path) - 1)):
            a, b = self.path[i, :2], self.path[i + 1, :2]
            ab = b - a
            t = float(np.clip((p - a) @ ab / (ab @ ab + 1e-9), 0, 1))
            d = float(np.linalg.norm(a + t * ab - p))
            if d < best:
                best, bs, bt = d, i, t
        self.seg = bs
        return self.cum[bs] + bt * (self.cum[bs + 1] - self.cum[bs])

    def _at(self, s):
        s = min(max(s, 0), self.total)
        i = int(np.searchsorted(self.cum, s, side="right") - 1)
        i = min(i, len(self.path) - 2)
        t = (s - self.cum[i]) / max(self.cum[i + 1] - self.cum[i], 1e-9)
        return self.path[i, :2] + t * (self.path[i + 1, :2] - self.path[i, :2])

    def _vz(self, z, target):
        return float(np.clip(1.2 * (target - z), -self.P.v_down_max * 0.8, self.P.v_up_max * 0.8))

    def command(self, pos, vel, yaw):
        cmd, done = self._command(pos, vel, yaw)
        return cmd, done

    def _command(self, pos, vel, yaw):
        P = self.P
        v_xy = np.zeros(2)
        vz = 0.0
        want_yaw = yaw
        s_now = self._project(pos[:2]) if self.phase != "climb" else 0.0
        self._update_poi(pos, s_now)
        if self.phase == "climb":
            v_xy = np.clip(1.0 * (self.a_xy - pos[:2]), -1, 1)
            vz = self._vz(pos[2], self.alt)
            if abs(pos[2] - self.alt) < 0.3:
                self.phase = "cruise"
        elif self.phase == "cruise":
            s = self._project(pos[:2])
            rem = self.total - s
            spd = float(np.linalg.norm(vel[:2]))
            look = 1.5 + 0.6 * spd
            target = self._at(s + look)
            d = target - pos[:2]
            dn = d / (np.linalg.norm(d) + 1e-9)
            v = min(self.speed, math.sqrt(max(0.0, 2 * self.a_b * max(rem - 0.3, 0))) + 0.2)
            for i in range(self.seg + 1, len(self.path)):
                dist = self.cum[i] - s
                if dist > 25:
                    break
                v = min(v, math.sqrt(self.v_corner[i] ** 2 + 2 * self.a_b * max(dist, 0)))
            v_xy = dn * v
            if rem < 0.6:
                v_xy = np.clip(1.0 * (self.path[-1, :2] - pos[:2]), -0.6, 0.6)
            want_yaw = math.atan2(dn[1], dn[0]) if np.linalg.norm(d) > 0.5 else yaw
            vz = self._vz(pos[2], self.alt)
            if rem < 0.5 and spd < 0.4:
                self.phase = "descend" if self.land_z is not None else "done"
        elif self.phase == "descend":
            v_xy = np.clip(1.0 * (self.path[-1, :2] - pos[:2]), -0.6, 0.6)
            vz = self._vz(pos[2], self.land_z)
            if pos[2] - self.land_z < 1.5:  # gentle final approach
                vz = max(vz, -0.5)
            if pos[2] - self.land_z < 0.1 and abs(vel[2]) < 0.2:
                self.phase = "done"
        if self.phase == "done":
            return [0.0, 0.0, 0.0, 0.0], True
        if self.pois:  # look-at: yaw + gimbal track the active POI; translation keeps following the path
            want_yaw, g_target = self._poi_aim(pos, yaw)
            step = self.GIMBAL_RATE * self.dt
            self.gimbal_deg = float(np.clip(g_target, self.gimbal_deg - step, self.gimbal_deg + step))
        c, s_ = math.cos(yaw), math.sin(yaw)
        fwd = v_xy[0] * c + v_xy[1] * s_
        right = v_xy[0] * s_ - v_xy[1] * c
        err = (want_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        r = float(np.clip(2.0 * err, -P.yaw_rate_max, P.yaw_rate_max))
        up = vz / P.v_up_max if vz >= 0 else vz / P.v_down_max
        return [float(np.clip(fwd / P.v_h_max, -1, 1)), float(np.clip(right / P.v_h_max, -1, 1)),
                float(np.clip(up, -1, 1)), float(np.clip(-r / P.yaw_rate_max, -1, 1))], False
