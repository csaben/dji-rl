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

    @torch.no_grad()
    def view_gaussians(self, s: Scene):
        """Splat for the replay/viewer cameras with the sky shell and haze culled: drop Gaussians above the
        collision grid (zmax), and Gaussians > 6 m above the local terrain that are sky-coloured (bright or
        grey DC). World Labs scenes wrap the capture in a sky dome / haze that otherwise fills the view above
        ~15 m. Cached per scene. Viewer only: training renders are unchanged."""
        key = ("view_g", s.name)
        if key not in self.cache:
            from .scene import SH_C0, Gaussians
            g = s.gaussians
            m = g.means
            ter = torch.cat([self._terrain_at(s, m[i:i + 500000, 0], m[i:i + 500000, 1]) for i in range(0, len(m), 500000)])
            rgb = (g.sh[:, 0] * SH_C0 + 0.5).clamp(0, 1)
            sat = rgb.max(1).values - rgb.min(1).values
            drop = (m[:, 2] > s.esdf.upper[2]) | ((m[:, 2] - ter > 6.0) & ((rgb.mean(1) > 0.45) | (sat < 0.12)))
            k = ~drop
            self.cache[key] = Gaussians(g.means[k], g.quats[k], g.scales[k], g.opacities[k], g.sh[k])
        return self.cache[key]

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
    def plan(self, s, a, b, alt, clear, res=0.5, via=(), via_tol=1.0, growth=0.0, origin=None, base_h=0.0):
        """A* per leg (a -> via... -> b) on the ESDF slice at alt with clearance, concatenated, then
        line-of-sight shortcutting that may cut across a via only if the shortcut passes within
        via_tol of it. Returns (path [K,3], via_idx [n_via] = index of the path vertex nearest each
        via, None) or (None, None, reason)."""
        e = s.esdf
        lo, hi = e.origin.tolist(), e.upper.tolist()
        xs = np.arange(lo[0] + res / 2, hi[0], res)
        ys = np.arange(lo[1] + res / 2, hi[1], res)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        P = torch.tensor(np.stack([X, Y, np.full_like(X, alt)], -1).reshape(-1, 3), dtype=torch.float32, device=self.dev)
        ox, oy = origin if origin is not None else a
        # horizontal sim-to-real margin (scale/heading error): blocked0 = ESDF < clear on this slice;
        # d2 = 2D distance (m) to the nearest blocked0 cell; free iff ~blocked0 and d2 >= base_h + growth*|cell - origin|
        from scipy.ndimage import distance_transform_edt
        blocked0 = (self.q(s, P) < clear).reshape(X.shape).cpu().numpy()
        d2 = distance_transform_edt(~blocked0) * res
        hreq_np = base_h + growth * np.hypot(X - ox, Y - oy)
        free = ~blocked0 & (d2 >= hreq_np)
        D2 = torch.tensor(d2, dtype=torch.float32, device=self.dev)
        x0g, y0g, nxg, nyg = float(xs[0]), float(ys[0]), len(xs), len(ys)

        def d2_at(pts):
            import torch.nn.functional as F
            u = (pts[..., 0] - x0g) / res / (nxg - 1) * 2 - 1
            v = (pts[..., 1] - y0g) / res / (nyg - 1) * 2 - 1
            g = torch.stack([v, u], -1).reshape(1, -1, 1, 2)
            return F.grid_sample(D2[None, None], g, align_corners=True, padding_mode="border").reshape(pts.shape[:-1])

        hreq = lambda pts: base_h + growth * torch.hypot(pts[..., 0] - ox, pts[..., 1] - oy)
        ok_pts = lambda pts: (self.q(s, pts) >= clear) & (d2_at(pts) >= hreq(pts))
        # empty slice (e.g. above zmax): no obstacle to measure a margin to -> margin_at returns +inf
        empty = not blocked0.any()
        self._margin_fns = (ok_pts, (lambda pts: torch.full(pts.shape[:-1], float("inf"), device=pts.device)) if empty
                            else (lambda pts: d2_at(pts) - hreq(pts)))
        nx, ny = free.shape
        cell = lambda p: (int(round((p[0] - xs[0]) / res)), int(round((p[1] - ys[0]) / res)))
        stops = [("A", a)] + [(f"waypoint {i}", v) for i, v in enumerate(via)] + [("B", b)]
        for nm, p in stops:
            c = cell(p)
            if not (0 <= c[0] < nx and 0 <= c[1] < ny):
                return None, None, f"{nm} is outside the scene bounds"
            if not free[c]:
                need_h = base_h + growth * math.hypot(p[0] - ox, p[1] - oy)
                return None, None, (f"{nm} is blocked at {alt} m with {clear} m clearance"
                                    + (f" + {need_h:.2f} m horizontal margin" if need_h > 0 else ""))
        nbr = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]

        def astar(sa, sb):
            h = lambda c: math.hypot(c[0] - sb[0], c[1] - sb[1])
            openq, g, parent, closed = [(h(sa), 0.0, sa)], {sa: 0.0}, {}, set()
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
            if sb != sa and sb not in parent:
                return None
            cells = [sb]
            while cells[-1] != sa:
                cells.append(parent[cells[-1]])
            return cells[::-1]

        pts, anchor = [], []  # anchor[k] = dense index of stop k
        for k in range(len(stops) - 1):
            cells = astar(cell(stops[k][1]), cell(stops[k + 1][1]))
            if cells is None:
                return None, None, (f"no path from {stops[k][0]} to {stops[k + 1][0]} at {alt} m with {clear} m clearance"
                                    + (f" + horizontal margin {base_h} + {growth}/m from origin" if (growth or base_h) else ""))
            seg = [np.array([xs[c[0]], ys[c[1]], alt]) for c in cells]
            seg[0] = np.array([stops[k][1][0], stops[k][1][1], alt])
            seg[-1] = np.array([stops[k + 1][1][0], stops[k + 1][1][1], alt])
            if k == 0:
                anchor.append(0)
                pts.extend(seg)
            else:
                pts.extend(seg[1:])
            anchor.append(len(pts) - 1)
        via_dense = anchor[1:-1]

        def visible(p, q):
            n = max(2, int(np.linalg.norm(q - p) / 0.2))
            seg = torch.tensor(np.linspace(p, q, n), dtype=torch.float32, device=self.dev)
            return bool(ok_pts(seg).all())

        def near_vias(i, j):
            for vi in via_dense:
                if i < vi < j:
                    p, q, v = pts[i][:2], pts[j][:2], pts[vi][:2]
                    ab = q - p
                    t = float(np.clip((v - p) @ ab / (ab @ ab + 1e-9), 0, 1))
                    if np.linalg.norm(p + t * ab - v) > via_tol:
                        return False
            return True

        out, keep_idx, i = [pts[0]], [0], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not (near_vias(i, j) and visible(pts[i], pts[j])):
                j -= 1
            out.append(pts[j])
            keep_idx.append(j)
            i = j
        out = np.array(out)
        via_idx = []
        for vi in via_dense:  # smoothed vertex closest to each via (vias may have been cut within tolerance)
            via_idx.append(int(np.argmin(np.linalg.norm(out[:, :2] - pts[vi][:2], axis=1))))
        return out, via_idx, None

    # ------------------------------------------------------------ fly
    @torch.no_grad()
    def fly(self, req: dict) -> dict:
        """See module doc. Per-point altitude: leg i flies at max(alt(p_i), alt(p_i+1)); consecutive legs at
        the same altitude form one section flown without stopping; between sections the drone stops at the
        via and climbs/descends in place (vertical-then-horizontal, like the Map planner)."""
        name = req["scene"]
        s = self.scene(name)
        alt_m = float(req.get("alt_m", 5.0))
        speed = float(req.get("speed_mps", 3.0))
        clear = float(req.get("clear_m", 1.5))
        tol = bool(req.get("takeoff_land", True))
        wind = bool(req.get("wind", False))
        growth = float(req.get("clear_growth_per_m", 0.0) or 0.0)
        base_h = float(req.get("clear_base_h", 0.0) or 0.0)
        pt = lambda d: (float(d["x"]), float(d["y"]), float(d["alt"]) if d.get("alt") is not None else alt_m)
        pts = [pt(req["a"])] + [pt(v) for v in (req.get("via") or [])] + [pt(req["b"])]
        a, b = pts[0][:2], pts[-1][:2]
        via = [p[:2] for p in pts[1:-1]]
        names = ["A"] + [f"waypoint {i}" for i in range(len(via))] + ["B"]
        co = req.get("clear_origin") or {"x": a[0], "y": a[1]}
        origin = (float(co["x"]), float(co["y"]))
        echo = {"via": [{"x": p[0], "y": p[1], "alt": p[2]} for p in pts[1:-1]], "per_point_alt": True,
                "clear_growth_per_m": growth, "clear_base_h": base_h, "clear_origin": {"x": origin[0], "y": origin[1]}}
        fail = lambda why: {"ok": False, "reason": why, **echo}
        style = req.get("flight_style", "normal")
        if style not in ("normal", "fpv"):
            return fail(f"flight_style must be 'normal' or 'fpv', got {style!r}")
        poi_mode = req.get("poi_mode", "nearest")
        if poi_mode not in ("nearest", "sequence"):
            return fail(f"poi_mode must be 'nearest' or 'sequence', got {poi_mode!r}")
        leg_alt = [max(pts[i][2], pts[i + 1][2]) for i in range(len(pts) - 1)]
        # sections: maximal runs of consecutive legs at the same altitude
        sections, i0 = [], 0
        for i in range(1, len(leg_alt) + 1):
            if i == len(leg_alt) or leg_alt[i] != leg_alt[i0]:
                sections.append((i0, i, leg_alt[i0]))  # points i0 .. i (legs i0..i-1)
                i0 = i
        T = lambda v: torch.tensor([[v]], device=self.dev)
        # plan each section at its altitude
        plans, legs = [], []
        for (p0, p1, alt) in sections:
            path, via_idx, why = self.plan(s, pts[p0][:2], pts[p1][:2], alt, clear, via=[q[:2] for q in pts[p0 + 1:p1]],
                                           growth=growth, origin=origin, base_h=base_h)
            if path is None:
                why = why.replace("A ", f"{names[p0]} ", 1) if p0 else why
                why = why.replace(" B ", f" {names[p1]} ") if p1 != len(pts) - 1 else why
                return fail(why)
            plans.append((path, via_idx, self._margin_fns, alt))
            cuts = [0] + list(via_idx) + [len(path) - 1]
            for k in range(p1 - p0):
                seg = path[cuts[k]:cuts[k + 1] + 1, :2]
                legs.append({"from": p0 + k, "to": p0 + k + 1, "alt": alt,
                             "length_m": round(float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()), 2)})
        # vertical checks: takeoff/landing columns at A and B; in-place altitude changes at section joints
        def col_check(i, z_to):
            x, y = pts[i][:2]
            blk, gz = self.column_blocked(s, T(x), T(y), z_to)
            if bool(blk[0, 0]):
                above = float(gz[0, 0]) - float(self._last_terrain[0, 0])
                return (f"{names[i]} is on a roof/canopy ({above:.1f} m above the terrain)" if above > GROUND_MAX
                        else f"vertical column above {names[i]} is blocked (overhang between ground and {z_to} m)")
            return None
        if tol:
            for i, z in ((0, sections[0][2]), (len(pts) - 1, sections[-1][2])):
                why = col_check(i, z)
                if why:
                    return fail(why)
        for (pa, pb, za), (_, _, zb) in zip(sections, sections[1:]):
            x, y = pts[pb][:2]
            zs = torch.arange(min(za, zb), max(za, zb) + 1e-6, 0.2, device=self.dev)
            col = torch.stack([torch.full_like(zs, x), torch.full_like(zs, y), zs], -1)
            if bool((self.q(s, col) < DRONE_R + 0.1).any()):
                verb = "climb" if zb > za else "descend"
                return fail(f"can't {verb} at {names[pb]} from {za:g} m to {zb:g} m (overhang)")
        gA = float(self.column_blocked(s, T(a[0]), T(a[1]), sections[0][2])[1][0, 0])
        gB = float(self.column_blocked(s, T(b[0]), T(b[1]), sections[-1][2])[1][0, 0])
        pois = []
        for i, pi in enumerate(req.get("pois") or []):
            px_, py_ = float(pi["x"]), float(pi["y"])
            pz = pi.get("z")
            if pz is None:  # terrain height + 1 m
                pz = float(self._terrain_at(s, torch.tensor([px_], device=self.dev), torch.tensor([py_], device=self.dev))[0]) + 1.0
            rad = pi.get("radius_m")
            rad = None if rad in (None, "") else float(rad)
            pois.append({"x": px_, "y": py_, "z": round(float(pz), 3), "label": pi.get("label", f"poi{i}"), "radius_m": rad})
        if style == "fpv":
            pois = []
        P = StickParams(dr=0.0, wind_std=0.3 if wind else 0.0)
        a_lat = float(req.get("a_lat", 0.8 * P.a_acc))
        gimbal0 = float(req.get("fpv_gimbal_deg", -10.0)) if style == "fpv" else -10.0

        def make_ctl(k):
            path, via_idx, (ok_pts, _), alt = plans[k]
            last = k == len(plans) - 1
            land = (gB + 0.35) if (tol and last) else None
            start_xy = pts[sections[k][0]][:2]
            if style == "fpv":
                ok_fn = lambda q: bool(ok_pts(torch.tensor(q, dtype=torch.float32, device=self.dev)).all())
                curve, kappa = fillet_curve(path, speed, a_lat, ok_fn, max_dev={vi: 1.0 for vi in via_idx})

                def depart_ok(pos, curve=curve):
                    if k > 0 or pos[2] - gA < 2.0:  # climbing departure only from the ground at A
                        return False
                    j = min(len(curve) - 1, int(10.0 / 0.25))
                    seg = np.linspace(pos, curve[j], 40)
                    return bool(ok_pts(torch.tensor(seg, dtype=torch.float32, device=self.dev)).all())
                c = FpvController(curve, kappa, alt, speed, P, land, start_xy, a_lat, gimbal0,
                                  depart_ok if tol else (lambda q: False))
                c.curve = curve
                return c
            return PursuitController(path, alt, speed, P, land, start_xy, pois=pois, poi_mode=poi_mode)

        dyn = StickDynamics(1, P, self.dev, control_dt=0.1, substeps=5)
        path0 = plans[0][0]
        start = torch.tensor([[a[0], a[1], (gA + 0.35) if tol else sections[0][2]]], device=self.dev)
        d0 = path0[1] - path0[0] if len(path0) > 1 else np.array([1.0, 0, 0])
        dyn.reset(torch.tensor([0], device=self.dev), start, torch.tensor([math.atan2(d0[1], d0[0])], device=self.dev))
        # full planned path (vertical joints appear as two points with the same x,y)
        full = np.concatenate([plans[0][0]] + [p[0] for p in plans[1:]])
        traj, collisions, contact, min_clear, touchdown = [], 0, False, 1e9, False
        prev_v, min_margin = None, 1e9
        arc = ArcTracker(full)
        via_s, start_i = [], 0
        for k, v in enumerate(via):  # arc length of each via along the planned path (in order)
            sv, start_i = arc.arc_of(np.array(v), start=start_i)
            via_s.append(sv)
        via_reached = [None] * len(via)
        t_max = sum(l["length_m"] for l in legs) / max(speed, 0.5) * 3 + 60 + 20 * len(sections)
        t, reached, sec = 0.0, False, 0
        ctl = make_ctl(0)
        while t < t_max:
            pos = dyn.pos[0].cpu().numpy()
            vel = dyn.vel[0].cpu().numpy()
            yaw = float(dyn.yaw[0])
            cmd, done = ctl.command(pos, vel, yaw)
            if done and sec < len(plans) - 1:  # hovering at a section joint: next section climbs/descends in place
                prev = ctl
                sec += 1
                ctl = make_ctl(sec)
                ctl.active, ctl.gimbal_deg = prev.active, prev.gimbal_deg
                cmd, done = ctl.command(pos, vel, yaw)
            clear_now = float(self.q(s, dyn.pos)[0])
            s_drone = arc.update(pos[:2])
            for k, sv in enumerate(via_s):
                if via_reached[k] is None and s_drone >= sv - 0.5:
                    via_reached[k] = round(t, 2)
            seg_i = sum(1 for r in via_reached if r is not None)
            sp = float(np.linalg.norm(vel[:2]))
            lat_acc = 0.0 if prev_v is None or sp < 0.2 else abs(float(vel[0] * (vel[1] - prev_v[1]) - vel[1] * (vel[0] - prev_v[0]))) / sp / 0.1
            traj.append({"t": round(t, 2), "x": round(float(pos[0]), 3), "y": round(float(pos[1]), 3), "z": round(float(pos[2]), 3),
                         "vx": round(float(vel[0]), 3), "vy": round(float(vel[1]), 3), "vz": round(float(vel[2]), 3),
                         "yaw": round(yaw, 4), "cmd": [round(float(c), 3) for c in cmd], "clear": round(clear_now, 3),
                         "gimbal_pitch_deg": round(ctl.gimbal_deg, 2), "poi": ctl.active,
                         "curvature": round(getattr(ctl, "cur_kappa", 0.0), 4), "lat_acc": round(lat_acc, 3),
                         "seg": seg_i, "section": sec})
            prev_v = vel
            if done:
                reached = True
                break
            subs = dyn.step(torch.tensor([cmd], dtype=torch.float32, device=self.dev))
            sd = self.q(s, torch.cat(subs))
            if ctl.phase == "cruise":  # clearance stat = en-route only (takeoff/landing touch the ground)
                min_clear = min(min_clear, float(sd.min()))
                min_margin = min(min_margin, float(plans[sec][2][1](torch.cat(subs)).min()))
            hit = bool(sd.min() < DRONE_R)
            landing = tol and sec == len(plans) - 1 and ctl.phase in ("descend", "done") and float(dyn.pos[0, 2]) - gB < 0.8
            if hit and landing:
                touchdown = True
            elif hit and not contact:
                collisions += 1
            contact = hit and not landing
            t += 0.1
        final = np.array([traj[-1]["x"], traj[-1]["y"]])
        tid = uuid.uuid4().hex[:10]
        length = float(sum(math.dist((p["x"], p["y"], p["z"]), (q["x"], q["y"], q["z"])) for p, q in zip(traj, traj[1:])))
        out = {"ok": True, "id": tid, "scene": name, "path": [[round(float(v), 3) for v in p] for p in full], "traj": traj,
               "pois": pois, "poi_mode": poi_mode, "style": style, **echo, "legs": legs, "via_reached": via_reached,
               "min_margin_m": round(min_margin, 3) if min_margin < 1e8 else None,
               **({"curve": [[round(float(v), 3) for v in p] for p in np.concatenate([fillet_curve(pl[0], speed, a_lat,
                   lambda q, f=pl[2][0]: bool(f(torch.tensor(q, dtype=torch.float32, device=self.dev)).all()),
                   max_dev={vi: 1.0 for vi in pl[1]})[0] for pl in plans])[::4]]} if style == "fpv" else {}),
               "length_m": round(length, 2), "duration_s": round(t, 2), "min_clear_m": round(min_clear, 3),
               "collisions": collisions, "touchdown": touchdown, "reached": reached,
               "final_err_m": round(float(np.linalg.norm(final - np.array(b))), 3)}
        self.trajs[tid] = out
        while len(self.trajs) > 20:
            self.trajs.popitem(last=False)
        return out


class ArcTracker:
    """Monotone arc-length progress of the drone along a polyline (safe for loops where B == A)."""

    def __init__(self, path):
        self.p = np.asarray(path)[:, :2]
        seg = np.linalg.norm(np.diff(self.p, axis=0), axis=1)
        self.cum = np.r_[0, np.cumsum(seg)]
        self.i = 0
        self.s = 0.0

    def arc_of(self, xy, start=0):
        best, bs, bi = 1e18, 0.0, start
        for i in range(start, len(self.p) - 1):
            a, b = self.p[i], self.p[i + 1]
            ab = b - a
            t = float(np.clip((xy - a) @ ab / (ab @ ab + 1e-9), 0, 1))
            d = float(np.linalg.norm(a + t * ab - xy))
            if d < best - 1e-9:
                best, bs, bi = d, self.cum[i] + t * (self.cum[i + 1] - self.cum[i]), i
        return bs, bi

    def update(self, xy):
        best, bs, bi = 1e18, self.s, self.i
        for i in range(self.i, min(self.i + 3, len(self.p) - 1)):
            a, b = self.p[i], self.p[i + 1]
            ab = b - a
            t = float(np.clip((xy - a) @ ab / (ab @ ab + 1e-9), 0, 1))
            d = float(np.linalg.norm(a + t * ab - xy))
            if d < best:
                best, bs, bi = d, self.cum[i] + t * (self.cum[i + 1] - self.cum[i]), i
        self.i, self.s = bi, max(self.s, bs)
        return self.s


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
        self.default_gimbal = default_gimbal
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
            # optional per-POI activation radius (horizontal); the active one releases at radius + hysteresis
            d = [math.hypot(q["x"] - pos[0], q["y"] - pos[1]) for q in self.pois]
            ok = [q.get("radius_m") is None or d[i] <= q["radius_m"] + (self.NEAREST_HYST if i == self.active else 0.0)
                  for i, q in enumerate(self.pois)]
            if not any(ok):
                self.active = -1
                return
            k = min((i for i in range(len(d)) if ok[i]), key=lambda i: d[i])
            if self.active < 0 or not ok[self.active] or d[k] < d[self.active] - self.NEAREST_HYST:
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
        # only segments starting within a few metres ahead of the current progress: on a loop (B == A)
        # the last segment also passes through A, and snapping to it would end the flight at takeoff
        s_last = getattr(self, "_s_last", 0.0)
        best, bs, bt = 1e9, self.seg, 0.0
        for i in range(self.seg, min(self.seg + 3, len(self.path) - 1)):
            if i > self.seg and self.cum[i] > s_last + 3.0:
                break
            a, b = self.path[i, :2], self.path[i + 1, :2]
            ab = b - a
            t = float(np.clip((p - a) @ ab / (ab @ ab + 1e-9), 0, 1))
            d = float(np.linalg.norm(a + t * ab - p))
            if d < best:
                best, bs, bt = d, i, t
        self.seg = bs
        self._s_last = self.cum[bs] + bt * (self.cum[bs + 1] - self.cum[bs])
        return self._s_last

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
        yaw_max = P.yaw_rate_max
        if self.pois:  # look-at: yaw + gimbal track the active POI; translation keeps following the path
            # no POI in range: yaw follows travel (want_yaw above), gimbal eases back to the default
            want_yaw, g_target = self._poi_aim(pos, yaw) if self.active >= 0 else (want_yaw, self.default_gimbal)
            step = self.GIMBAL_RATE * self.dt
            self.gimbal_deg = float(np.clip(g_target, self.gimbal_deg - step, self.gimbal_deg + step))
            yaw_max = min(yaw_max, math.radians(self.GIMBAL_RATE))   # smooth look-at hand-offs (phone caps 60 deg/s too)
        c, s_ = math.cos(yaw), math.sin(yaw)
        fwd = v_xy[0] * c + v_xy[1] * s_
        right = v_xy[0] * s_ - v_xy[1] * c
        err = (want_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        r = float(np.clip(2.0 * err, -yaw_max, yaw_max))
        up = vz / P.v_up_max if vz >= 0 else vz / P.v_down_max
        return [float(np.clip(fwd / P.v_h_max, -1, 1)), float(np.clip(right / P.v_h_max, -1, 1)),
                float(np.clip(up, -1, 1)), float(np.clip(-r / P.yaw_rate_max, -1, 1))], False


# ---------------------------------------------------------------- FPV / fixed-wing style

def fillet_curve(path, speed, a_lat, clear_ok, ds=0.25, r_floor=0.5, max_dev=None):
    """Polyline -> dense curve with circular-arc fillets at interior vertices.
    Nominal radius R = v^2/a_lat, limited by half of each adjacent segment; each fillet is shrunk
    (x0.7) until its samples pass `clear_ok(points[N,3]) -> bool`. Returns (pts [M,3], kappa [M])."""
    P = np.asarray(path, float)
    if len(P) < 3:
        n = max(2, int(np.linalg.norm(P[-1] - P[0]) / ds) + 1)
        return np.linspace(P[0], P[-1], n), np.zeros(n)
    R_nom = speed ** 2 / a_lat
    fillets = []  # (t_in point, t_out point, center, R, turn sign) per interior vertex
    for i in range(1, len(P) - 1):
        u = P[i, :2] - P[i - 1, :2]; w = P[i + 1, :2] - P[i, :2]
        lu, lw = np.linalg.norm(u), np.linalg.norm(w)
        u, w = u / lu, w / lw
        th = math.acos(float(np.clip(u @ w, -1, 1)))
        if th < 1e-3:
            fillets.append(None); continue
        R = R_nom
        if max_dev and i in max_dev:  # user waypoint: the arc must pass within max_dev of the vertex
            sec = 1.0 / max(math.cos(th / 2), 1e-6)
            R = min(R, max_dev[i] / max(sec - 1.0, 1e-6))
        while True:
            L = R * math.tan(th / 2)
            L_max = 0.5 * min(lu, lw)
            if L > L_max:
                R = L_max / math.tan(th / 2); L = L_max
            t_in, t_out = P[i, :2] - u * L, P[i, :2] + w * L
            sign = 1.0 if (u[0] * w[1] - u[1] * w[0]) > 0 else -1.0
            nrm = np.array([-u[1], u[0]]) * sign
            c = t_in + nrm * R
            a0 = math.atan2(t_in[1] - c[1], t_in[0] - c[0])
            arc_n = max(2, int(R * th / ds) + 1)
            ang = a0 + sign * np.linspace(0, th, arc_n)
            arc = np.stack([c[0] + R * np.cos(ang), c[1] + R * np.sin(ang), np.full(arc_n, P[i, 2])], 1)
            if clear_ok(arc) or R <= r_floor:
                break
            R *= 0.7
        fillets.append((t_in, t_out, arc, R, sign))
    pts, kap = [], []
    cur = P[0, :2]
    for i in range(1, len(P)):
        f = fillets[i - 1] if i - 1 < len(fillets) else None
        end = f[0] if f is not None else P[i, :2]
        seg_n = max(2, int(np.linalg.norm(end - cur) / ds) + 1)
        seg = np.linspace(cur, end, seg_n)
        pts.append(np.c_[seg, np.full(seg_n, P[i, 2])]); kap.append(np.zeros(seg_n))
        if f is not None:
            pts.append(f[2]); kap.append(np.full(len(f[2]), 1.0 / f[3]))
            cur = f[1]
        else:
            cur = P[i, :2]
    pts = np.concatenate(pts); kap = np.concatenate(kap)
    keep = np.r_[True, np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1) > 1e-6]
    return pts[keep], kap[keep]


class FpvController:
    """Fixed-wing-like tracking of a curvature-limited curve: constant speed except where curvature
    (lateral accel or yaw-rate limit) or the final stop forces it, yaw always along the velocity
    tangent, climbing departure when clear. Same action space as PursuitController."""

    def __init__(self, curve, kappa, alt, speed, P: StickParams, land_z, a_xy, a_lat, gimbal_deg, depart_ok):
        self.c, self.k, self.alt, self.speed, self.P = curve, kappa, alt, speed, P
        self.land_z, self.a_xy, self.a_lat, self.depart_ok = land_z, np.array(a_xy), a_lat, depart_ok
        self.gimbal_deg, self.active = gimbal_deg, -1
        self.phase = "climb"
        seg = np.linalg.norm(np.diff(curve[:, :2], axis=0), axis=1)
        self.s = np.r_[0, np.cumsum(seg)]
        self.total = self.s[-1]
        k = np.maximum(kappa, 1e-6)
        vmax = np.minimum.reduce([np.full(len(k), speed), np.sqrt(a_lat / k), 0.9 * P.yaw_rate_max / k])
        vmax[-1] = 0.0
        a_f, a_b = 0.8 * P.a_acc, 0.6 * P.a_brake
        v = vmax.copy()
        for i in range(1, len(v)):           # forward pass (accel limit)
            v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * a_f * seg[i - 1]))
        for i in range(len(v) - 2, -1, -1):  # backward pass (brake ahead of time)
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * a_b * seg[i]))
        self.v = v
        self.idx = 0
        self.cur_kappa = 0.0

    def _project(self, p):
        lo, hi = self.idx, min(self.idx + 200, len(self.c))
        d = np.linalg.norm(self.c[lo:hi, :2] - p, axis=1)
        self.idx = lo + int(np.argmin(d))
        return self.idx

    def _vz(self, z, target):
        return float(np.clip(1.2 * (target - z), -self.P.v_down_max * 0.8, self.P.v_up_max * 0.8))

    def command(self, pos, vel, yaw):
        P = self.P
        v_xy, vz, want_yaw, ff_rate = np.zeros(2), 0.0, yaw, 0.0
        if self.phase == "climb":
            v_xy = np.clip(1.0 * (self.a_xy - pos[:2]), -1, 1)
            vz = self._vz(pos[2], self.alt)
            t0 = self.c[min(4, len(self.c) - 1), :2] - self.c[0, :2]
            want_yaw = math.atan2(t0[1], t0[0])  # face the first leg while climbing
            if abs(pos[2] - self.alt) < 0.3 or self.depart_ok(pos):
                self.phase = "cruise"
        if self.phase == "cruise":
            i = self._project(pos[:2])
            spd = float(np.linalg.norm(vel[:2]))
            look = 0.5 + 0.4 * spd
            j = int(np.searchsorted(self.s, self.s[i] + look))
            j = min(j, len(self.c) - 1)
            tgt = self.c[j, :2]
            tan = self.c[min(i + 1, len(self.c) - 1), :2] - self.c[max(i - 1, 0), :2]
            tan = tan / (np.linalg.norm(tan) + 1e-9)
            to = tgt - pos[:2]
            to = to / (np.linalg.norm(to) + 1e-9)
            vref = float(self.v[min(j, len(self.v) - 1)])
            rem_now = self.total - self.s[i]
            vref = max(vref, min(0.6, 0.8 * rem_now))  # creep in over the last metres (profile hits 0 at B)
            v_xy = to * vref
            self.cur_kappa = float(self.k[i])
            sgn = np.sign(tan[0] * to[1] - tan[1] * to[0]) or 1.0
            ff_rate = sgn * spd * self.cur_kappa
            want_yaw = math.atan2(to[1], to[0]) if spd > 0.3 or vref > 0.3 else yaw
            vz = self._vz(pos[2], self.alt)
            rem = self.total - self.s[i]
            if rem < 0.6:
                v_xy = np.clip(1.0 * (self.c[-1, :2] - pos[:2]), -0.6, 0.6)
                want_yaw = yaw
                if spd < 0.4:
                    self.phase = "descend" if self.land_z is not None else "done"
        elif self.phase == "descend":
            v_xy = np.clip(1.0 * (self.c[-1, :2] - pos[:2]), -0.6, 0.6)
            vz = self._vz(pos[2], self.land_z)
            if pos[2] - self.land_z < 1.5:
                vz = max(vz, -0.5)
            if pos[2] - self.land_z < 0.1 and abs(vel[2]) < 0.2:
                self.phase = "done"
        if self.phase == "done":
            return [0.0, 0.0, 0.0, 0.0], True
        c, s_ = math.cos(yaw), math.sin(yaw)
        fwd = v_xy[0] * c + v_xy[1] * s_
        right = v_xy[0] * s_ - v_xy[1] * c
        err = (want_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        r = float(np.clip(ff_rate + 2.0 * err, -P.yaw_rate_max, P.yaw_rate_max))
        up = vz / P.v_up_max if vz >= 0 else vz / P.v_down_max
        return [float(np.clip(fwd / P.v_h_max, -1, 1)), float(np.clip(right / P.v_h_max, -1, 1)),
                float(np.clip(up, -1, 1)), float(np.clip(-r / P.yaw_rate_max, -1, 1))], False
