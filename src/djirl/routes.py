"""Route library: map-valid A* routes (same planner as /api/sim/fly) at several altitudes, densified
to 0.5 m, for the route-following + unmapped-obstacle task.

    .venv/bin/djirl routes --scene house-worldlabs-refined --n 400
-> data/splats/<scene>/routes.pt  {"pts": [R,L,3] float, "len": [R] int, "alt": [R], "spacing": 0.5}
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

SPACING = 0.5


def densify(path: np.ndarray, ds: float = SPACING) -> np.ndarray:
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / ds)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)


def build(scene_name: str, n: int = 400, alts=(3.0, 5.0, 8.0, 12.0), clear: float = 1.5, d_range=(25.0, 90.0),
          coverage: float = 20.0, device="cuda", seed: int = 0) -> Path:
    from .sim_api import SPLATS, SimAPI
    api = SimAPI(device)
    s = api.scene(scene_name)
    rng = np.random.default_rng(seed)
    e = s.esdf
    lo, hi = e.origin.cpu().numpy(), e.upper.cpu().numpy()
    wps = s.waypoints[:, :2].cpu().numpy()

    def near_capture(p):  # only where the splat is good: within `coverage` m of the capture path
        a, b = wps, np.roll(wps, -1, 0)
        ab = b - a
        t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
        return np.min(np.linalg.norm(a + t[:, None] * ab - p, axis=1)) < coverage

    routes, t0, tries = [], time.time(), 0
    while len(routes) < n and tries < n * 20:
        tries += 1
        alt = float(rng.choice(alts))
        a = rng.uniform(lo[:2] + 2, hi[:2] - 2)
        b = rng.uniform(lo[:2] + 2, hi[:2] - 2)
        if not (d_range[0] < np.linalg.norm(b - a) < d_range[1]) or not (near_capture(a) and near_capture(b)):
            continue
        path, _, why = api.plan(s, tuple(a), tuple(b), alt, clear)
        if path is None:
            continue
        dense = densify(path)
        routes.append((dense.astype(np.float32), alt))
        if len(routes) % 50 == 0:
            print(f"routes: {len(routes)}/{n} ({tries} tries, {time.time() - t0:.0f}s)", flush=True)
    L = max(len(r) for r, _ in routes)
    pts = np.zeros((len(routes), L, 3), np.float32)
    lens = np.zeros(len(routes), np.int64)
    for i, (r, _) in enumerate(routes):
        pts[i, :len(r)] = r
        pts[i, len(r):] = r[-1]
        lens[i] = len(r)
    out = SPLATS / scene_name / "routes.pt"
    torch.save({"pts": torch.from_numpy(pts), "len": torch.from_numpy(lens),
                "alt": torch.tensor([a for _, a in routes]), "spacing": SPACING, "clear": clear}, out)
    print(f"saved {len(routes)} routes (max {L} pts = {L * SPACING:.0f} m) -> {out}", flush=True)
    return out
