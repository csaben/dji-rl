"""A procedural stand-in neighbourhood (ground, house, trees, fence posts) so the harness runs
before real footage exists. Collision geometry is baked from the splat exactly like a real scene."""
from __future__ import annotations

import math

import torch

from .scene import SH_C0, Gaussians, Scene, bake_esdf


def _surface(points, color, spacing, normal_axis=None, jitter=0.08, gen=None):
    n = len(points)
    rgb = (torch.tensor(color).float()[None] * (1 + jitter * torch.randn(n, 3, generator=gen))).clamp(0, 1)
    s = torch.full((n, 3), spacing * 0.75)
    if normal_axis is not None:
        s[:, normal_axis] = 0.02
    return points, s, rgb


def _grid(a0, a1, b0, b1, step):
    a = torch.arange(a0, a1 + 1e-6, step)
    b = torch.arange(b0, b1 + 1e-6, step)
    A, B = torch.meshgrid(a, b, indexing="ij")
    return A.reshape(-1), B.reshape(-1)


def _box(cx, cy, w, d, h, color, step, gen):
    parts = []
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - d / 2, cy + d / 2
    for y in (y0, y1):
        X, Z = _grid(x0, x1, 0, h, step)
        parts.append(_surface(torch.stack([X, torch.full_like(X, y), Z], -1), color, step, 1, gen=gen))
    for x in (x0, x1):
        Y, Z = _grid(y0, y1, 0, h, step)
        parts.append(_surface(torch.stack([torch.full_like(Y, x), Y, Z], -1), color, step, 0, gen=gen))
    X, Y = _grid(x0, x1, y0, y1, step)
    parts.append(_surface(torch.stack([X, Y, torch.full_like(X, h)], -1), (0.45, 0.25, 0.2), step, 2, gen=gen))
    return parts


def _sphere(c, r, color, step, gen):
    n = int(4 * math.pi * r * r / step ** 2)
    v = torch.nn.functional.normalize(torch.randn(n, 3, generator=gen), dim=-1)
    rr = r * (0.85 + 0.15 * torch.rand(n, 1, generator=gen))  # leafy, slightly volumetric
    return _surface(torch.tensor(c)[None] + v * rr, color, step, None, jitter=0.2, gen=gen)


def _cylinder(cx, cy, r, h, color, step, gen):
    th, z = _grid(0, 2 * math.pi, 0, h, step / max(r, 0.05))
    z = z * max(r, 0.05)
    z = z[z <= h]
    th = th[: len(z)]
    return _surface(torch.stack([cx + r * torch.cos(th), cy + r * torch.sin(th), z], -1), color, step, None, gen=gen)


def make_synthetic(seed: int = 0, voxel: float = 0.1, device: str = "cuda") -> Scene:
    gen = torch.Generator().manual_seed(seed)
    parts = []
    X, Y = _grid(-35, 35, -35, 35, 0.3)
    parts.append(_surface(torch.stack([X, Y, torch.zeros_like(X)], -1), (0.30, 0.45, 0.22), 0.3, 2, gen=gen))
    parts += _box(0, 0, 10, 8, 6, (0.80, 0.78, 0.72), 0.15, gen)          # house
    parts += _box(9.5, -6, 4, 6, 3, (0.55, 0.55, 0.58), 0.15, gen)        # garage
    # trees placed on/near the waypoint loop so the course needs avoiding
    trees = [(14, 3, 2.2, 6), (-4, 13, 2.5, 7), (-13, -5, 2.0, 5.5), (3, -14, 2.4, 6.5), (11, 11, 1.8, 5)]
    for cx, cy, r, h in trees:
        parts.append(_cylinder(cx, cy, 0.25, h - r, (0.35, 0.25, 0.15), 0.1, gen))
        parts.append(_sphere((cx, cy, h - r * 0.6), r, (0.15, 0.40, 0.12), 0.15, gen))
    for i in range(10):  # fence posts along the north side
        parts.append(_cylinder(-12 + 2.6 * i, 18, 0.08, 1.4, (0.5, 0.4, 0.3), 0.06, gen))
    means = torch.cat([p[0] for p in parts]).float()
    scales = torch.cat([p[1] for p in parts]).float()
    rgb = torch.cat([p[2] for p in parts]).float()
    n = len(means)
    g = Gaussians(means, torch.tensor([1.0, 0, 0, 0]).repeat(n, 1), scales,
                  torch.full((n,), 0.95), ((rgb - 0.5) / SH_C0)[:, None, :]).to(device)
    esdf = bake_esdf(g, voxel=voxel, bounds=((-36, -36, -1), (36, 36, 25)))
    r, alt = 15.0, 4.0
    wps = torch.tensor([[r * math.cos(a), r * math.sin(a), alt + (i % 2)]
                        for i, a in enumerate(torch.linspace(0, 2 * math.pi, 7)[:-1].tolist())])
    return Scene("synthetic", g, esdf, wps)
