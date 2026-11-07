"""Unmapped obstacles: primitives the global planner doesn't know about, injected into a scene so the
local policy has to see them (rendered as Gaussians) and avoid them (analytic SDF unioned with the
scene ESDF). Types: pole (vertical cylinder), wall (vertical box), tree (trunk + canopy sphere)."""
from __future__ import annotations

import math

import torch

from .scene import SH_C0, Gaussians

# natural-ish palette (things a real camera sees near houses): concrete, wood, brick, stucco, dark paint, foliage
PALETTE = [(0.55, 0.55, 0.53), (0.42, 0.33, 0.24), (0.55, 0.30, 0.22), (0.80, 0.78, 0.72), (0.20, 0.21, 0.23),
           (0.30, 0.36, 0.30), (0.65, 0.60, 0.50), (0.35, 0.35, 0.38)]

POLE, WALL, TREE = 0, 1, 2


class Obstacles:
    """K primitives on one device. Params per obstacle (all world ENU, metres):
    kind, cx, cy, z0 (base), z1 (top), r (pole radius / trunk radius), w, d, yaw (wall), canopy_r, canopy_z."""

    def __init__(self, device):
        self.dev = torch.device(device)
        self.p = torch.zeros(0, 11, device=self.dev)
        self.gauss: Gaussians | None = None

    @property
    def n(self):
        return len(self.p)

    # -------------------------------------------------------------- sampling
    def sample_at(self, centers: torch.Tensor, alts: torch.Tensor, ground: torch.Tensor, gen=None):
        """One obstacle per center [K,2], tall enough to block flight at alts [K] (above ground [K])."""
        K, dev = len(centers), self.dev
        r = lambda *s: torch.rand(*s, device=dev, generator=gen)
        kind = torch.randint(0, 3, (K,), device=dev, generator=gen).float()
        top = alts + 0.5 + 3.0 * r(K)
        p = torch.zeros(K, 11, device=dev)
        p[:, 0] = kind
        p[:, 1:3] = centers
        p[:, 3] = ground
        p[:, 4] = top
        p[:, 5] = torch.where(kind == POLE, 0.15 + 0.6 * r(K), 0.12 + 0.2 * r(K))   # pole r / trunk r
        p[:, 6] = 1.0 + 4.0 * r(K)                                                  # wall width
        p[:, 7] = 0.2 + 1.5 * r(K)                                                  # wall depth
        p[:, 8] = r(K) * math.pi                                                    # wall yaw
        p[:, 9] = 1.2 + 2.0 * r(K)                                                  # canopy radius
        p[:, 10] = alts + (r(K) - 0.5) * 2.0                                        # canopy centre z
        self.p = p
        self.gauss = self._gaussians(gen)

    # -------------------------------------------------------------- collision
    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        """Signed distance [M] from points [M,3] to the union of obstacles (+inf if none)."""
        if self.n == 0:
            return torch.full(x.shape[:-1], 1e3, device=x.device)
        out = torch.full(x.shape[:-1], 1e3, device=x.device)
        P = self.p
        for i0 in range(0, len(x), 262144):
            xs = x[i0:i0 + 262144][:, None, :]                                       # [m,1,3]
            dx, dy = xs[..., 0] - P[:, 1], xs[..., 1] - P[:, 2]                      # [m,K]
            z = xs[..., 2]
            zc = (P[:, 3] + P[:, 4]) / 2
            hz = (P[:, 4] - P[:, 3]) / 2
            # vertical capped cylinder (pole, or trunk up to the canopy)
            dr = torch.hypot(dx, dy) - P[:, 5]
            dzc = (z - zc).abs() - hz
            cyl = torch.hypot(dr.clamp(min=0), dzc.clamp(min=0)) + torch.maximum(dr, dzc).clamp(max=0)
            # oriented box (wall)
            c, s = torch.cos(P[:, 8]), torch.sin(P[:, 8])
            lx, ly = (c * dx + s * dy).abs() - P[:, 6] / 2, (-s * dx + c * dy).abs() - P[:, 7] / 2
            q = torch.stack([lx, ly, dzc], -1)
            box = q.clamp(min=0).norm(dim=-1) + q.max(-1).values.clamp(max=0)
            # canopy sphere
            sph = torch.sqrt(dx ** 2 + dy ** 2 + (z - P[:, 10]) ** 2) - P[:, 9]
            k = P[:, 0]
            d = torch.where(k == WALL, box, torch.where(k == TREE, torch.minimum(cyl, sph), cyl))
            out[i0:i0 + 262144] = d.min(-1).values
        return out

    # -------------------------------------------------------------- rendering
    def _gaussians(self, gen=None, spacing=0.12) -> Gaussians:
        dev, P = self.dev, self.p
        pts, cols = [], []
        r = lambda *s: torch.rand(*s, device=dev, generator=gen)
        for i in range(self.n):
            kind, cx, cy, z0, z1, rad, w, d, yaw, cr, cz = P[i].tolist()
            h = z1 - z0
            if kind in (POLE, TREE):
                rr = rad
                nz, nth = max(2, int(h / spacing)), max(6, int(2 * math.pi * rr / spacing))
                th = torch.linspace(0, 2 * math.pi, nth, device=dev)[:-1]
                zz = torch.linspace(z0, z1 if kind == POLE else min(z1, cz), nz, device=dev)
                T, Z = torch.meshgrid(th, zz, indexing="ij")
                p = torch.stack([cx + rr * torch.cos(T), cy + rr * torch.sin(T), Z], -1).reshape(-1, 3)
                base = (torch.tensor(PALETTE[int(torch.randint(0, len(PALETTE), (1,), device=dev, generator=gen))], device=dev)
                        if kind == POLE else torch.tensor([0.35, 0.25, 0.15], device=dev)) * (0.8 + 0.4 * r(1))
                pts.append(p); cols.append(base.expand(len(p), 3) * (0.85 + 0.3 * r(len(p), 1)))
            if kind == TREE:
                n = int(4 * math.pi * cr * cr / spacing ** 2)
                v = torch.nn.functional.normalize(torch.randn(n, 3, device=dev, generator=gen), dim=-1)
                p = torch.tensor([cx, cy, cz], device=dev) + v * cr * (0.85 + 0.15 * r(n, 1))
                g = torch.tensor([0.15, 0.30, 0.12], device=dev) * (0.7 + 0.5 * r(1))
                pts.append(p); cols.append(g.expand(n, 3) * (0.7 + 0.6 * r(n, 1)))
            if kind == WALL:
                c, s = math.cos(yaw), math.sin(yaw)
                faces = []
                for sgn in (-1, 1):  # two big faces + two ends + top
                    a = torch.linspace(-w / 2, w / 2, max(2, int(w / spacing)), device=dev)
                    zz = torch.linspace(z0, z1, max(2, int(h / spacing)), device=dev)
                    A, Z = torch.meshgrid(a, zz, indexing="ij")
                    lx, ly = A, torch.full_like(A, sgn * d / 2)
                    faces.append(torch.stack([cx + c * lx - s * ly, cy + s * lx + c * ly, Z], -1).reshape(-1, 3))
                    b = torch.linspace(-d / 2, d / 2, max(2, int(d / spacing)), device=dev)
                    B, Z = torch.meshgrid(b, zz, indexing="ij")
                    lx, ly = torch.full_like(B, sgn * w / 2), B
                    faces.append(torch.stack([cx + c * lx - s * ly, cy + s * lx + c * ly, Z], -1).reshape(-1, 3))
                p = torch.cat(faces)
                base = torch.tensor(PALETTE[int(torch.randint(0, len(PALETTE), (1,), device=dev, generator=gen))], device=dev)
                pts.append(p); cols.append(base.expand(len(p), 3) * (0.85 + 0.3 * r(len(p), 1)))
        if not pts:
            return None
        means = torch.cat(pts)
        rgb = torch.cat(cols).clamp(0, 1)
        n = len(means)
        return Gaussians(means, torch.tensor([1.0, 0, 0, 0], device=dev).repeat(n, 1),
                         torch.full((n, 3), spacing * 0.7, device=dev), torch.full((n,), 0.95, device=dev),
                         ((rgb - 0.5) / SH_C0)[:, None, :])


def merge(scene_g: Gaussians, extra: Gaussians | None) -> Gaussians:
    """Scene + obstacle Gaussians; obstacle SH is padded with zeros up to the scene's SH degree."""
    if extra is None:
        return scene_g
    K = scene_g.sh.shape[1]
    sh = torch.zeros(len(extra.means), K, 3, device=extra.sh.device)
    sh[:, :1] = extra.sh[:, :1]
    return Gaussians(torch.cat([scene_g.means, extra.means]), torch.cat([scene_g.quats, extra.quats]),
                     torch.cat([scene_g.scales, extra.scales]), torch.cat([scene_g.opacities, extra.opacities]),
                     torch.cat([scene_g.sh, sh]))
