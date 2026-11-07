"""Batched camera rendering of the splat with gsplat, from drone pose (ENU position, yaw, gimbal pitch)."""
from __future__ import annotations

import math

import torch
from gsplat import rasterization

from .scene import Gaussians


def viewmats(pos: torch.Tensor, yaw: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    """World->camera (OpenCV: x right, y down, z forward). pitch < 0 looks down."""
    cy, sy, cp, sp = torch.cos(yaw), torch.sin(yaw), torch.cos(pitch), torch.sin(pitch)
    fwd = torch.stack([cy * cp, sy * cp, sp], -1)
    right = torch.stack([sy, -cy, torch.zeros_like(cy)], -1)
    down = torch.cross(fwd, right, dim=-1)
    R = torch.stack([right, down, fwd], -2)          # rows = camera axes in world coords
    t = -(R @ pos[..., None])[..., 0]
    V = torch.zeros(*pos.shape[:-1], 4, 4, device=pos.device)
    V[..., :3, :3], V[..., :3, 3], V[..., 3, 3] = R, t, 1
    return V


class Renderer:
    def __init__(self, g: Gaussians, width: int = 128, height: int = 96, hfov_deg: float = 82.0,
                 chunk: int = 128, sky=(0.62, 0.75, 0.90), near: float = 0.05, far: float = 200.0,
                 radius_clip: float = 0.0, full_frac: float = 1.0):
        """radius_clip: skip Gaussians whose projected radius is <= this many px (2.0 is ~3.5x faster on
        large scenes at 128x96, loses fine texture). full_frac: fraction of camera chunks rendered
        without the clip anyway, so training still sees full texture."""
        self.g, self.W, self.H, self.chunk = g, width, height, chunk
        f = width / 2 / math.tan(math.radians(hfov_deg) / 2)
        self.K = torch.tensor([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], device=g.means.device)
        self.sky = torch.tensor(sky, device=g.means.device)
        self.near, self.far = near, far
        self.radius_clip, self.full_frac = radius_clip, full_frac

    @torch.no_grad()
    def render(self, pos, yaw, pitch, depth: bool = False) -> torch.Tensor:
        """-> [B,H,W,3] RGB in [0,1] (or [B,H,W,4] with expected depth appended)."""
        V = viewmats(pos, yaw, pitch)
        outs = []
        g = self.g
        for s in range(0, len(V), self.chunk):
            v = V[s:s + self.chunk]
            c = len(v)
            img, alpha, _ = rasterization(
                g.means, g.quats, g.scales, g.opacities, g.sh, v, self.K.expand(c, 3, 3), self.W, self.H,
                sh_degree=g.sh_degree, near_plane=self.near, far_plane=self.far,
                radius_clip=0.0 if (self.radius_clip == 0 or torch.rand(()) < self.full_frac) else self.radius_clip,
                render_mode="RGB+ED" if depth else "RGB",
            )
            img[..., :3] = img[..., :3] + (1 - alpha) * self.sky  # sky composite (gsplat's packed
            outs.append(img)                                        # mode rejects `backgrounds`)
        out = torch.cat(outs)
        out[..., :3] = out[..., :3].clamp(0, 1)
        return out
