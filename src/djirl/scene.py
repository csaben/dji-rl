"""Scene = Gaussian splat (for the camera) + ESDF voxel grid (for collisions) + waypoint loop.

World frame is ENU in metres: x east, y north, z up, origin at the home point on the ground.
A scene lives in data/splats/<name>/:
    splat.ply     standard 3DGS PLY (x,y,z, f_dc_*, f_rest_*, opacity, scale_*, rot_*)
    esdf.npz      sdf [X,Y,Z] float16 metres (+ outside, - inside), origin [3], voxel float
    scene.json    {"waypoints": [[x,y,z], ...], "transform": 4x4 splat->ENU, "notes": ...}
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SH_C0 = 0.28209479177387814


@dataclass
class Gaussians:
    means: torch.Tensor      # [N,3]
    quats: torch.Tensor      # [N,4] wxyz, unit
    scales: torch.Tensor     # [N,3] linear
    opacities: torch.Tensor  # [N] in (0,1)
    sh: torch.Tensor         # [N,K,3] SH coefficients, K = (deg+1)^2

    @property
    def sh_degree(self) -> int:
        return int(math.isqrt(self.sh.shape[1])) - 1

    def to(self, device) -> "Gaussians":
        return Gaussians(*(t.to(device) for t in (self.means, self.quats, self.scales, self.opacities, self.sh)))

    def transformed(self, T: np.ndarray) -> "Gaussians":
        """Apply a similarity transform (4x4, rotation*scale + translation). SH beyond DC are not rotated."""
        T = torch.as_tensor(T, dtype=torch.float32, device=self.means.device)
        A, t = T[:3, :3], T[:3, 3]
        s = torch.linalg.det(A).abs() ** (1 / 3)
        R = A / s
        means = self.means @ A.T + t
        q = _rotmat_to_quat(R)
        quats = _quat_mul(q.expand_as(self.quats), self.quats)
        return Gaussians(means, quats, self.scales * s, self.opacities, self.sh)


@dataclass
class ESDF:
    sdf: torch.Tensor    # [X,Y,Z]
    origin: torch.Tensor  # [3] world position of voxel (0,0,0) centre
    voxel: float

    def to(self, device) -> "ESDF":
        return ESDF(self.sdf.to(device), self.origin.to(device), self.voxel)

    @property
    def upper(self) -> torch.Tensor:
        return self.origin + self.voxel * (torch.tensor(self.sdf.shape, device=self.sdf.device) - 1)

    def query(self, p: torch.Tensor) -> torch.Tensor:
        """Trilinear SDF at world points p [...,3]. Outside the grid -> distance to the box is not
        known, so we return the border value (callers treat out-of-bounds separately)."""
        shp = p.shape[:-1]
        n = torch.tensor(self.sdf.shape, device=p.device, dtype=p.dtype)
        g = (p.reshape(-1, 3) - self.origin) / self.voxel / (n - 1) * 2 - 1   # [M,3] in [-1,1] (x,y,z)
        vol = self.sdf.permute(2, 1, 0)[None, None].to(p.dtype)                # [1,1,Z,Y,X]
        out = F.grid_sample(vol, g[None, :, None, None, :], mode="bilinear",
                            padding_mode="border", align_corners=True)
        return out.reshape(shp)

    def in_bounds(self, p: torch.Tensor) -> torch.Tensor:
        return ((p >= self.origin) & (p <= self.upper)).all(-1)


@dataclass
class Scene:
    name: str
    gaussians: Gaussians
    esdf: ESDF
    waypoints: torch.Tensor  # [W,3] ENU, a closed loop (last connects back to first)

    def to(self, device) -> "Scene":
        return Scene(self.name, self.gaussians.to(device), self.esdf.to(device), self.waypoints.to(device))


# ---------------------------------------------------------------- PLY io

def load_ply(path: Path) -> Gaussians:
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"]
    names = v.data.dtype.names
    f = lambda k: np.asarray(v[k], dtype=np.float32)
    means = np.stack([f("x"), f("y"), f("z")], -1)
    dc = np.stack([f(f"f_dc_{i}") for i in range(3)], -1)[:, None, :]
    rest_names = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    if rest_names:
        rest = np.stack([f(n) for n in rest_names], -1).reshape(len(means), 3, -1).transpose(0, 2, 1)
        sh = np.concatenate([dc, rest], 1)
    else:
        sh = dc
    opac = 1 / (1 + np.exp(-f("opacity")))
    scales = np.exp(np.stack([f(f"scale_{i}") for i in range(3)], -1))
    quats = np.stack([f(f"rot_{i}") for i in range(4)], -1)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-12
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    return Gaussians(t(means), t(quats), t(scales), t(opac), t(sh))


def save_ply(g: Gaussians, path: Path) -> None:
    from plyfile import PlyData, PlyElement
    n, k = g.sh.shape[:2]
    sh = g.sh.detach().cpu().numpy()
    cols = {"x": 0, "y": 1, "z": 2}
    fields = [(c, "f4") for c in ("x", "y", "z")] + [(f"f_dc_{i}", "f4") for i in range(3)]
    fields += [(f"f_rest_{i}", "f4") for i in range(3 * (k - 1))]
    fields += [("opacity", "f4")] + [(f"scale_{i}", "f4") for i in range(3)] + [(f"rot_{i}", "f4") for i in range(4)]
    arr = np.empty(n, dtype=fields)
    m = g.means.detach().cpu().numpy()
    for c, i in cols.items():
        arr[c] = m[:, i]
    for i in range(3):
        arr[f"f_dc_{i}"] = sh[:, 0, i]
    rest = sh[:, 1:, :].transpose(0, 2, 1).reshape(n, -1)
    for i in range(rest.shape[1]):
        arr[f"f_rest_{i}"] = rest[:, i]
    o = g.opacities.detach().cpu().numpy().clip(1e-6, 1 - 1e-6)
    arr["opacity"] = np.log(o / (1 - o))
    s = np.log(g.scales.detach().cpu().numpy())
    q = g.quats.detach().cpu().numpy()
    for i in range(3):
        arr[f"scale_{i}"] = s[:, i]
    for i in range(4):
        arr[f"rot_{i}"] = q[:, i]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


# ---------------------------------------------------------------- ESDF baking

@torch.no_grad()
def bake_esdf(g: Gaussians, voxel: float = 0.1, pad: float = 2.0, min_opacity: float = 0.3,
              min_neighbors: int = 2, bounds: tuple | None = None, max_scale: float | None = None) -> ESDF:
    """Occupancy from opaque Gaussians (centre + points on their 1.5-sigma ellipsoid), isolated-voxel
    (floater) removal, then a signed Euclidean distance transform."""
    from scipy.ndimage import distance_transform_edt
    dev = g.means.device
    keep = g.opacities > min_opacity
    if max_scale is not None:  # big soft blobs are sky/background/floaters, not surfaces
        keep &= g.scales.max(-1).values < max_scale
    means, quats, scales = g.means[keep], g.quats[keep], g.scales[keep]
    if bounds is None:
        lo, hi = means.min(0).values - pad, means.max(0).values + pad
    else:
        lo, hi = (torch.tensor(b, device=dev, dtype=torch.float32) for b in bounds)
    dims = torch.ceil((hi - lo) / voxel).long() + 1
    occ = torch.zeros(*dims.tolist(), dtype=torch.bool, device=dev)
    dirs = torch.tensor([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
                        dtype=torch.float32, device=dev)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1)  # centre + 26 unit-ish directions
    R = _quat_to_rotmat(quats)
    for s in range(0, len(means), 200_000):
        sl = slice(s, s + 200_000)
        local = dirs[None] * scales[sl, None, :] * 1.5                    # [n,27,3]
        pts = means[sl, None, :] + torch.einsum("nij,nkj->nki", R[sl], local)
        idx = torch.round((pts.reshape(-1, 3) - lo) / voxel).long()
        ok = ((idx >= 0) & (idx < dims)).all(-1)
        idx = idx[ok]
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    if min_neighbors > 0:
        nb = F.conv3d(occ[None, None].float(), torch.ones(1, 1, 3, 3, 3, device=dev), padding=1)[0, 0] - occ.float()
        occ &= nb >= min_neighbors
    o = occ.cpu().numpy()
    sdf = (distance_transform_edt(~o) - distance_transform_edt(o)) * voxel
    return ESDF(torch.from_numpy(sdf.astype(np.float32)), lo.cpu().float(), voxel)


# ---------------------------------------------------------------- scene dir io

def save_scene(scene: Scene, root: Path, transform=None, notes: str = "") -> Path:
    d = Path(root) / scene.name
    d.mkdir(parents=True, exist_ok=True)
    save_ply(scene.gaussians, d / "splat.ply")
    np.savez_compressed(d / "esdf.npz", sdf=scene.esdf.sdf.cpu().numpy().astype(np.float16),
                        origin=scene.esdf.origin.cpu().numpy(), voxel=scene.esdf.voxel)
    meta = {"waypoints": scene.waypoints.cpu().tolist(),
            "transform": (np.eye(4) if transform is None else np.asarray(transform)).tolist(),
            "notes": notes}
    (d / "scene.json").write_text(json.dumps(meta, indent=2))
    return d


def load_scene(d: Path) -> Scene:
    d = Path(d)
    meta = json.loads((d / "scene.json").read_text())
    g = load_ply(d / "splat.ply")
    T = np.asarray(meta.get("transform", np.eye(4)))
    if not np.allclose(T, np.eye(4)):
        g = g.transformed(T)
    z = np.load(d / "esdf.npz")
    esdf = ESDF(torch.from_numpy(z["sdf"].astype(np.float32)), torch.from_numpy(z["origin"]).float(), float(z["voxel"]))
    return Scene(d.name, g, esdf, torch.tensor(meta["waypoints"], dtype=torch.float32))


# ---------------------------------------------------------------- quaternion helpers

def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def _rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    w = torch.sqrt((1 + R[0, 0] + R[1, 1] + R[2, 2]).clamp_min(1e-12)) / 2
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    return F.normalize(torch.stack([w, x, y, z]), dim=0)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)
