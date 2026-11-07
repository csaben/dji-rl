#!/usr/bin/env python3
"""Coloured point cloud (PLY x,y,z,red,green,blue) -> 3DGS splat PLY usable as a djirl scene.

Each point becomes a small isotropic Gaussian sized to the local point spacing. Not a trained splat
(no view-dependent colour, no learned shapes) but it renders and bakes collisions the same way.

    python -I tools/pointcloud_to_splat.py IN.ply OUT.ply --up -y --scale 100 \
        --crop-center 0 0 --crop-size 150 --voxel 0.08
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch  # noqa: E402

from djirl.scene import SH_C0, Gaussians, save_ply  # noqa: E402

AXES = {"x": (0, 1), "-x": (0, -1), "y": (1, 1), "-y": (1, -1), "z": (2, 1), "-z": (2, -1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--up", default="z", help="which input axis points up: x,-x,y,-y,z,-z")
    ap.add_argument("--scale", type=float, default=1.0, help="input units -> metres")
    ap.add_argument("--crop-center", type=float, nargs=2, help="EN centre (m, after scale) of the slice to keep")
    ap.add_argument("--crop-size", type=float, default=150.0, help="side length (m) of the square slice")
    ap.add_argument("--voxel", type=float, default=0.08, help="downsample to one point per voxel (m)")
    ap.add_argument("--opacity", type=float, default=0.9)
    ap.add_argument("--ground-fill", type=float, default=0.3,
                    help="add a flat ground layer at this spacing (m) under holes; 0 = off")
    a = ap.parse_args()

    v = PlyData.read(a.inp)["vertex"]
    p = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64) * a.scale
    rgb = np.stack([v["red"], v["green"], v["blue"]], -1).astype(np.float32) / 255.0

    # build a right-handed ENU basis with the chosen up axis
    ui, us = AXES[a.up]
    up = np.zeros(3); up[ui] = us
    e = np.zeros(3); e[(ui + 1) % 3] = 1.0
    n = np.cross(up, e)
    p = p @ np.stack([e, n, up], 0).T

    if a.crop_center is not None:
        c = np.array(a.crop_center)
    else:
        c = np.median(p[:, :2], 0)
    half = a.crop_size / 2
    keep = (np.abs(p[:, 0] - c[0]) < half) & (np.abs(p[:, 1] - c[1]) < half)
    p, rgb = p[keep], rgb[keep]
    p[:, :2] -= c
    # street level = densest 0.5 m height band in the lowest 30 m
    lo_z = np.percentile(p[:, 2], 1)
    h, edges = np.histogram(p[:, 2], bins=np.arange(lo_z, lo_z + 30, 0.5))
    ground = edges[h.argmax()]
    p[:, 2] -= ground
    keep = p[:, 2] > -1.0  # drop stray points below the street
    p, rgb = p[keep], rgb[keep]

    # voxel downsample (keep the first point in each voxel)
    key = np.floor(p / a.voxel).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    p, rgb = p[first], rgb[first]

    if a.ground_fill > 0:
        gs = a.ground_fill
        xs = np.arange(-half, half, gs)
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        q = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], -1)
        low = p[np.abs(p[:, 2]) < 0.5]
        dist, nn = cKDTree(low[:, :2]).query(q[:, :2], k=1)
        hole = dist > gs * 1.5
        fill_rgb = np.where(dist[:, None] < 3.0, rgb[np.abs(p[:, 2]) < 0.5][nn], 0.35)[hole]
        p = np.concatenate([p, q[hole] - [0, 0, 0.05]])
        rgb = np.concatenate([rgb, fill_rgb.astype(np.float32)])
    d, _ = cKDTree(p).query(p, k=4)
    s = np.clip(d[:, 1:].mean(1) * 0.7, a.voxel * 0.4, a.voxel * 3)
    N = len(p)
    g = Gaussians(torch.from_numpy(p).float(), torch.tensor([1.0, 0, 0, 0]).repeat(N, 1),
                  torch.from_numpy(np.repeat(s[:, None], 3, 1)).float(), torch.full((N,), a.opacity),
                  torch.from_numpy((rgb - 0.5) / SH_C0)[:, None, :].float())
    save_ply(g, Path(a.out))
    lo, hi = p.min(0), p.max(0)
    print(f"{N} gaussians, extent {np.round(hi - lo, 1)} m -> {a.out}")


if __name__ == "__main__":
    main()
