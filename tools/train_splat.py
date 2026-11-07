#!/usr/bin/env python3
"""SfM model -> trained Gaussian splat, georeferenced to ENU with the drone GPS, ready for djirl.

    .venv/bin/python tools/train_splat.py data/recon/poc --steps 7000 --downscale 2 --name house-poc

Steps:
  1. undistort the largest COLMAP model to PINHOLE (pycolmap)
  2. train 3DGS with gsplat (DefaultStrategy densification, L1 + SSIM)
  3. fit a similarity transform SfM -> ENU from drone frames that have GPS (frames.jsonl next to the
     source frames; lat/lon -> local metres around the first fix, up = rel_alt above takeoff)
  4. write data/splats/_src/<name>.ply, <name>.transform.json, <name>.waypoints.json
     then: djirl import-ply data/splats/_src/<name>.ply --name <name> --transform ... --waypoints ...
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from djirl.scene import SH_C0, Gaussians, save_ply  # noqa: E402


# ------------------------------------------------------------------ data
def largest_model(sparse: Path) -> Path:
    best, n = None, -1
    for d in sorted(p for p in sparse.iterdir() if p.is_dir()):
        r = pycolmap.Reconstruction(d)
        if r.num_reg_images() > n:
            best, n = d, r.num_reg_images()
    return best


def load_views(und: Path, downscale: int, device, max_side: int = 0):
    r = pycolmap.Reconstruction(und / "sparse")
    views = []
    for img in sorted(r.images.values(), key=lambda i: i.name):
        cam = r.cameras[img.camera_id]
        fx, fy, cx, cy = cam.params[:4] if len(cam.params) >= 4 else (cam.params[0], cam.params[0], cam.params[1], cam.params[2])
        im = Image.open(und / "images" / img.name).convert("RGB")
        f = float(downscale)
        if max_side:
            f = max(f, max(im.width, im.height) / max_side)
        W, H = round(im.width / f), round(im.height / f)
        im = im.resize((W, H), Image.BILINEAR)
        sx, sy = W / cam.width, H / cam.height
        K = torch.tensor([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], dtype=torch.float32)
        T = img.cam_from_world() if callable(getattr(img, "cam_from_world", None)) else img.cam_from_world
        M = np.eye(4)
        M[:3, :4] = T.matrix()
        views.append({"name": img.name, "K": K.to(device), "viewmat": torch.tensor(M, dtype=torch.float32, device=device),
                      "rgb": torch.from_numpy(np.asarray(im).copy()).pin_memory(), "W": W, "H": H,
                      "center": -M[:3, :3].T @ M[:3, 3]})
    pts = np.array([p.xyz for p in r.points3D.values()])
    cols = np.array([p.color for p in r.points3D.values()]) / 255.0
    return views, pts, cols


# ------------------------------------------------------------------ loss
def _gauss(ws=11, sigma=1.5, device="cuda"):
    g = torch.exp(-((torch.arange(ws, device=device) - ws // 2) ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None].repeat(3, 1, 1, 1)


def ssim(x, y, w):
    """x,y [1,3,H,W]"""
    mu_x, mu_y = F.conv2d(x, w, padding=5, groups=3), F.conv2d(y, w, padding=5, groups=3)
    sxx = F.conv2d(x * x, w, padding=5, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=5, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=5, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))).mean()


# ------------------------------------------------------------------ georef
def umeyama(src, dst):
    """similarity s,R,t minimising |s R src + t - dst|"""
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum()
    return s, R, md - s * R @ ms


def gps_targets(views, srcs: dict):
    """name -> ENU (m) for frames that have GPS. Returns (names, enu, origin_latlon)."""
    rows = {}
    for src, frames_dir in srcs.items():
        fj = Path(frames_dir).parent / "frames.jsonl"
        if not fj.exists():
            continue
        for line in fj.open():
            d = json.loads(line)
            if "latitude" in d and "rel_alt" in d:
                rows[f"{src}/{Path(d['file']).name}"] = d
    names = [v["name"] for v in views if v["name"] in rows]
    if not names:
        return [], None, None
    lat0, lon0 = rows[names[0]]["latitude"], rows[names[0]]["longitude"]
    enu = np.array([[(rows[n]["longitude"] - lon0) * 111320 * math.cos(math.radians(lat0)),
                     (rows[n]["latitude"] - lat0) * 110574, rows[n]["rel_alt"]] for n in names])
    return names, enu, (lat0, lon0)


def robust_sim(src, dst, iters=5, k=2.5):
    keep = np.ones(len(src), bool)
    for _ in range(iters):
        s, R, t = umeyama(src[keep], dst[keep])
        err = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        thr = max(k * np.median(err[keep]), 1.0)
        keep = err < thr
    return s, R, t, err, keep


def level_ground(T, means, opac, cam_enu, iters=400, tol=0.15):
    """RANSAC a plane through the ground under the cameras; rotate it to z=0 with +z up."""
    p = (T[:3, :3] @ means.T).T + T[:3, 3]
    lo, hi = cam_enu[:, :2].min(0) - 15, cam_enu[:, :2].max(0) + 15
    sel = (opac > 0.5) & (p[:, 0] > lo[0]) & (p[:, 0] < hi[0]) & (p[:, 1] > lo[1]) & (p[:, 1] < hi[1])
    sel &= p[:, 2] < np.percentile(p[sel, 2], 40)  # ground is the lower layer
    q = p[sel]
    if len(q) < 1000:
        print("ground: too few points, not levelled")
        return T
    rng = np.random.default_rng(0)
    best, bn, bd = 0, None, None
    for _ in range(iters):
        a, b, c = q[rng.choice(len(q), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-6:
            continue
        n /= np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        if n[2] < 0.8:  # must be roughly horizontal already
            continue
        d = -n @ a
        inl = (np.abs(q @ n + d) < tol).sum()
        if inl > best:
            best, bn, bd = inl, n, d
    if bn is None:
        print("ground: no plane found, not levelled")
        return T
    inl = q[np.abs(q @ bn + bd) < tol]
    c = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
    n = vt[2] if vt[2][2] > 0 else -vt[2]
    z = np.array([0.0, 0, 1])
    v, cth = np.cross(n, z), n @ z
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx / (1 + cth)
    L = np.eye(4)
    L[:3, :3] = R
    L[:3, 3] = -R @ c
    L[:2, 3] += c[:2]  # keep x,y origin where it was; only level and set ground to z=0
    tilt = np.degrees(np.arccos(np.clip(cth, -1, 1)))
    print(f"ground: {best}/{len(q)} inliers, tilt corrected {tilt:.1f} deg, ground was at z={c[2]:.2f} m")
    return L @ T


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recon", type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=7000)
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--crop-margin", type=float, default=30.0, help="drop gaussians this far (m) beyond the camera bbox")
    ap.add_argument("--max-side", type=int, default=0, help="resize so the longest image side is at most this")
    ap.add_argument("--holdout", type=int, default=20, help="every Nth view is held out for PSNR (0 = none)")
    ap.add_argument("--view-pad", type=float, default=0.25, help="keep views within the drone camera bbox padded by this fraction")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.manual_seed(0)

    # images/<src>/ holds symlinks into the processed frames dir; frames.jsonl sits next to it
    srcs = {d.name: next(d.iterdir()).resolve().parent for d in (a.recon / "images").iterdir() if d.is_dir()}
    model = largest_model(a.recon / "sparse")
    und = a.recon / "undistorted"
    if not (und / "sparse").exists():
        pycolmap.undistort_images(und, model, a.recon / "images")
    views, pts, cols = load_views(und, a.downscale, dev, a.max_side)
    print(f"{len(views)} views at {views[0]['W']}x{views[0]['H']}, {len(pts)} sfm points", flush=True)
    # drop mis-registered views: SfM sometimes places a few frames far from where they could be
    # (e.g. phone frames 1 km away). Keep views inside the drone's camera volume (+pad), else a
    # robust median/MAD box over all views.
    C = np.array([v["center"] for v in views])
    ref = np.array([v["center"] for v in views if v["name"].startswith("drone/")])
    if len(ref) >= 10:
        lo, hi = ref.min(0), ref.max(0)
        pad = (hi - lo) * a.view_pad
        lo, hi = lo - pad, hi + pad
    else:
        med = np.median(C, 0)
        mad = np.median(np.abs(C - med), 0) * 1.4826 + 1e-6
        lo, hi = med - 4 * mad, med + 4 * mad
    inb = ((C > lo) & (C < hi)).all(1)
    if (~inb).any():
        dropped = sorted({v["name"].split("/")[0] for v, k in zip(views, inb) if not k})
        print(f"dropping {int((~inb).sum())}/{len(views)} out-of-volume views (sources: {dropped})", flush=True)
        views = [v for v, k in zip(views, inb) if k]

    centers = np.array([v["center"] for v in views])
    scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max()) * 1.1

    # init
    p = torch.tensor(pts, dtype=torch.float32, device=dev)
    d = torch.cdist(p[None], p[None])[0] if len(p) < 30000 else None
    if d is not None:
        d.fill_diagonal_(1e9)
        nn3 = d.topk(4, largest=False).values[:, 1:].mean(1)
    else:
        from scipy.spatial import cKDTree
        nn3 = torch.tensor(cKDTree(pts).query(pts, k=4)[0][:, 1:].mean(1), device=dev, dtype=torch.float32)
    N = len(p)
    K = (a.sh_degree + 1) ** 2
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(p),
        "scales": torch.nn.Parameter(torch.log(nn3.clamp_min(1e-4))[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0], device=dev).repeat(N, 1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.1, device=dev))),
        "sh0": torch.nn.Parameter(((torch.tensor(cols, device=dev, dtype=torch.float32) - 0.5) / SH_C0)[:, None]),
        "shN": torch.nn.Parameter(torch.zeros(N, K - 1, 3, device=dev)),
    })
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15) for k in params}
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1 / a.steps))
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy
    strat = DefaultStrategy(refine_stop_iter=int(a.steps * 0.6), reset_every=3000, verbose=False)
    strat.check_sanity(params, opts)
    state = strat.initialize_state(scene_scale=scene_scale)
    w = _gauss(device=dev)
    test = views[::a.holdout] if a.holdout else []
    train_v = [v for i, v in enumerate(views) if not (a.holdout and i % a.holdout == 0)]
    t0 = time.time()
    for step in range(a.steps):
        v = train_v[np.random.randint(len(train_v))]
        deg = min(step // 1000, a.sh_degree)
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        img, alpha, info = rasterization(params["means"], F.normalize(params["quats"], dim=-1), params["scales"].exp(),
                                         params["opacities"].sigmoid(), colors, v["viewmat"][None], v["K"][None],
                                         v["W"], v["H"], sh_degree=deg, packed=False)
        strat.step_pre_backward(params, opts, state, step, info)
        pred = img[0].permute(2, 0, 1)[None]
        gt = v["rgb"].to(dev, non_blocking=True).float().div(255).permute(2, 0, 1)[None]
        loss = 0.8 * (pred - gt).abs().mean() + 0.2 * (1 - ssim(pred, gt, w))
        loss.backward()
        strat.step_post_backward(params, opts, state, step, info, packed=False)
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        sched.step()
        if step % 500 == 0 or step == a.steps - 1:
            print(f"step {step} loss {loss.item():.4f} gaussians {len(params['means'])} {time.time() - t0:.0f}s", flush=True)

    if test:
        with torch.no_grad():
            ps = []
            for v in test:
                img, _, _ = rasterization(params["means"], F.normalize(params["quats"], dim=-1), params["scales"].exp(),
                                          params["opacities"].sigmoid(), torch.cat([params["sh0"], params["shN"]], 1),
                                          v["viewmat"][None], v["K"][None], v["W"], v["H"], sh_degree=a.sh_degree)
                gt = v["rgb"].to(dev).float().div(255)
                ps.append(-10 * torch.log10(((img[0].clamp(0, 1) - gt) ** 2).mean()).item())
            print(f"held-out PSNR over {len(test)} views: {np.mean(ps):.2f} dB (min {np.min(ps):.2f})", flush=True)

    # georeference
    names, enu, origin = gps_targets(views, srcs)
    by = {v["name"]: v["center"] for v in views}
    T = np.eye(4)
    if len(names) >= 5:
        src = np.array([by[n] for n in names])
        s, R, t, err, keep = robust_sim(src, enu)
        T[:3, :3], T[:3, 3] = s * R, t
        print(f"georef: {keep.sum()}/{len(names)} GPS frames used, scale {s:.3f}, "
              f"residual median {np.median(err[keep]):.2f} m (GPS is ~±11 m at 4 decimals), origin {origin}")
    else:
        print("georef: not enough GPS frames; leaving SfM frame (no metric scale!)")
    T = level_ground(T, params["means"].detach().cpu().numpy(), params["opacities"].detach().sigmoid().cpu().numpy(),
                     (T[:3, :3] @ centers.T).T + T[:3, 3])

    g = Gaussians(params["means"].detach(), F.normalize(params["quats"].detach(), dim=-1), params["scales"].detach().exp(),
                  params["opacities"].detach().sigmoid(), torch.cat([params["sh0"], params["shN"]], 1).detach())
    # crop far floaters/background to the camera bbox + margin (in SfM units -> use ENU margin / scale)
    sc = np.linalg.norm(T[:3, 0]) if len(names) >= 5 else 1.0
    lo = torch.tensor(centers.min(0) - a.crop_margin / sc, device=dev, dtype=torch.float32)
    hi = torch.tensor(centers.max(0) + a.crop_margin / sc, device=dev, dtype=torch.float32)
    keep = ((g.means > lo) & (g.means < hi)).all(1) & (g.opacities > 0.02)
    g = Gaussians(*(x[keep] for x in (g.means, g.quats, g.scales, g.opacities, g.sh)))
    out = REPO / "data" / "splats" / "_src"
    out.mkdir(parents=True, exist_ok=True)
    save_ply(g, out / f"{a.name}.ply")
    (out / f"{a.name}.transform.json").write_text(json.dumps(T.tolist()))
    # waypoints: 6 points spread along the flown (GPS-registered) drone path, in ENU
    enu_c = (T[:3, :3] @ centers.T).T + T[:3, 3]
    drone = [i for i, v in enumerate(views) if v["name"].startswith("drone/")] or list(range(len(views)))
    pick = [drone[int(i)] for i in np.linspace(0, len(drone) - 1, 7)[:-1]]
    wps = [[round(float(x), 2) for x in enu_c[i]] for i in pick]
    (out / f"{a.name}.waypoints.json").write_text(json.dumps(wps))
    print(f"{int(keep.sum())} gaussians -> {out / (a.name + '.ply')}; waypoints {wps}")


if __name__ == "__main__":
    main()
