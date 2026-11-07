"""djirl command line.

    djirl synth                       build data/splats/synthetic (procedural stand-in scene)
    djirl import-ply X.ply --name N   wrap a trained splat as a scene (bakes the ESDF)
    djirl bench --scene S             env throughput (render + encode + step)
    djirl preview --scene S           write sample camera frames to runs/preview/
    djirl train --scene S --minutes 90
    djirl rollout --scene S --ckpt runs/X/policy.pt --out rollout.mp4
    djirl serve --scene S --port 8965 fly the scene from a browser (sim camera + sim dynamics)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
SPLATS = REPO / "data" / "splats"
RUNS = REPO / "runs"


def _scene(name_or_path: str):
    from .scene import load_scene
    p = Path(name_or_path)
    return load_scene(p if p.exists() else SPLATS / name_or_path)


def _env(args, n=None):
    from .env import BatchEnv, EnvConfig
    cfg = EnvConfig(num_envs=n or args.envs, render_chunk=args.chunk)
    return BatchEnv(_scene(args.scene), cfg, device=args.device)


def cmd_synth(a):
    from .scene import save_scene
    from .synthetic import make_synthetic
    s = make_synthetic(seed=a.seed, device=a.device)
    d = save_scene(s, SPLATS, notes="procedural stand-in: house, garage, 5 trees, fence posts")
    print(f"{len(s.gaussians.means)} gaussians, esdf {tuple(s.esdf.sdf.shape)} @ {s.esdf.voxel} m -> {d}")


def cmd_import_ply(a):
    import numpy as np

    from .scene import Scene, bake_esdf, load_ply, save_scene
    g = load_ply(Path(a.ply))
    T = np.eye(4)
    if a.transform:
        T = np.asarray(json.loads(Path(a.transform).read_text()))
        g = g.transformed(T)
    g = g.to(a.device)
    esdf = bake_esdf(g, voxel=a.voxel)
    wps = torch.tensor(json.loads(Path(a.waypoints).read_text())) if a.waypoints else torch.zeros(1, 3)
    # save the (already transformed) gaussians with an identity transform
    d = save_scene(Scene(a.name, g, esdf, wps.float()), SPLATS, notes=f"imported from {a.ply}")
    print(f"{len(g.means)} gaussians, esdf {tuple(esdf.sdf.shape)} -> {d}")


def cmd_bench(a):
    env = _env(a)
    obs = env.reset()
    act = torch.zeros(env.n, 4, device=env.dev)
    act[:, 0] = 0.5
    for _ in range(3):
        env.step(act)
    torch.cuda.synchronize()
    t = time.time()
    k = a.steps
    for _ in range(k):
        env.step(act + 0.2 * torch.randn_like(act))
    torch.cuda.synchronize()
    sps = k * env.n / (time.time() - t)
    print(json.dumps({"envs": env.n, "gaussians": len(env.s.gaussians.means), "env_steps_per_s": round(sps),
                      "max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}))


def cmd_preview(a):
    from PIL import Image

    from .render import Renderer
    s = _scene(a.scene).to(a.device)
    out = RUNS / "preview" / s.name
    out.mkdir(parents=True, exist_ok=True)
    r = Renderer(s.gaussians, a.width, a.height)
    w = s.waypoints
    nxt = torch.roll(w, -1, 0)
    yaw = torch.atan2(nxt[:, 1] - w[:, 1], nxt[:, 0] - w[:, 0])
    imgs = r.render(w, yaw, torch.full_like(yaw, -0.17))
    for i, im in enumerate(imgs):
        Image.fromarray((im[..., :3].cpu().numpy() * 255).astype("uint8")).save(out / f"wp{i}.png")
    print(f"{len(imgs)} frames -> {out}")


def cmd_train(a):
    from .ppo import PPOConfig, train
    env = _env(a)
    run = RUNS / (a.name or time.strftime("%Y%m%d-%H%M%S"))
    cfg = PPOConfig(minutes=a.minutes, steps_per_iter=a.horizon)
    train(env, run, cfg, meta={"scene": a.scene, "envs": a.envs}, resume=Path(a.resume) if a.resume else None)
    print(f"done -> {run}")


def cmd_rollout(a):
    from .ppo import ActorCritic
    from .render import Renderer
    env = _env(a, n=a.envs)
    obs = env.reset()
    model = ActorCritic(env.cfg.frame_stack, env.VEC_DIM, env.PRIV_DIM).to(env.dev)
    model.value({k: v[:1] for k, v in obs.items()}), model.dist({k: v[:1] for k, v in obs.items()})
    if a.ckpt:
        model.load_state_dict(torch.load(a.ckpt, map_location=env.dev)["model"])
    hi = Renderer(env.s.gaussians, 512, 384)
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", "1024x384", "-r", str(int(env.cfg.control_hz)), "-i", "-", "-pix_fmt", "yuv420p", a.out],
                          stdin=subprocess.PIPE)
    for _ in range(a.seconds * int(env.cfg.control_hz)):
        act = model.act(obs, deterministic=True)
        obs, r, done, info = env.step(act)
        big = hi.render(env.dyn.pos[:1], env.dyn.yaw[:1], env.pitch[:1])[0]
        small = torch.nn.functional.interpolate(obs["rgb"][:1].permute(0, 3, 1, 2), size=(384, 512), mode="nearest")[0].permute(1, 2, 0)
        frame = torch.cat([big[..., :3], small], 1)
        ff.stdin.write((frame.cpu().numpy() * 255).astype("uint8").tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"-> {a.out} (left: 512x384 view, right: what the policy sees at {env.cfg.width}x{env.cfg.height})")


def cmd_serve(a):
    from .viewer import serve
    serve(_scene(a.scene), host=a.host, port=a.port, device=a.device)


def main():
    ap = argparse.ArgumentParser(prog="djirl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="cuda:0")
    common.add_argument("--chunk", type=int, default=128, help="cameras per gsplat call")

    p = sub.add_parser("synth", parents=[common]); p.add_argument("--seed", type=int, default=0); p.set_defaults(f=cmd_synth)
    p = sub.add_parser("import-ply", parents=[common]); p.add_argument("ply"); p.add_argument("--name", required=True)
    p.add_argument("--voxel", type=float, default=0.1); p.add_argument("--waypoints", help="JSON [[x,y,z],...] ENU")
    p.add_argument("--transform", help="JSON 4x4 splat->ENU similarity"); p.set_defaults(f=cmd_import_ply)
    for name, f in (("bench", cmd_bench), ("train", cmd_train), ("rollout", cmd_rollout), ("preview", cmd_preview), ("serve", cmd_serve)):
        p = sub.add_parser(name, parents=[common])
        p.add_argument("--scene", default="synthetic")
        p.set_defaults(f=f)
        if name in ("bench", "train", "rollout"):
            p.add_argument("--envs", type=int, default={"bench": 512, "train": 512, "rollout": 1}[name])
        if name == "bench":
            p.add_argument("--steps", type=int, default=20)
        if name == "train":
            p.add_argument("--minutes", type=float, default=90); p.add_argument("--name")
            p.add_argument("--horizon", type=int, default=32); p.add_argument("--resume")
        if name == "rollout":
            p.add_argument("--ckpt"); p.add_argument("--out", default="rollout.mp4"); p.add_argument("--seconds", type=int, default=30)
        if name == "preview":
            p.add_argument("--width", type=int, default=640); p.add_argument("--height", type=int, default=480)
        if name == "serve":
            p.add_argument("--host", default="0.0.0.0"); p.add_argument("--port", type=int, default=8965)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
