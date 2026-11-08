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
    cfg = EnvConfig(num_envs=n or args.envs, render_chunk=args.chunk, task=getattr(args, "task", "loop"),
                    ceiling=getattr(args, "ceiling", 30.0), radius_clip=getattr(args, "radius_clip", 0.0),
                    full_quality_frac=getattr(args, "full_frac", 1.0))
    if getattr(args, "route_lead", None):
        cfg.route_lead_m = tuple(float(x) for x in args.route_lead.split(","))
    if getattr(args, "route_speed0", None) is not None:
        cfg.route_speed0 = args.route_speed0
    if getattr(args, "route_takeover_frac", None) is not None:
        cfg.route_takeover_frac = args.route_takeover_frac
    if getattr(args, "route_obstacle_frac", None) is not None:
        cfg.route_obstacle_frac = args.route_obstacle_frac
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
    # bound the grid to where the scene actually is: trim outlier gaussians (floaters, far
    # background) and cap the height, otherwise the distance transform can need 10s of GB
    m = g.means[g.opacities > 0.3]
    lo = torch.quantile(m[::max(1, len(m) // 500_000)], 0.01, dim=0) - 5
    hi = torch.quantile(m[::max(1, len(m) // 500_000)], 0.99, dim=0) + 5
    lo[2], hi[2] = max(lo[2].item(), -3.0), min(hi[2].item(), a.max_height)
    esdf = bake_esdf(g, voxel=a.voxel, bounds=(lo.tolist(), hi.tolist()), min_opacity=a.min_opacity,
                     min_neighbors=a.min_neighbors, max_scale=a.max_scale, min_component=a.min_component)
    print(f"esdf bounds {[round(x, 1) for x in lo.tolist()]} .. {[round(x, 1) for x in hi.tolist()]}")
    wps = torch.tensor(json.loads(Path(a.waypoints).read_text())) if a.waypoints else torch.zeros(1, 3)
    # save the (already transformed) gaussians with an identity transform
    d = save_scene(Scene(a.name, g, esdf, wps.float()), SPLATS, notes=f"imported from {a.ply}")
    geo = Path(a.ply).with_suffix(".geo.json")
    if geo.exists():  # gitignored sidecar with GPS origin + fit quality
        (d / "geo.json").write_text(geo.read_text())
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


def cmd_hazard_collect(a):
    from .hazard import BrakeModel, collect
    env = _env(a)
    out = Path(a.out) if a.out else REPO / "data" / "hazard" / f"{env.s.name}.pt"
    collect(env, a.frames, out, BrakeModel(latency=a.latency, a_brake=a.a_brake, margin=a.margin), seed=a.seed)


def cmd_hazard_train(a):
    from .hazard import train_hazard
    run = RUNS / (a.name or f"hazard-{time.strftime('%Y%m%d-%H%M%S')}")
    train_hazard(Path(a.data), run, device=a.device, epochs=a.epochs)
    print(f"done -> {run}/hazard.pt")


def cmd_serve(a):
    from .viewer import serve
    serve(_scene(a.scene), host=a.host, port=a.port, device=a.device)


def cmd_pilot_serve(a):
    from .pilot import PilotCore, serve
    core = PilotCore(a.ckpt, a.hazard_ckpt, a.device, thr=a.thr, lookahead=a.lookahead, speed_cap=a.speed_cap, log_dir=a.log_dir)
    serve(core, a.host, a.port)


def main():
    ap = argparse.ArgumentParser(prog="djirl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="cuda:0")
    common.add_argument("--chunk", type=int, default=128, help="cameras per gsplat call")

    p = sub.add_parser("synth", parents=[common]); p.add_argument("--seed", type=int, default=0); p.set_defaults(f=cmd_synth)
    p = sub.add_parser("import-ply", parents=[common]); p.add_argument("ply"); p.add_argument("--name", required=True)
    p.add_argument("--voxel", type=float, default=0.1); p.add_argument("--waypoints", help="JSON [[x,y,z],...] ENU")
    p.add_argument("--transform", help="JSON 4x4 splat->ENU similarity")
    p.add_argument("--max-height", type=float, default=40.0, help="collision grid ceiling (m)")
    p.add_argument("--min-opacity", type=float, default=0.3, help="gaussians below this don't count as obstacles")
    p.add_argument("--min-neighbors", type=int, default=2, help="drop isolated occupied voxels (floaters)")
    p.add_argument("--max-scale", type=float, default=None, help="gaussians larger than this (m) don't count as obstacles")
    p.add_argument("--min-component", type=int, default=0, help="drop occupied 3D blobs smaller than this many voxels (floaters)")
    p.set_defaults(f=cmd_import_ply)
    for name, f in (("bench", cmd_bench), ("train", cmd_train), ("rollout", cmd_rollout), ("preview", cmd_preview), ("serve", cmd_serve)):
        p = sub.add_parser(name, parents=[common])
        p.add_argument("--scene", default="synthetic")
        p.set_defaults(f=f)
        if name in ("bench", "train", "rollout"):
            p.add_argument("--task", choices=["loop", "a2b", "route"], default="loop")
            p.add_argument("--ceiling", type=float, default=30.0)
            p.add_argument("--radius-clip", type=float, default=0.0, help="skip gaussians <= N px (2.0 = fast)")
            p.add_argument("--full-frac", type=float, default=1.0, help="fraction of chunks rendered at full quality")
            p.add_argument("--envs", type=int, default={"bench": 512, "train": 512, "rollout": 1}[name])
        if name == "bench":
            p.add_argument("--steps", type=int, default=20)
        if name == "train":
            p.add_argument("--minutes", type=float, default=90); p.add_argument("--name")
            p.add_argument("--horizon", type=int, default=32); p.add_argument("--resume")
            p.add_argument("--route-lead", help="lo,hi m before the obstacle at spawn (default 8,18)")
            p.add_argument("--route-speed0", type=float, help="spawn speed ~ U(0, this) (default 4)")
            p.add_argument("--route-obstacle-frac", type=float)
            p.add_argument("--route-takeover-frac", type=float,
                           help="obstacle episodes spawned like a handoff: cruise 2-8 m/s, stopping distance + 0-10 m out")
        if name == "rollout":
            p.add_argument("--ckpt"); p.add_argument("--out", default="rollout.mp4"); p.add_argument("--seconds", type=int, default=30)
        if name == "preview":
            p.add_argument("--width", type=int, default=640); p.add_argument("--height", type=int, default=480)
        if name == "serve":
            p.add_argument("--host", default="0.0.0.0"); p.add_argument("--port", type=int, default=8965)
    p = sub.add_parser("routes", parents=[common], help="build the route library for the route task")
    p.add_argument("--scene", default="house-worldlabs-refined"); p.add_argument("--n", type=int, default=400)
    p.set_defaults(f=lambda a: __import__("djirl.routes", fromlist=["build"]).build(a.scene, a.n, device=a.device))
    p = sub.add_parser("hazard-collect", parents=[common], help="scripted flights -> hazard training data")
    p.add_argument("--scene", default="synthetic"); p.add_argument("--envs", type=int, default=1024)
    p.add_argument("--frames", type=int, default=1_000_000); p.add_argument("--out"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--latency", type=float, default=0.30); p.add_argument("--a-brake", type=float, default=5.0)
    p.add_argument("--margin", type=float, default=1.5); p.set_defaults(f=cmd_hazard_collect)
    p.add_argument("--task", choices=["loop", "a2b", "route"], default="loop"); p.add_argument("--ceiling", type=float, default=30.0)
    p.add_argument("--radius-clip", type=float, default=0.0); p.add_argument("--full-frac", type=float, default=1.0)
    p = sub.add_parser("hazard-train", parents=[common], help="train the hazard model on collected data")
    p.add_argument("data"); p.add_argument("--name"); p.add_argument("--epochs", type=int, default=8)
    p.set_defaults(f=cmd_hazard_train)
    p = sub.add_parser("handoff", parents=[common], help="planner -> policy -> planner handoff eval on obstacle missions")
    p.add_argument("--scene", default="house-worldlabs-refined"); p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=512); p.add_argument("--envs", type=int, default=256)
    p.add_argument("--trigger", choices=["oracle", "net"], default="oracle"); p.add_argument("--hazard-ckpt")
    p.add_argument("--thr", type=float, default=0.5); p.add_argument("--cruise", type=float, default=5.0)
    p.add_argument("--pilots", default="planner,policy,handoff"); p.add_argument("--out"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lookahead", default="0", help="comma list of extra trigger metres beyond stopping distance")
    p.set_defaults(f=lambda a: __import__("djirl.handoff", fromlist=["evaluate"]).evaluate(
        _scene(a.scene), a.ckpt, a.episodes, a.envs, a.device, a.trigger, a.hazard_ckpt, a.thr, a.cruise,
        tuple(a.pilots.split(",")), a.chunk, a.out, a.seed,
        tuple(float(x) for x in a.lookahead.split(","))))
    p = sub.add_parser("pilot-serve", help="live hazard trigger + policy for the phone mission runner (docs/pilot-protocol.md)")
    p.add_argument("--ckpt", default="runs/route-wlref-2/policy.pt"); p.add_argument("--hazard-ckpt", default="runs/hazard-wlref/hazard.pt")
    p.add_argument("--device", default="cuda:1"); p.add_argument("--host", default="100.80.98.4"); p.add_argument("--port", type=int, default=8971)
    p.add_argument("--thr", type=float, default=0.5); p.add_argument("--lookahead", type=float, default=4.0)
    p.add_argument("--speed-cap", type=float, default=4.0, help="m/s cap on the policy's horizontal command")
    p.add_argument("--log-dir", default="runs/pilot")
    p.set_defaults(f=cmd_pilot_serve)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
