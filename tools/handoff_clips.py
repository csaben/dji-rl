"""Clips of the hazard classifier driving the planner -> policy handoff on obstacle missions.

    .venv/bin/python tools/handoff_clips.py --ckpt runs/route-wlref-2/policy.pt \
        --hazard runs/hazard-wlref/hazard.pt --out runs/clips --n 8 --device cuda:1

Each clip: left = drone camera (HD) with HUD (pilot in control, speed, classifier p(brake), predicted vs
true clearance along the velocity, stopping distance); right = top-down map with the route (white), the
unmapped obstacle (red), and the flown trail coloured by who was flying (cyan planner, orange policy),
plus a strip chart of p(brake) / clearance over time.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from djirl.env import BatchEnv, EnvConfig  # noqa: E402
from djirl.handoff import POLICY, PLANNER, Trigger, map_los, planner_act  # noqa: E402
from djirl.hazard import BrakeModel, hazard_labels  # noqa: E402
from djirl.ppo import ActorCritic  # noqa: E402
from djirl.render import Renderer  # noqa: E402
from djirl.scene import load_scene  # noqa: E402
from djirl.sim_api import SimAPI  # noqa: E402

FW, FH, MW = 640, 480, 480
CYAN, ORANGE, RED, WHITE = (60, 220, 255), (255, 150, 40), (255, 60, 60), (240, 240, 240)


def font(sz):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="house-worldlabs-refined")
    ap.add_argument("--ckpt", required=True); ap.add_argument("--hazard", required=True)
    ap.add_argument("--out", default="runs/clips"); ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--lookahead", type=float, default=4.0); ap.add_argument("--cruise", type=float, default=5.0)
    ap.add_argument("--device", default="cuda:1"); ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = a.device
    torch.manual_seed(a.seed)
    scene = load_scene(REPO / "data" / "splats" / a.scene)
    cfg = EnvConfig(num_envs=a.n, task="route", render_chunk=64, radius_clip=2.0, full_quality_frac=0.25, ceiling=15.0, route_obstacle_frac=1.0,
                    route_lead_m=(25.0, 35.0), route_tail_m=(20.0, 25.0), route_speed0=0.0, episode_s=40.0)
    env = BatchEnv(scene, cfg, device=dev)
    env.resample_obstacles(seed=a.seed)
    obs = env.reset()
    model = ActorCritic(cfg.frame_stack, env.VEC_DIM, env.PRIV_DIM).to(dev)
    model.value({k: v[:1] for k, v in obs.items()}), model.dist({k: v[:1] for k, v in obs.items()})
    model.load_state_dict(torch.load(a.ckpt, map_location=dev)["model"]); model.eval()
    brake = BrakeModel()
    trig = Trigger(env, brake, "net", a.hazard, 0.5, a.cruise, a.lookahead)
    hd = Renderer(env.renderer.g, FW, FH, 82.0, chunk=a.n)
    api = SimAPI(dev)
    meta = api.topdown_meta(a.scene, 2048)
    top = Image.open(io.BytesIO(api.topdown_png(a.scene, 2048))).convert("RGB")
    ppm = meta["px_w"] / (meta["x1"] - meta["x0"])
    to_px = lambda x, y: ((x - meta["x0"]) * ppm, (meta["y1"] - y) * ppm)

    # per-env episode records (first episode only)
    n, hz = a.n, cfg.control_hz
    rec = [dict(frames=[], trail=[], mode=[], p=[], cp=[], ct=[], stop=[], done=False) for _ in range(n)]
    seg = [(int(env.r_id[i]), int(env.r_idx[i]), int(env.r_end[i])) for i in range(n)]
    ob = env.obst.p.clone()
    mode = torch.zeros(n, dtype=torch.long, device=dev)
    t_mode = torch.zeros(n, device=dev); quiet = torch.zeros(n, device=dev)
    f_big, f_sm = font(22), font(16)
    for step in range(int(cfg.episode_s * hz)):
        hazard, clear_c = trig(obs)
        clear, logit = trig.net(obs["latent"], obs["vec"])
        p = torch.sigmoid(logit[:, -1])
        lab = hazard_labels(env, brake)
        quiet = torch.where(hazard, torch.zeros_like(quiet), quiet + 1 / hz)
        take = (mode == PLANNER) & hazard
        back = (mode == POLICY) & (t_mode >= 1.0) & (quiet >= 1.0) & clear_c & map_los(env, env.dyn.pos, env.goal, cfg.drone_radius + 0.5)
        mode = torch.where(take, POLICY, torch.where(back, PLANNER, mode))
        t_mode = torch.where(take | back, torch.zeros_like(t_mode), t_mode) + 1 / hz
        act = torch.where((mode == POLICY)[:, None], model.act(obs, deterministic=True), planner_act(env, a.cruise))
        rgb = hd.render(env.dyn.pos, env.dyn.yaw, env.pitch)[..., :3]
        speed = env.dyn.vel.norm(dim=-1)
        for i in range(n):
            r = rec[i]
            if r["done"]:
                continue
            r["frames"].append((rgb[i].cpu().numpy() * 255).astype(np.uint8))
            r["trail"].append(env.dyn.pos[i, :2].tolist()); r["mode"].append(int(mode[i]))
            r["p"].append(float(p[i])); r["cp"].append(float(clear[i, 0])); r["ct"].append(float(lab["clear_vel"][i]))
            r["stop"].append(float(brake.stopping_distance(speed[i]) + a.lookahead)); r.setdefault("speed", []).append(float(speed[i]))
            r.setdefault("trig", []).append(bool(take[i]))
        obs, rew, done, info = env.step(act)
        if done.any():
            dl = done.nonzero()[:, 0].tolist()
            for j, i in enumerate(dl):
                if not rec[i]["done"]:
                    rec[i]["done"] = True
                    rec[i]["success"] = bool(info["final_success"][j] > 0)
                    rec[i]["collided"] = bool(info["final_collided"][j] > 0)
        if all(r["done"] for r in rec):
            break

    summary = []
    for i, r in enumerate(rec):
        rid, s0, e = seg[i]
        route = env.rpts[rid, s0:e + 1, :2].cpu().numpy()
        tr = np.array(r["trail"])
        k = int(((ob[:, 1:3] - env.rpts[rid, (s0 + e) // 2, :2]).norm(dim=-1)).argmin())  # obstacle on this segment
        pts = np.concatenate([route, tr])
        lo, hi = pts.min(0) - 12, pts.max(0) + 12
        side = max(hi - lo); c = (lo + hi) / 2
        x0, y1 = c[0] - side / 2, c[1] + side / 2
        u0, v0 = to_px(x0, y1)
        crop = top.crop((int(u0), int(v0), int(u0 + side * ppm), int(v0 + side * ppm))).resize((MW, MW))
        sc = MW / side
        P = lambda x, y: ((x - x0) * sc, (y1 - y) * sc)
        name = f"clip{i:02d}_{'ok' if r.get('success') else ('crash' if r.get('collided') else 'timeout')}"
        ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{FW + MW}x{FH + 120}",
                               "-r", str(int(hz)), "-i", "-", "-vf", "tpad=stop_mode=clone:stop_duration=1.5",
                               "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "23", "-movflags", "+faststart",
                               str(out / f"{name}.mp4")], stdin=subprocess.PIPE)
        T = len(r["frames"])
        for t in range(T):
            im = Image.new("RGB", (FW + MW, FH + 120), (18, 18, 22))
            im.paste(Image.fromarray(r["frames"][t]), (0, 0))
            mp = crop.copy(); d = mp_d = ImageDraw.Draw(mp)
            d.line([P(*q) for q in route], fill=WHITE, width=2)
            for kk in range(len(ob)):  # every unmapped obstacle in view (others' routes cross this one too)
                kd, cx, cy, rad = int(ob[kk, 0]), float(ob[kk, 1]), float(ob[kk, 2]), float(ob[kk, 5])
                ox, oy = P(cx, cy)
                if not (-20 < ox < MW + 20 and -20 < oy < MW + 20):
                    continue
                if kd == 1:  # wall: oriented rectangle
                    w2, d2, yw = float(ob[kk, 6]) / 2, float(ob[kk, 7]) / 2, float(ob[kk, 8])
                    c_, s_ = math.cos(yw), math.sin(yw)
                    cr = [P(cx + c_ * lx - s_ * ly, cy + s_ * lx + c_ * ly) for lx, ly in ((-w2, -d2), (w2, -d2), (w2, d2), (-w2, d2))]
                    d.polygon(cr, outline=RED, width=3)
                else:
                    rr = max(4, (float(ob[kk, 9]) if kd == 2 else rad) * sc)
                    d.ellipse([ox - rr, oy - rr, ox + rr, oy + rr], outline=RED, width=3)
            kind = int(ob[k, 0])
            for j in range(1, t + 1):
                d.line([P(*r["trail"][j - 1]), P(*r["trail"][j])], fill=ORANGE if r["mode"][j] == POLICY else CYAN, width=3)
            dx, dy = P(*r["trail"][t]); d.ellipse([dx - 5, dy - 5, dx + 5, dy + 5], fill=(255, 255, 255))
            im.paste(mp, (FW, 0))
            dr = ImageDraw.Draw(im)
            pol = r["mode"][t] == POLICY
            dr.rectangle([0, 0, 250, 34], fill=ORANGE if pol else CYAN)
            dr.text((10, 5), "POLICY (dodging)" if pol else "PLANNER (route)", fill=(0, 0, 0), font=f_big)
            pb = r["p"][t]
            dr.text((10, FH - 60), f"{r['speed'][t]:4.1f} m/s   p(brake) {pb:.2f}", fill=WHITE, font=f_big)
            dr.text((10, FH - 30), f"clearance pred {min(r['cp'][t], 20):4.1f} m  true {r['ct'][t]:4.1f} m  trigger < {r['stop'][t]:.1f} m",
                    fill=WHITE, font=f_sm)
            bw = int(300 * pb); dr.rectangle([330, 8, 630, 28], outline=WHITE); dr.rectangle([330, 8, 330 + bw, 28], fill=RED if pb > 0.5 else (90, 200, 90))
            dr.text((FW + 8, 6), "route (white)  obstacles (red)  trail", fill=WHITE, font=f_sm)
            # strip chart: p(brake) red, predicted clearance (cyan) vs true (white), 0-20 m
            y0s, hs, W = FH + 10, 100, FW + MW - 20
            dr.rectangle([10, y0s, 10 + W, y0s + hs], outline=(80, 80, 80))
            sx = W / max(T, 1)
            for j in range(1, t + 1):
                xa, xb = 10 + (j - 1) * sx, 10 + j * sx
                if r["mode"][j] == POLICY:
                    dr.rectangle([xa, y0s + 1, xb, y0s + hs - 1], fill=(70, 45, 15))
                for key, col, f in (("ct", WHITE, lambda v: min(v, 20) / 20), ("cp", CYAN, lambda v: min(v, 20) / 20), ("p", RED, lambda v: v)):
                    dr.line([(xa, y0s + hs - f(r[key][j - 1]) * hs), (xb, y0s + hs - f(r[key][j]) * hs)], fill=col, width=2)
            dr.text((14, y0s + 2), "p(brake)", fill=RED, font=f_sm); dr.text((110, y0s + 2), "clear pred", fill=CYAN, font=f_sm)
            dr.text((220, y0s + 2), "clear true (0-20 m)", fill=WHITE, font=f_sm)
            if t == T - 1:
                msg = "REACHED END" if r.get("success") else ("COLLISION" if r.get("collided") else "TIMEOUT")
                dr.text((FW // 2 - 90, FH // 2 - 20), msg, fill=(90, 255, 90) if r.get("success") else RED, font=font(36))
            ff.stdin.write(np.asarray(im).tobytes())
        ff.stdin.close(); ff.wait()
        summary.append({"clip": name, "seconds": round(T / hz, 1), "takeovers": int(sum(r["trig"])),
                        "policy_frac": round(float(np.mean(r["mode"])), 2), "obstacle": ["pole", "wall", "tree"][kind]})
        print(json.dumps(summary[-1]), flush=True)
    (out / "clips.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
