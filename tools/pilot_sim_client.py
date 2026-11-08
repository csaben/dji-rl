"""Play the phone against a running pilot server (docs/pilot-protocol.md): fly sim obstacle missions
(route task, unmapped obstacle 25-35 m in), send rendered frames + synthetic telemetry over the real
WebSocket protocol, follow the server's want/sticks/rejoin, and report success and timing.

    .venv/bin/djirl pilot-serve &                      # GPU 1
    .venv/bin/python tools/pilot_sim_client.py --episodes 20 --mode policy

mode shadow: never applies the sticks (the planner flies into the obstacle; measures trigger timing).
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import struct
import time

import numpy as np
import torch
from PIL import Image


def mission_from_env(env, scale=1.0, trim_deg=0.0):
    r, s0, e = int(env.r_id[0]), int(env.r_idx[0]), int(env.r_end[0])
    pts = env.rpts[r, s0:e + 1].cpu().numpy()
    keep = list(range(0, len(pts), 8)) + ([len(pts) - 1] if (len(pts) - 1) % 8 else [])
    pts = pts[keep]
    A = pts[0, :2]
    t = math.radians(trim_deg)
    rel = []
    for p in pts:
        d = (p[:2] - A) * scale
        rel.append({"e_m": float(math.cos(t) * d[0] - math.sin(t) * d[1]), "n_m": float(math.sin(t) * d[0] + math.cos(t) * d[1]),
                    "up_m": float(p[2] * scale)})
    return {"version": 1, "name": "pilot_sim", "obstacle_handoff": "policy", "limits": {"cruise_speed_mps": 5},
            "sim_preview": {"scene": env.s.name}, "sim_request": {"scene": env.s.name, "a": {"x": float(A[0]), "y": float(A[1])}},
            "route": [{"lat": 0, "lon": 0, "alt_m": float(p[2])} for p in pts], "route_rel": rel,
            "alignment": {"scale_corr": scale, "yaw_trim_deg": trim_deg}}, A, t


def telemetry(env, A, t, scale, seq, t_ms, flying_by, sticks):
    p, v, yaw = env.dyn.pos[0].cpu().numpy(), env.dyn.vel[0].cpu().numpy(), float(env.dyn.yaw[0])
    c, s = math.cos(t), math.sin(t)
    d = (p[:2] - A) * scale
    e, n = c * d[0] - s * d[1], s * d[0] + c * d[1]
    ve, vn = c * v[0] - s * v[1], s * v[0] + c * v[1]
    return {"seq": seq, "t_ms": t_ms, "pos_rel": {"e": float(e), "n": float(n), "up": float(p[2] * scale)},
            "vel_ned": [float(vn), float(ve), float(-v[2])], "yaw_deg": float((90 - math.degrees(yaw + t)) % 360),
            "gimbal_pitch_deg": -10, "flying_by": flying_by, "sticks_now": [float(x) for x in sticks]}


async def run(a):
    import websockets
    from djirl.cli import _scene
    from djirl.env import BatchEnv, EnvConfig
    from djirl.handoff import planner_act
    cfg = EnvConfig(num_envs=1, task="route", render_chunk=1, radius_clip=2.0, full_quality_frac=1.0, ceiling=15.0,
                    route_obstacle_frac=1.0, route_lead_m=(25.0, 35.0), route_tail_m=(25.0, 35.0), route_speed0=0.0,
                    episode_s=40.0, width=a.width, height=a.height)
    env = BatchEnv(_scene(a.scene), cfg, device=a.device)
    torch.manual_seed(a.seed)
    env.resample_obstacles(seed=a.seed)
    env.reset()
    res = []
    async with websockets.connect(a.url, max_size=8 * 1024 * 1024, compression=None) as ws:
        for ep in range(a.episodes):
            mission, A, t = mission_from_env(env, a.scale, a.trim)
            await ws.send(json.dumps({"type": "hello", "mission": mission, "mode": a.mode}))
            ready = json.loads(await ws.recv())
            assert ready["type"] == "ready", ready
            flying, seq, rtt, srv, takeovers, applied = "mission", 0, [], [], 0, np.zeros(4)
            first_hazard_clear = None
            while True:
                rgb = env._obs()["rgb"][0, ..., :3].clamp(0, 1).mul(255).byte().cpu().numpy()
                buf = io.BytesIO()
                Image.fromarray(rgb).save(buf, "JPEG", quality=80)
                h = telemetry(env, A, t, a.scale, seq, int(time.time() * 1000), flying, applied)
                hb = json.dumps(h).encode()
                t0 = time.time()
                await ws.send(struct.pack(">I", len(hb)) + hb + buf.getvalue())
                cmd = json.loads(await ws.recv())
                rtt.append((time.time() - t0) * 1000)
                srv.append(cmd.get("server_ms", 0))
                want = cmd.get("want", "hover")
                if want == "policy" and flying != "policy":
                    takeovers += 1
                    if first_hazard_clear is None:
                        from djirl.hazard import ray_clearance
                        class Q: query = staticmethod(env._sdf)
                        sp = env.dyn.vel.norm(dim=-1)
                        vd = env.dyn.vel / sp.clamp_min(1e-6)[:, None]
                        first_hazard_clear = float(ray_clearance(Q, env.dyn.pos, vd, 0.25)[0])
                if a.mode == "shadow":
                    flying = "mission"
                else:
                    flying = want
                if flying == "policy":
                    act = torch.tensor([cmd["sticks"]["norm"]], device=env.dev)
                    f, r = cmd["sticks"]["fwd_mps"] / 8, cmd["sticks"]["right_mps"] / 8   # speed-capped
                    act[0, 0], act[0, 1] = f, r
                elif flying == "hover":
                    act = torch.zeros(1, 4, device=env.dev)
                else:
                    act = planner_act(env, a.cruise)
                applied = act[0].cpu().numpy()
                _, _, done, info = env.step(act)
                seq += 1
                if bool(done[0]):
                    row = {"ep": ep, "success": float(info["final_success"][0]), "collided": float(info["final_collided"][0]),
                           "time_s": float(info["final_length"][0]) / 10, "takeovers": takeovers,
                           "true_clear_at_takeover_m": first_hazard_clear,
                           "rtt_ms_p50": round(float(np.median(rtt)), 1), "rtt_ms_p95": round(float(np.percentile(rtt, 95)), 1),
                           "server_ms_p50": round(float(np.median(srv)), 1)}
                    print(json.dumps(row), flush=True)
                    res.append(row)
                    break
    k = len(res)
    print(json.dumps({"episodes": k, "mode": a.mode, "success": sum(r["success"] for r in res) / k,
                      "collided": sum(r["collided"] for r in res) / k,
                      "took_over": sum(r["takeovers"] > 0 for r in res) / k,
                      "rtt_ms_p50": float(np.median([r["rtt_ms_p50"] for r in res])),
                      "rtt_ms_p95": float(np.median([r["rtt_ms_p95"] for r in res]))}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="ws://100.80.98.4:8971/")
    ap.add_argument("--scene", default="house-worldlabs-refined")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--mode", choices=["policy", "shadow"], default="policy")
    ap.add_argument("--cruise", type=float, default=5.0)
    ap.add_argument("--scale", type=float, default=1.0, help="alignment.scale_corr to exercise the frame conversion")
    ap.add_argument("--trim", type=float, default=0.0, help="alignment.yaw_trim_deg")
    ap.add_argument("--width", type=int, default=320); ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--seed", type=int, default=7)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
