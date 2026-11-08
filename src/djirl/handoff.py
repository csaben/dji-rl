"""Deployment handoff, evaluated in the sim: the map planner flies the route, a hazard trigger hands
control to the learned local policy when something the map doesn't know about is in the way, and the
planner takes back over (rejoining the route at the carrot ahead) once the way back is clear.

    .venv/bin/djirl handoff --scene house-worldlabs-refined --ckpt runs/route-wlref-1/policy.pt

Compares three pilots on the same obstacle missions (route task, every episode has an unmapped
obstacle 25-35 m in, 25-35 m of route after it):
  planner   pure pursuit of the route carrot, no avoidance (what the A* follower does today)
  policy    the learned policy flies the whole mission
  handoff   planner by default; policy while the trigger is active, planner again when clear

Trigger ("oracle"): stopping distance along the current velocity, from the true geometry incl. the
injected obstacle (= the hazard model's label; a perfect hazard model). "net": a trained HazardNet
(runs/X/hazard.pt) on the policy's own latents: brake if p(must_brake) > thr or predicted clearance
along velocity < stopping distance.
Handback needs: >= `min_policy_s` under the policy, the trigger quiet for `quiet_s`, the straight
line to the route carrot clear on the *map* (the planner's knowledge), and the way to the carrot clear
of hazards (oracle: ray to the carrot; net: predicted forward clearance with the camera facing it).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from .hazard import BrakeModel, HazardNet, ray_clearance

PLANNER, POLICY = 0, 1


def planner_act(env, cruise: float, a_stop: float = 3.0) -> torch.Tensor:
    """Pure pursuit of the env's route carrot at `cruise`, slowing for the end; yaw faces the carrot."""
    dyn, P = env.dyn, env.cfg.stick
    g = env.goal - dyn.pos
    end = env.rpts[env.r_id, env.r_end]
    d_end = (end - dyn.pos).norm(dim=-1)
    v = torch.minimum(torch.full_like(d_end, cruise), (2 * a_stop * d_end).sqrt())
    vdes = torch.nn.functional.normalize(g, dim=-1) * v[:, None]
    body = env._to_body(vdes)
    err = torch.remainder(torch.atan2(g[:, 1], g[:, 0]) - dyn.yaw + math.pi, 2 * math.pi) - math.pi
    a = torch.stack([body[:, 0] / P.v_h_max, body[:, 1] / P.v_h_max,
                     (body[:, 2] / P.v_up_max).clamp(-1, 1), (-err * 1.5).clamp(-1, 1)], -1)
    return a.clamp(-1, 1)


class Trigger:
    def __init__(self, env, brake: BrakeModel, kind="oracle", net_ckpt=None, thr=0.5, cruise=5.0, lookahead=0.0):
        """lookahead: extra metres beyond the stopping distance (hand over early enough to dodge, not
        just to stop; the policy trained with obstacles 8-18 m ahead)."""
        self.env, self.brake, self.kind, self.thr, self.cruise, self.look = env, brake, kind, thr, cruise, lookahead
        if kind == "net":
            d = torch.load(net_ckpt, map_location=env.dev)
            self.net = HazardNet(d["stack"], d["vec_dim"]).to(env.dev)
            obs = env._obs()
            self.net(obs["latent"][:2], obs["vec"][:2])
            self.net.load_state_dict(d["model"])
            self.net.eval()

    class _Q:
        pass

    @torch.no_grad()
    def __call__(self, obs):
        """-> (hazard [N] bool: brake now along current velocity, clear_to_carrot [N] bool)."""
        env, dyn, r = self.env, self.env.dyn, self.env.cfg.drone_radius
        speed = dyn.vel.norm(dim=-1)
        fwd = torch.stack([torch.cos(dyn.yaw), torch.sin(dyn.yaw), torch.zeros_like(dyn.yaw)], -1)
        to_c = torch.nn.functional.normalize(env.goal - dyn.pos, dim=-1)
        need_c = self.brake.stopping_distance(torch.full_like(speed, self.cruise)) + 1.0
        if self.kind == "oracle":
            Q = self._Q
            Q.query = staticmethod(env._sdf)
            vdir = torch.where((speed > 0.3)[:, None], dyn.vel / speed.clamp_min(1e-6)[:, None], fwd)
            hazard = ray_clearance(Q, dyn.pos, vdir, r) < self.brake.stopping_distance(speed) + self.look
            dist_c = (env.goal - dyn.pos).norm(dim=-1)
            clear_c = ray_clearance(Q, dyn.pos, to_c, r) >= torch.minimum(need_c, dist_c)
            return hazard, clear_c
        clear, logit = self.net(obs["latent"], obs["vec"])
        hazard = (torch.sigmoid(logit[:, -1]) > self.thr) | (clear[:, 0] < self.brake.stopping_distance(speed) + self.look)
        facing = (fwd * to_c).sum(-1) > math.cos(math.radians(25))
        clear_c = facing & (clear[:, 1] >= need_c)
        return hazard, clear_c


def map_los(env, a: torch.Tensor, b: torch.Tensor, clear: float, step: float = 0.25) -> torch.Tensor:
    """Straight segment a->b [N,3] clear of the *map* (scene ESDF only, not the unmapped obstacles)."""
    L = (b - a).norm(dim=-1)
    n = int(max(2, math.ceil(float(L.max()) / step) + 1))
    t = torch.linspace(0, 1, n, device=a.device)
    pts = a[:, None] + (b - a)[:, None] * t[None, :, None]
    return (env.s.esdf.query(pts.reshape(-1, 3)).reshape(len(a), n) > clear).all(1)


@torch.no_grad()
def run(env, model, pilot: str, episodes: int, trig: Trigger | None = None, cruise=5.0,
        min_policy_s=1.0, quiet_s=1.0, max_steps_mult=3.0, seed=0) -> dict:
    torch.manual_seed(seed)
    env.resample_obstacles(seed=seed)
    obs = env.reset()
    n, dev, hz = env.n, env.dev, env.cfg.control_hz
    mode = torch.zeros(n, dtype=torch.long, device=dev)
    t_mode = torch.zeros(n, device=dev)
    quiet = torch.zeros(n, device=dev)
    pol_steps = torch.zeros(n, device=dev)
    switches = torch.zeros(n, device=dev)
    took_over = torch.zeros(n, dtype=torch.bool, device=dev)
    xt_sum = torch.zeros(n, device=dev)
    prev_mode = mode.clone()
    res = {"crash_mode": [], "success": [], "collided": [], "time_s": [], "policy_frac": [], "takeovers": [], "switches": [],
           "mean_xtrack_m": [], "took_over": []}
    done_eps, steps = 0, 0
    max_steps = int(episodes / n * env.max_steps * max_steps_mult) + env.max_steps
    while done_eps < episodes and steps < max_steps:
        steps += 1
        if pilot == "planner":
            act = planner_act(env, cruise)
        elif pilot == "policy":
            act = model.act(obs, deterministic=True)
            mode[:] = POLICY
        else:
            hazard, clear_c = trig(obs)
            quiet = torch.where(hazard, torch.zeros_like(quiet), quiet + 1 / hz)
            take = (mode == PLANNER) & hazard
            back = ((mode == POLICY) & (t_mode >= min_policy_s) & (quiet >= quiet_s) & clear_c
                    & map_los(env, env.dyn.pos, env.goal, env.cfg.drone_radius + 0.5))
            sw = take | back
            took_over |= take
            switches += sw.float()
            mode = torch.where(take, POLICY, torch.where(back, PLANNER, mode))
            t_mode = torch.where(sw, torch.zeros_like(t_mode), t_mode)
            a_pol = model.act(obs, deterministic=True)
            act = torch.where((mode == POLICY)[:, None], a_pol, planner_act(env, cruise))
        t_mode += 1 / hz
        pol_steps += (mode == POLICY).float()
        _, xtrack = env._route_progress()
        xt_sum += xtrack
        prev_mode = mode.clone()
        obs, r, done, info = env.step(act)
        if done.any():
            d = done.nonzero()[:, 0]
            T = env_t = info["final_length"]
            res["success"] += info["final_success"].tolist()
            res["collided"] += info["final_collided"].tolist()
            res["time_s"] += (env_t / hz).tolist()
            res["policy_frac"] += (pol_steps[d] / T).tolist()
            res["takeovers"] += ((switches[d] + 1) // 2).tolist()
            res["switches"] += switches[d].tolist()
            res["mean_xtrack_m"] += (xt_sum[d] / T).tolist()
            res["took_over"] += took_over[d].float().tolist()
            res["crash_mode"] += torch.where(info["final_collided"] > 0, prev_mode[d].float(), -1.0).tolist()
            done_eps += len(d)
            for t in (t_mode, quiet, pol_steps, switches, xt_sum):
                t[d] = 0
            mode[d] = PLANNER
            took_over[d] = False
    k = min(episodes, len(res["success"]))
    out = {"pilot": pilot, "episodes": k}
    cm = torch.tensor(res.pop("crash_mode")[:k])
    out["crash_under_planner"] = round((cm == PLANNER).float().mean().item(), 3)
    out["crash_under_policy"] = round((cm == POLICY).float().mean().item(), 3)
    for key, v in res.items():
        v = torch.tensor(v[:k])
        out[key] = round(v.mean().item(), 3) if len(v) else None
    if pilot == "handoff":
        s, c, to = (torch.tensor(res[x][:k]) for x in ("success", "collided", "took_over"))
        out["missed_trigger_crash"] = round(((to == 0) & (c > 0)).float().mean().item(), 3)
    return out


def evaluate(scene, ckpt, episodes=512, envs=256, device="cuda", trigger="oracle", hazard_ckpt=None, thr=0.5,
             cruise=5.0, pilots=("planner", "policy", "handoff"), chunk=64, out=None, seed=0, lookaheads=(0.0,)):
    from .env import BatchEnv, EnvConfig
    from .ppo import ActorCritic
    cfg = EnvConfig(num_envs=envs, task="route", render_chunk=chunk, radius_clip=2.0, full_quality_frac=0.25,
                    ceiling=15.0, route_obstacle_frac=1.0, route_lead_m=(25.0, 35.0), route_tail_m=(25.0, 35.0),
                    route_speed0=0.0, episode_s=40.0)
    env = BatchEnv(scene, cfg, device=device)
    obs = env.reset()
    model = ActorCritic(cfg.frame_stack, env.VEC_DIM, env.PRIV_DIM).to(env.dev)
    model.value({k: v[:1] for k, v in obs.items()}), model.dist({k: v[:1] for k, v in obs.items()})
    model.load_state_dict(torch.load(ckpt, map_location=env.dev)["model"])
    model.eval()
    rows = []
    jobs = [(p, None) for p in pilots if p != "handoff"] + [("handoff", la) for la in lookaheads if "handoff" in pilots]
    for p, la in jobs:
        t0 = time.time()
        trig = Trigger(env, BrakeModel(), trigger, hazard_ckpt, thr, cruise, la or 0.0) if p == "handoff" else None
        r = run(env, model, p, episodes, trig, cruise=cruise, seed=seed)
        r["trigger"] = f"{trigger}+{la:g}m" if p == "handoff" else None
        r["wall_s"] = round(time.time() - t0, 1)
        print(json.dumps(r), flush=True)
        rows.append(r)
    if out:
        Path(out).write_text(json.dumps({"ckpt": str(ckpt), "scene": scene.name, "cruise": cruise, "rows": rows}, indent=2))
    return rows
