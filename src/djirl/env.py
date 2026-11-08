"""BatchEnv: N drones flying the waypoint loop in one scene, entirely on GPU.

API (za-sim style): reset() -> obs dict; step(action [N,4]) -> (obs, reward, done, info), with
autoreset. obs = {"latent": [N,S,4,h,w] stacked encoder latents, "vec": [N,13] proprio/goal,
"priv": [N,P] privileged critic input (not available on the real drone)}.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .dynamics import StickDynamics, StickParams
from .encoder import LatentEncoder
from .render import Renderer
from .scene import Scene


@dataclass
class EnvConfig:
    num_envs: int = 512
    control_hz: float = 10.0
    substeps: int = 5
    episode_s: float = 60.0
    width: int = 128
    height: int = 96
    hfov_deg: float = 82.0
    gimbal_pitch_deg: float = -10.0
    frame_stack: int = 2
    drone_radius: float = 0.25    # Mini 3 is ~0.25 m across with props, plus margin
    safe_dist: float = 1.0        # clearance shaping kicks in below this
    wp_radius: float = 1.5
    ceiling: float = 30.0
    min_agl: float = 0.4
    spawn_lateral_std: float = 2.0
    height_max: float = 10.0      # downward sensor range (m); -1 when out of range
    height_available: float = 0.8  # fraction of episodes where the downward sensor reads
    # reward weights
    w_progress: float = 1.0
    w_waypoint: float = 5.0
    w_collision: float = 10.0
    w_clearance: float = 0.5
    w_smooth: float = 0.02
    w_alive: float = 0.0
    render_chunk: int = 128
    radius_clip: float = 0.0       # see Renderer; 2.0 for big scenes
    full_quality_frac: float = 1.0
    hazard_labels: bool = False   # add ESDF hazard labels (clearance along velocity, must_brake) to info
    # task: "loop" = fly the scene's waypoint loop; "a2b" = random start -> random goal in free space
    task: str = "loop"
    a2b_min_dist: float = 15.0
    a2b_max_dist: float = 60.0
    a2b_alt: tuple = (1.5, 8.0)       # m above ground for starts/goals
    a2b_clearance: float = 2.0        # starts/goals at least this far from any surface
    a2b_coverage: float = 20.0        # only within this horizontal distance of the capture path (where the splat is good)
    a2b_heading_noise: float = 1.2    # rad, initial yaw = bearing to goal + N(0, this)
    w_success: float = 20.0
    # task "route": follow map-valid A* routes (data/splats/<scene>/routes.pt) toward a carrot point,
    # with unmapped obstacles injected on them (rendered + collidable) that the policy must dodge
    route_carrot_m: float = 6.0
    route_obstacles: int = 24
    route_obstacle_frac: float = 0.7      # episodes that start just before an injected obstacle
    route_xtrack_free: float = 1.5        # m off-route before a penalty applies (dodging is allowed)
    w_xtrack: float = 0.05
    route_seg_m: tuple = (15.0, 35.0)     # segment length for obstacle-free episodes
    route_lead_m: tuple = (8.0, 18.0)     # obstacle episodes start this far before the obstacle
    route_tail_m: tuple = (10.0, 20.0)    # and end this far after it
    route_speed0: float = 4.0             # initial speed ~ U(0, this)
    stick: StickParams = field(default_factory=StickParams)


class BatchEnv:
    VEC_DIM = 13

    def __init__(self, scene: Scene, cfg: EnvConfig, device="cuda", encoder: LatentEncoder | None = None):
        self.s, self.cfg, self.dev = scene.to(device), cfg, torch.device(device)
        n = self.n = cfg.num_envs
        self.dt = 1.0 / cfg.control_hz
        self.max_steps = int(cfg.episode_s * cfg.control_hz)
        self.dyn = StickDynamics(n, cfg.stick, self.dev, self.dt, cfg.substeps)
        self.renderer = Renderer(self.s.gaussians, cfg.width, cfg.height, cfg.hfov_deg, cfg.render_chunk,
                                 radius_clip=cfg.radius_clip, full_frac=cfg.full_quality_frac)
        self.encoder = encoder or LatentEncoder(self.dev)
        self.wps = self.s.waypoints
        self.W = len(self.wps)
        self.pitch = torch.full((n,), cfg.gimbal_pitch_deg * torch.pi / 180, device=self.dev)
        self.target = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.goal = torch.zeros(n, 3, device=self.dev)
        self.ep_success = torch.zeros(n, device=self.dev)
        if cfg.task == "a2b":
            self.cands = self._a2b_candidates()
        self.obst = None
        if cfg.task == "route":
            self._route_init()
        self.t = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.prev_dist = torch.zeros(n, device=self.dev)
        self.last_action = torch.zeros(n, 4, device=self.dev)
        self.has_height = torch.ones(n, dtype=torch.bool, device=self.dev)
        self.ep_return = torch.zeros(n, device=self.dev)
        self.ep_wps = torch.zeros(n, device=self.dev)
        self.latents = None
        # privileged probe offsets: 5x5x3 grid at 1 m in body frame (forward, right, up)
        g = torch.linspace(-2, 2, 5, device=self.dev)
        F_, R_, U_ = torch.meshgrid(g, g, torch.tensor([-1.0, 0.0, 1.0], device=self.dev), indexing="ij")
        self.probe = torch.stack([F_, R_, U_], -1).reshape(-1, 3)
        self.PRIV_DIM = len(self.probe) + 1 + 3 + 3

    # ------------------------------------------------------------ helpers
    def _sdf(self, p):
        d = self.s.esdf.query(p)
        if self.obst is not None and self.obst.n:
            d = torch.minimum(d, self.obst.sdf(p))
        return d

    def _goal(self):
        return self.goal if self.cfg.task in ("a2b", "route") else self.wps[self.target]

    # ------------------------------------------------------------ route task
    def _route_init(self):
        from pathlib import Path

        from .obstacles import Obstacles
        f = Path(__file__).resolve().parents[2] / "data" / "splats" / self.s.name / "routes.pt"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing: run `djirl routes --scene {self.s.name}` first")
        R = torch.load(f)
        self.rpts, self.rlen = R["pts"].to(self.dev), R["len"].to(self.dev)
        self.rsp = float(R["spacing"])
        n = self.n
        self.r_id = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.r_idx = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.r_end = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.r_has_obst = torch.zeros(n, dtype=torch.bool, device=self.dev)
        self.obst = Obstacles(self.dev)
        self.resample_obstacles()
        print(f"route task: {len(self.rpts)} routes, {self.obst.n} unmapped obstacles", flush=True)

    @torch.no_grad()
    def resample_obstacles(self, seed=None):
        """Place K unmapped obstacles on random routes (25-75% along); renderer gets the merged splat."""
        from .obstacles import merge
        g = torch.Generator(device=self.dev)
        g.manual_seed(int(seed if seed is not None else torch.randint(0, 2**31, ()).item()))
        K, Rn = self.cfg.route_obstacles, len(self.rpts)
        r = torch.randint(0, Rn, (K,), device=self.dev, generator=g)
        L = self.rlen[r]
        idx = (L.float() * (0.25 + 0.5 * torch.rand(K, device=self.dev, generator=g))).long()
        c = self.rpts[r, idx]
        # ground under each obstacle: first scene-ESDF surface scanning down from the route altitude
        zs = torch.arange(0, 1, 0.01, device=self.dev)[None] * (c[:, 2:3] + 3.0)
        col = torch.stack([c[:, None, 0].expand(-1, zs.shape[1]), c[:, None, 1].expand(-1, zs.shape[1]),
                           c[:, 2:3] - zs], -1)
        hit = self.s.esdf.query(col.reshape(-1, 3)).reshape(K, -1) < 0.05
        ground = torch.where(hit.any(1), col[torch.arange(K), hit.float().argmax(1), 2], torch.zeros(K, device=self.dev))
        self.obst.sample_at(c[:, :2], c[:, 2], ground, gen=g)
        self.ob_route, self.ob_idx = r, idx
        self.renderer.g = merge(self.s.gaussians, self.obst.gauss)

    def _spawn_route(self, idx):
        cfg, m, dev = self.cfg, len(idx), self.dev
        sp = self.rsp
        use_ob = torch.rand(m, device=dev) < cfg.route_obstacle_frac
        k = torch.randint(0, self.obst.n, (m,), device=dev)
        r_ob, i_ob = self.ob_route[k], self.ob_idx[k]
        r_rand = torch.randint(0, len(self.rpts), (m,), device=dev)
        r = torch.where(use_ob, r_ob, r_rand)
        L = self.rlen[r]
        U = lambda r: r[0] + (r[1] - r[0]) * torch.rand(m, device=dev)
        lead = (U(cfg.route_lead_m) / sp).long()
        tail = (U(cfg.route_tail_m) / sp).long()
        s0_ob = (i_ob - lead).clamp(min=0)
        e_ob = torch.minimum(i_ob + tail, L - 1)
        seg = ((cfg.route_seg_m[0] + (cfg.route_seg_m[1] - cfg.route_seg_m[0]) * torch.rand(m, device=dev)) / sp).long()
        s0_r = (torch.rand(m, device=dev) * (L - seg).clamp(min=1).float()).long()
        e_r = torch.minimum(s0_r + seg, L - 1)
        s0 = torch.where(use_ob, s0_ob, s0_r)
        e = torch.where(use_ob, e_ob, e_r)
        p = self.rpts[r, s0].clone()
        nxt = self.rpts[r, torch.minimum(s0 + 4, L - 1)]
        tan = nxt - p
        yaw = torch.atan2(tan[:, 1], tan[:, 0]) + torch.randn(m, device=dev) * 0.25
        p[:, :2] += torch.randn(m, 2, device=dev) * 0.4
        speed0 = torch.rand(m, device=dev) * cfg.route_speed0                           # often already moving, like mid-route
        v0 = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], -1) * speed0[:, None]
        return p, yaw, v0, r, s0, e, use_ob

    def _route_progress(self):
        """Advance each env's nearest route index within a forward window; returns (idx, xtrack)."""
        W = 24
        off = torch.arange(-2, W, device=self.dev)
        cand = (self.r_idx[:, None] + off[None]).clamp(min=0)
        cand = torch.minimum(cand, (self.rlen[self.r_id] - 1)[:, None])
        pts = self.rpts[self.r_id[:, None], cand]
        d = (pts - self.dyn.pos[:, None]).norm(dim=-1)
        j = d.argmin(1)
        new = cand[torch.arange(self.n, device=self.dev), j]
        return torch.maximum(new, self.r_idx), d.min(1).values

    def _route_carrot(self):
        ci = torch.minimum(self.r_idx + int(self.cfg.route_carrot_m / self.rsp), self.r_end)
        return self.rpts[self.r_id, ci]

    def _a2b_candidates(self):
        """Free-space points (1 m grid) at flight altitudes, near the capture path (scene waypoints are
        spread along it), with clearance. Starts and goals are sampled from these."""
        cfg, e = self.cfg, self.s.esdf
        lo, hi = e.origin, e.upper
        xs = torch.arange(float(lo[0]) + 1, float(hi[0]) - 1, 1.0, device=self.dev)
        ys = torch.arange(float(lo[1]) + 1, float(hi[1]) - 1, 1.0, device=self.dev)
        zs = torch.arange(cfg.a2b_alt[0], cfg.a2b_alt[1] + 1e-6, 1.0, device=self.dev)
        X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
        P = torch.stack([X, Y, Z], -1).reshape(-1, 3)
        # distance (xy) to the polyline through the waypoints (closed loop)
        a, b = self.wps[:, :2], torch.roll(self.wps, -1, 0)[:, :2]
        ab = b - a
        t = (((P[:, None, :2] - a[None]) * ab[None]).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-6)).clamp(0, 1)
        d = (P[:, None, :2] - (a[None] + t[..., None] * ab[None])).norm(dim=-1).min(1).values
        P = P[d < cfg.a2b_coverage]
        P = P[self._sdf(P) > cfg.a2b_clearance]
        if len(P) < 100:
            raise RuntimeError(f"only {len(P)} free a2b points; check scene/waypoints/alt range")
        print(f"a2b: {len(P)} candidate start/goal points", flush=True)
        return P

    def _to_body(self, v_world):
        c, s = torch.cos(self.dyn.yaw), torch.sin(self.dyn.yaw)
        return torch.stack([c * v_world[:, 0] + s * v_world[:, 1], s * v_world[:, 0] - c * v_world[:, 1], v_world[:, 2]], -1)

    def _height(self, pos):
        steps = torch.linspace(0, self.cfg.height_max, 41, device=self.dev)
        pts = pos[:, None, :] - torch.stack([torch.zeros_like(steps)] * 2 + [steps], -1)[None]
        hit = self._sdf(pts.reshape(-1, 3)).reshape(len(pos), -1) < 0.1
        first = torch.where(hit.any(1), hit.float().argmax(1), torch.full((len(pos),), -1, device=self.dev))
        h = torch.where(first >= 0, steps[first.clamp_min(0)], torch.full_like(first, -1.0, dtype=torch.float))
        return h

    # ------------------------------------------------------------ reset
    def _spawn(self, idx):
        m = len(idx)
        cands = 8
        tgt = torch.randint(0, self.W, (m, cands), device=self.dev)
        a, b = self.wps[(tgt - 1) % self.W], self.wps[tgt]
        frac = torch.rand(m, cands, 1, device=self.dev)
        p = a + (b - a) * frac
        seg = torch.nn.functional.normalize((b - a)[..., :2], dim=-1)
        lat = torch.stack([-seg[..., 1], seg[..., 0]], -1) * torch.randn(m, cands, 1, device=self.dev) * self.cfg.spawn_lateral_std
        p = p + torch.cat([lat, torch.randn(m, cands, 1, device=self.dev) * 0.7], -1)
        ok = (self._sdf(p.reshape(-1, 3)).reshape(m, cands) > self.cfg.safe_dist) & self.s.esdf.in_bounds(p)
        pick = torch.where(ok.any(1), ok.float().argmax(1), torch.zeros(m, dtype=torch.long, device=self.dev))
        r = torch.arange(m, device=self.dev)
        p, tgt = p[r, pick], tgt[r, pick]
        d = self.wps[tgt] - p
        yaw = torch.atan2(d[:, 1], d[:, 0]) + torch.randn(m, device=self.dev) * 0.3
        v0 = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], -1) * torch.rand(m, 1, device=self.dev) * 3
        return p, yaw, v0, tgt

    def _spawn_a2b(self, idx):
        cfg, C, m = self.cfg, self.cands, len(idx)
        k = 16
        si = torch.randint(0, len(C), (m,), device=self.dev)
        p = C[si]
        gi = torch.randint(0, len(C), (m, k), device=self.dev)
        g = C[gi]
        dd = (g - p[:, None]).norm(dim=-1)
        ok = (dd > cfg.a2b_min_dist) & (dd < cfg.a2b_max_dist)
        pick = torch.where(ok.any(1), ok.float().argmax(1), dd.argmax(1))
        goal = g[torch.arange(m, device=self.dev), pick]
        d = goal - p
        yaw = torch.atan2(d[:, 1], d[:, 0]) + torch.randn(m, device=self.dev) * cfg.a2b_heading_noise
        return p, yaw, torch.zeros(m, 3, device=self.dev), goal

    def _reset_idx(self, idx):
        if len(idx) == 0:
            return
        if self.cfg.task == "route":
            p, yaw, v0, r, s0, e, use_ob = self._spawn_route(idx)
            self.r_id[idx], self.r_idx[idx], self.r_end[idx], self.r_has_obst[idx] = r, s0, e, use_ob
            self.dyn.reset(idx, p, yaw, v0)
            self.goal[idx] = self._route_carrot()[idx]
            self.ep_success[idx] = 0
            self.target[idx] = 0
            self.t[idx] = 0
            self.prev_dist[idx] = (self.r_idx[idx] * self.rsp).float()   # route task: prev arc length
            self.last_action[idx] = 0
            self.has_height[idx] = torch.rand(len(idx), device=self.dev) < self.cfg.height_available
            self.ep_return[idx] = 0
            self.ep_wps[idx] = 0
            return
        if self.cfg.task == "a2b":
            p, yaw, v0, goal = self._spawn_a2b(idx)
            self.goal[idx] = goal
            tgt = torch.zeros(len(idx), dtype=torch.long, device=self.dev)
            self.ep_success[idx] = 0
            self.dyn.reset(idx, p, yaw, v0)
            self.target[idx] = tgt
            self.t[idx] = 0
            self.prev_dist[idx] = (goal - p).norm(dim=-1)
            self.last_action[idx] = 0
            self.has_height[idx] = torch.rand(len(idx), device=self.dev) < self.cfg.height_available
            self.ep_return[idx] = 0
            self.ep_wps[idx] = 0
            return
        p, yaw, v0, tgt = self._spawn(idx)
        self.dyn.reset(idx, p, yaw, v0)
        self.target[idx] = tgt
        self.t[idx] = 0
        self.prev_dist[idx] = (self.wps[tgt] - p).norm(dim=-1)
        self.last_action[idx] = 0
        self.has_height[idx] = torch.rand(len(idx), device=self.dev) < self.cfg.height_available
        self.ep_return[idx] = 0
        self.ep_wps[idx] = 0

    # ------------------------------------------------------------ observation
    def _obs(self, reset_mask=None):
        rgb = self.renderer.render(self.dyn.pos, self.dyn.yaw, self.pitch)
        z = self.encoder(rgb)
        if self.latents is None:
            self.latents = z[:, None].repeat(1, self.cfg.frame_stack, 1, 1, 1)
        else:
            self.latents = torch.cat([self.latents[:, 1:], z[:, None]], 1)
            if reset_mask is not None and reset_mask.any():
                self.latents[reset_mask] = z[reset_mask, None]
        g = self._goal() - self.dyn.pos
        dist = g.norm(dim=-1, keepdim=True)
        h = self._height(self.dyn.pos)
        h = torch.where(self.has_height, h, torch.full_like(h, -1.0))
        vec = torch.cat([
            self._to_body(g / dist.clamp_min(1e-3)), (dist / 20).clamp(max=2),
            self.dyn.body_velocity() / 8, self.dyn.yaw_rate[:, None] / 1.6,
            (h[:, None] / self.cfg.height_max), self.last_action,
        ], -1)
        c, s = torch.cos(self.dyn.yaw), torch.sin(self.dyn.yaw)
        pr = self.probe
        world_off = torch.stack([c[:, None] * pr[:, 0] + s[:, None] * pr[:, 1],
                                 s[:, None] * pr[:, 0] - c[:, None] * pr[:, 1],
                                 pr[:, 2].expand(self.n, -1)], -1)
        probes = self._sdf((self.dyn.pos[:, None] + world_off).reshape(-1, 3)).reshape(self.n, -1)
        priv = torch.cat([probes.clamp(-1, 5) / 5, self._sdf(self.dyn.pos)[:, None].clamp(-1, 5) / 5,
                          self.dyn.vel / 8, self.dyn.wind], -1)
        return {"latent": self.latents, "vec": vec, "priv": priv, "rgb": rgb}

    # ------------------------------------------------------------ api
    def reset(self):
        self._reset_idx(torch.arange(self.n, device=self.dev))
        self.latents = None
        return self._obs()

    def step(self, action: torch.Tensor):
        cfg = self.cfg
        action = action.clamp(-1, 1)
        subs = self.dyn.step(action)
        pos = self.dyn.pos
        sweep = torch.stack(subs, 1)  # [N,sub,3]
        sdf_sweep = self._sdf(sweep.reshape(-1, 3)).reshape(self.n, -1)
        collided = (sdf_sweep < cfg.drone_radius).any(1)
        oob = ~self.s.esdf.in_bounds(pos) | (pos[:, 2] > cfg.ceiling)
        clearance = sdf_sweep.min(1).values

        if cfg.task == "route":
            self.r_idx, xtrack = self._route_progress()
            arc = (self.r_idx * self.rsp).float()
            progress = arc - self.prev_dist
            self.prev_dist = arc
            self.goal = self._route_carrot()
            dist = (self.rpts[self.r_id, self.r_end] - pos).norm(dim=-1)
            reached = (self.r_idx >= self.r_end - 2) & (dist < 2.5)
        else:
            dist = (self._goal() - pos).norm(dim=-1)
            progress = self.prev_dist - dist
            reached = dist < cfg.wp_radius
        if cfg.task == "a2b":
            self.prev_dist = dist
        elif cfg.task == "route":
            pass
        else:
            self.target = torch.where(reached, (self.target + 1) % self.W, self.target)
            self.prev_dist = torch.where(reached, (self._goal() - pos).norm(dim=-1), dist)

        smooth = ((action - self.last_action) ** 2).sum(-1)
        reward = (cfg.w_progress * progress + cfg.w_waypoint * reached.float()
                  - cfg.w_clearance * (cfg.safe_dist - clearance).clamp(min=0)
                  - cfg.w_smooth * smooth + cfg.w_alive
                  - cfg.w_collision * (collided | oob).float())
        self.last_action = action
        self.t += 1
        terminated = collided | oob
        if cfg.task == "route":
            reward = reward - cfg.w_xtrack * (xtrack - cfg.route_xtrack_free).clamp(min=0)
        if cfg.task in ("a2b", "route"):
            success = reached & ~collided & ~oob
            reward = reward + cfg.w_success * success.float()
            terminated = terminated | success
            self.ep_success = torch.maximum(self.ep_success, success.float())
        truncated = (self.t >= self.max_steps) & ~terminated
        done = terminated | truncated
        self.ep_return += reward
        self.ep_wps += reached.float()

        info = {"terminated": terminated, "truncated": truncated, "collided": collided, "oob": oob,
                "speed": self.dyn.vel.norm(dim=-1), "clearance": clearance}
        if done.any():
            info["final_return"] = self.ep_return[done].clone()
            info["final_length"] = self.t[done].float().clone()
            info["final_waypoints"] = self.ep_wps[done].clone()
            info["final_collided"] = collided[done].float()
            info["final_success"] = self.ep_success[done].clone()
            if cfg.task == "route":
                ob = self.r_has_obst[done]
                info["final_success_obstacle"] = self.ep_success[done][ob].clone()
                info["final_collided_obstacle"] = collided[done][ob].float()
        if cfg.hazard_labels:  # labels for the post-step state, before autoreset replaces it
            from .hazard import BrakeModel, hazard_labels
            info["hazard"] = hazard_labels(self, getattr(self, "brake", None) or BrakeModel())
        self._reset_idx(done.nonzero()[:, 0])
        obs = self._obs(reset_mask=done)
        return obs, reward, done, info
