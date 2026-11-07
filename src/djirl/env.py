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
    stick: StickParams = field(default_factory=StickParams)


class BatchEnv:
    VEC_DIM = 13

    def __init__(self, scene: Scene, cfg: EnvConfig, device="cuda", encoder: LatentEncoder | None = None):
        self.s, self.cfg, self.dev = scene.to(device), cfg, torch.device(device)
        n = self.n = cfg.num_envs
        self.dt = 1.0 / cfg.control_hz
        self.max_steps = int(cfg.episode_s * cfg.control_hz)
        self.dyn = StickDynamics(n, cfg.stick, self.dev, self.dt, cfg.substeps)
        self.renderer = Renderer(self.s.gaussians, cfg.width, cfg.height, cfg.hfov_deg, cfg.render_chunk)
        self.encoder = encoder or LatentEncoder(self.dev)
        self.wps = self.s.waypoints
        self.W = len(self.wps)
        self.pitch = torch.full((n,), cfg.gimbal_pitch_deg * torch.pi / 180, device=self.dev)
        self.target = torch.zeros(n, dtype=torch.long, device=self.dev)
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
        return self.s.esdf.query(p)

    def _goal(self):
        return self.wps[self.target]

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

    def _reset_idx(self, idx):
        if len(idx) == 0:
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

        dist = (self._goal() - pos).norm(dim=-1)
        progress = self.prev_dist - dist
        reached = dist < cfg.wp_radius
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
        self._reset_idx(done.nonzero()[:, 0])
        obs = self._obs(reset_mask=done)
        return obs, reward, done, info
