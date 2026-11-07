"""PPO with an asymmetric critic. Actor sees only what the real drone has (encoder latents,
velocity, goal direction from GPS, height, last action); the critic also sees privileged SDF probes."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .env import BatchEnv


@dataclass
class PPOConfig:
    steps_per_iter: int = 32
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    ent: float = 0.003
    vf: float = 0.5
    max_grad_norm: float = 1.0
    minutes: float = 90.0
    max_iters: int = 10_000_000
    log_every: int = 5
    save_every: int = 50


class LatentTrunk(nn.Module):
    def __init__(self, stack: int, ch: int = 4, out: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(stack * ch, 32, 3, padding=1), nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ELU(), nn.Flatten(),
            nn.LazyLinear(out), nn.ELU())

    def forward(self, z):
        return self.net(z.flatten(1, 2))


class ActorCritic(nn.Module):
    def __init__(self, stack: int, vec_dim: int, priv_dim: int, act_dim: int = 4):
        super().__init__()
        self.a_trunk, self.c_trunk = LatentTrunk(stack), LatentTrunk(stack)
        self.actor = nn.Sequential(nn.Linear(256 + vec_dim, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
                                   nn.Linear(128, act_dim))
        self.critic = nn.Sequential(nn.Linear(256 + vec_dim + priv_dim, 256), nn.ELU(), nn.Linear(256, 128),
                                    nn.ELU(), nn.Linear(128, 1))
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.7))
        nn.init.zeros_(self.actor[-1].bias)
        self.actor[-1].weight.data.mul_(0.01)

    def dist(self, obs):
        mu = self.actor(torch.cat([self.a_trunk(obs["latent"]), obs["vec"]], -1))
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, obs):
        return self.critic(torch.cat([self.c_trunk(obs["latent"]), obs["vec"], obs["priv"]], -1))[:, 0]

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        d = self.dist(obs)
        return d.mean if deterministic else d.sample()


def _slice(obs, idx):
    return {k: v[idx] for k, v in obs.items() if k != "rgb"}


def train(env: BatchEnv, run_dir: Path, cfg: PPOConfig, meta: dict | None = None, resume: Path | None = None):
    from torch.utils.tensorboard import SummaryWriter
    dev = env.dev
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({"ppo": asdict(cfg), **(meta or {})}, indent=2, default=str))
    tb = SummaryWriter(str(run_dir / "tb"))
    obs = env.reset()
    model = ActorCritic(env.cfg.frame_stack, env.VEC_DIM, env.PRIV_DIM).to(dev)
    model.value(_slice(obs, slice(0, 2)))  # materialise lazy layers
    model.dist(_slice(obs, slice(0, 2)))
    if resume:
        model.load_state_dict(torch.load(resume, map_location=dev)["model"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    T, N = cfg.steps_per_iter, env.n
    buf = {k: torch.zeros(T, *v.shape, device=dev, dtype=v.dtype) for k, v in obs.items() if k != "rgb"}
    acts = torch.zeros(T, N, 4, device=dev)
    logp = torch.zeros(T, N, device=dev)
    rews = torch.zeros(T, N, device=dev)
    dones = torch.zeros(T, N, device=dev)
    vals = torch.zeros(T, N, device=dev)
    t0 = time.time()
    total_steps, it = 0, 0
    stats: dict[str, list] = {}

    def log_stat(k, v):
        stats.setdefault(k, []).append(float(v))

    while it < cfg.max_iters and (time.time() - t0) < cfg.minutes * 60:
        if hasattr(env, "resample_obstacles") and env.cfg.task == "route" and it > 0 and it % 10 == 0:
            env.resample_obstacles()   # new unmapped obstacles: no memorising; restart episodes cleanly
            obs = env.reset()
        model.eval()
        for t in range(T):
            o = _slice(obs, slice(None))
            with torch.no_grad():
                d = model.dist(o)
                a = d.sample()
                v = model.value(o)
            for k in buf:
                buf[k][t] = o[k]
            acts[t], logp[t], vals[t] = a, d.log_prob(a).sum(-1), v
            obs, r, done, info = env.step(a)
            # truncation: bootstrap with the value estimate instead of treating it as a terminal
            r = r + cfg.gamma * v * info["truncated"].float()
            rews[t], dones[t] = r, done.float()
            log_stat("speed", info["speed"].mean())
            log_stat("collide_rate", info["collided"].float().mean())
            if "final_return" in info:
                for k in ("final_return", "final_length", "final_waypoints", "final_collided", "final_success",
                          "final_success_obstacle", "final_collided_obstacle"):
                    if k in info:
                        stats.setdefault(k, []).extend(info[k].tolist())
        with torch.no_grad():
            last_v = model.value(_slice(obs, slice(None)))
        adv = torch.zeros_like(rews)
        gae = torch.zeros(N, device=dev)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else vals[t + 1]
            nonterm = 1 - dones[t]
            delta = rews[t] + cfg.gamma * nv * nonterm - vals[t]
            gae = delta + cfg.gamma * cfg.lam * nonterm * gae
            adv[t] = gae
        ret = adv + vals
        model.train()
        flat = lambda x: x.reshape(T * N, *x.shape[2:])
        B = {k: flat(v) for k, v in buf.items()}
        fa, flp, fadv, fret = flat(acts), flat(logp), flat(adv), flat(ret)
        fadv = (fadv - fadv.mean()) / (fadv.std() + 1e-8)
        mb = T * N // cfg.minibatches
        for _ in range(cfg.epochs):
            perm = torch.randperm(T * N, device=dev)
            for i in range(cfg.minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                o = {k: v[idx] for k, v in B.items()}
                d = model.dist(o)
                nlp = d.log_prob(fa[idx]).sum(-1)
                ratio = (nlp - flp[idx]).exp()
                pg = -torch.min(ratio * fadv[idx], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * fadv[idx]).mean()
                vl = ((model.value(o) - fret[idx]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pg + cfg.vf * vl - cfg.ent * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()
        it += 1
        total_steps += T * N
        if it % cfg.log_every == 0:
            el = time.time() - t0
            row = {"iter": it, "steps": total_steps, "sps": round(total_steps / el), "min": round(el / 60, 1),
                   "pg": round(pg.item(), 4), "vl": round(vl.item(), 3), "std": round(model.log_std.exp().mean().item(), 3)}
            for k, v in stats.items():
                if v:
                    row[k] = round(sum(v) / len(v), 3)
                    tb.add_scalar(f"rollout/{k}", row[k], total_steps)
            for k in ("pg", "vl", "std", "sps"):
                tb.add_scalar(f"train/{k}", row[k], total_steps)
            print(json.dumps(row), flush=True)
            with (run_dir / "log.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            stats = {}
        if it % cfg.save_every == 0:
            torch.save({"model": model.state_dict(), "iter": it, "steps": total_steps}, run_dir / "policy.pt")
    torch.save({"model": model.state_dict(), "iter": it, "steps": total_steps}, run_dir / "policy.pt")
    return model
