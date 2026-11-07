"""Batched DJI virtual-stick response model (velocity mode).

DJI's flight controller turns stick setpoints into motion, so we model that closed loop, not
rotors: each channel lags toward its setpoint (time constant tau) under separate accel/brake
caps, after a command delay. All parameters are guesses until fitted to Mini 3 flight logs.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch


@dataclass
class StickParams:
    v_h_max: float = 8.0      # m/s horizontal speed at full stick (Mini 3 Normal mode ~10)
    v_up_max: float = 3.0     # m/s
    v_down_max: float = 2.5   # m/s
    yaw_rate_max: float = 1.6  # rad/s (~90 deg/s)
    tau: float = 0.4          # s, horizontal velocity lag
    tau_z: float = 0.3        # s
    tau_yaw: float = 0.2      # s
    a_acc: float = 4.0        # m/s^2 horizontal accel cap
    a_brake: float = 6.0      # m/s^2 horizontal decel cap (braking needs backward tilt too)
    a_z: float = 4.0          # m/s^2
    delay_steps: int = 2      # control steps of command latency (2 @ 10 Hz = 200 ms)
    wind_std: float = 0.3     # m/s residual drift the FC does not cancel
    dr: float = 0.2           # domain randomisation: each float param x U(1-dr, 1+dr) per env


class StickDynamics:
    """State tensors are [B,...] on one device. Action a in [-1,1]^4 =
    (forward, right, up, yaw_right), like the RC sticks."""

    FLOATS = ("v_h_max", "v_up_max", "v_down_max", "yaw_rate_max", "tau", "tau_z", "tau_yaw",
              "a_acc", "a_brake", "a_z")

    def __init__(self, n: int, params: StickParams, device, control_dt: float = 0.1, substeps: int = 5):
        self.n, self.p, self.dev = n, params, device
        self.dt, self.sub = control_dt / substeps, substeps
        self.pos = torch.zeros(n, 3, device=device)
        self.vel = torch.zeros(n, 3, device=device)
        self.yaw = torch.zeros(n, device=device)
        self.yaw_rate = torch.zeros(n, device=device)
        self.buf = torch.zeros(n, max(params.delay_steps, 0) + 1, 4, device=device)
        self.k = {f: torch.full((n,), getattr(params, f), device=device) for f in self.FLOATS}
        self.wind = torch.zeros(n, 3, device=device)

    def reset(self, idx: torch.Tensor, pos, yaw, vel=None):
        self.pos[idx], self.yaw[idx] = pos, yaw
        self.vel[idx] = 0 if vel is None else vel
        self.yaw_rate[idx] = 0
        self.buf[idx] = 0
        m = len(idx)
        for f in self.FLOATS:
            u = 1 + self.p.dr * (2 * torch.rand(m, device=self.dev) - 1)
            self.k[f][idx] = getattr(self.p, f) * u
        w = torch.randn(m, 3, device=self.dev) * self.p.wind_std
        w[:, 2] *= 0.2
        self.wind[idx] = w

    def step(self, action: torch.Tensor) -> list[torch.Tensor]:
        """Advance one control step; returns the positions after every substep (for swept collision)."""
        self.buf = torch.roll(self.buf, -1, dims=1)
        self.buf[:, -1] = action.clamp(-1, 1)
        a = self.buf[:, 0]
        k = self.k
        c, s = torch.cos(self.yaw), torch.sin(self.yaw)
        fwd, right = a[:, 0] * k["v_h_max"], a[:, 1] * k["v_h_max"]
        v_cmd_xy = torch.stack([c * fwd + s * right, s * fwd - c * right], -1)  # body FRU -> ENU
        v_cmd_xy = v_cmd_xy * (k["v_h_max"] / v_cmd_xy.norm(dim=-1).clamp_min(k["v_h_max"]))[:, None]
        v_cmd_z = torch.where(a[:, 2] >= 0, a[:, 2] * k["v_up_max"], a[:, 2] * k["v_down_max"])
        r_cmd = -a[:, 3] * k["yaw_rate_max"]  # yaw right = clockwise = negative ENU yaw rate
        out = []
        for _ in range(self.sub):
            v_xy = self.vel[:, :2]
            acc = (v_cmd_xy - v_xy) / k["tau"][:, None]
            braking = (acc * v_xy).sum(-1) < 0
            cap = torch.where(braking, k["a_brake"], k["a_acc"])
            acc = acc * (cap / acc.norm(dim=-1).clamp_min(cap))[:, None]
            az = ((v_cmd_z - self.vel[:, 2]) / k["tau_z"]).clamp(-k["a_z"], k["a_z"])
            self.vel = self.vel + torch.cat([acc, az[:, None]], -1) * self.dt
            self.yaw_rate = self.yaw_rate + (r_cmd - self.yaw_rate) / k["tau_yaw"] * self.dt
            self.yaw = torch.remainder(self.yaw + self.yaw_rate * self.dt + torch.pi, 2 * torch.pi) - torch.pi
            self.pos = self.pos + (self.vel + self.wind) * self.dt
            out.append(self.pos)
        return out

    def body_velocity(self) -> torch.Tensor:
        """Velocity in body FRU (forward, right, up), what DJI telemetry reports after rotation."""
        c, s = torch.cos(self.yaw), torch.sin(self.yaw)
        vx, vy = self.vel[:, 0], self.vel[:, 1]
        return torch.stack([c * vx + s * vy, s * vx - c * vy, self.vel[:, 2]], -1)


def params_from_dict(d: dict) -> StickParams:
    names = {f.name for f in fields(StickParams)}
    return StickParams(**{k: v for k, v in d.items() if k in names})
