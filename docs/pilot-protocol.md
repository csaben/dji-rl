# Pilot link: phone mission runner <-> cradle hazard/policy server

The phone (Flight Lab) flies the planned mission with its own GPS mission runner, exactly as today.
While it flies, it streams camera frames and telemetry to the pilot server on cradle. The server runs
the hazard classifier on each frame. When something the map doesn't know about is in the way, it says
`want: "policy"` and streams stick commands from the learned avoidance policy. When the way back to the
route is clear, it says `want: "mission"` with the route vertex to resume from.

    .venv/bin/djirl pilot-serve      # ws://100.80.98.4:8971/ (tailnet only), GPU 1

## Connection

WebSocket `ws://100.80.98.4:8971/`. One session at a time.

### 1. hello (text, phone -> server)

```json
{"type": "hello", "mission": <the mission JSON the runner is flying, or "latest">, "mode": "shadow" | "policy"}
```

The mission must be one launched from the sim planner (it needs `sim_request`, `route`, `route_rel`,
`alignment`). Reply:

```json
{"type": "ready", "scene": "house-worldlabs-refined", "route_n": 5, "path_m": 65.0, "mode": "shadow",
 "lookahead_m": 4.0, "thr": 0.5, "warnings": [], "log": "runs/pilot/20261108_101500"}
```

or `{"type": "error", "error": "..."}`.

### 2. frame (binary, phone -> server), about 10 Hz

`[u32 big-endian header length][header JSON, UTF-8][JPEG bytes]`

```json
{"seq": 123, "t_ms": 456789,
 "pos_rel": {"e": 3.2, "n": -1.1, "up": 8.0},
 "vel_ned": [1.2, 0.3, -0.1],
 "yaw_deg": 49.4,
 "gimbal_pitch_deg": -10,
 "flying_by": "mission",
 "sticks_now": [0.4, 0.0, 0.0, 0.0],
 "ground_h_m": null}
```

| field | meaning |
|---|---|
| `seq` | frame counter |
| `t_ms` | phone monotonic ms when the frame came out of the SDK |
| `pos_rel` | **real** metres east/north/up from the takeoff spot, the same frame as `route_rel` (scene offsets x `scale_corr`, rotated by `yaw_trim_deg`) |
| `vel_ned` | aircraft velocity, m/s, north/east/down (DJI's frame) |
| `yaw_deg` | aircraft compass yaw (0 = north, clockwise) |
| `flying_by` | who is actually flying right now: `mission`, `policy`, `hover`, or `rc` (pilot took the sticks) |
| `cmd_vel` | optional: the velocity-mode virtual-stick command currently sent, `{fwd_mps, right_mps, up_mps, yaw_rate_dps}` (+ = clockwise) |
| `sticks_now` | optional alternative: normalised `[fwd, right, up, yaw_right]` in -1..1 |
| `ground_h_m` | optional: downward-sensor height, or null |

Telemetry should be the latest sample at the frame's time. With no active mission, send `pos_rel`/`vel_ned`/`yaw_deg` as null: the reply is `{"idle": true, "want": "mission"}`.

**Camera:**
- Send 4:3 if the stream allows it; 16:9 is centre-cropped to 4:3, which narrows the field of view. The policy was trained on 82 deg horizontal FOV, 4:3.
- Gimbal at -10 deg.
- Send at least 320x240, JPEG quality around 80.
- The policy only sees 128x96, so more resolution doesn't help.

The server handles only the newest frame. If frames arrive faster than it processes them, it drops the older ones.

### 3. cmd (text, server -> phone), one per processed frame

```json
{"type": "cmd", "seq": 123, "t_ms": 456789, "hazard": false, "p_brake": 0.03, "clear_m": 14.2, "clear_fwd_m": 15.0,
 "stop_m": 7.3, "want": "mission", "route_progress_m": 22.5, "xtrack_m": 0.4, "server_ms": 9.1}
```

| `want` | what the phone should do |
|---|---|
| `mission` | keep flying the mission |
| `policy` | apply `sticks` (below) as virtual-stick velocity commands, body frame |
| `hover` | stop, hold position, and alert the pilot (policy timed out after 20 s, or a server error) |

In `shadow` mode, `want: "policy"` means "would be avoiding now". It releases once the hazard has been quiet for 1 s, because the mission keeps flying.

While `want` is `policy`, each cmd carries `sticks`:

```json
{"fwd_mps": 2.1, "right_mps": -1.4, "up_mps": 0.0, "yaw_rate_dps": 12.0, "norm": [0.26, -0.18, 0.0, 0.13]}
```

- `fwd`/`right` are body-frame horizontal velocity.
- `up` is vertical velocity.
- `yaw_rate_dps`: positive means clockwise (yaw right).
- The horizontal speed is capped at 4 m/s (`--speed-cap`).

When it switches back to `mission`, the cmd carries:

```json
"rejoin": {"route_idx": 3, "e_m": 12.0, "n_m": -4.5, "up_m": 8.0}
```

Resume the mission at route vertex `route_idx` of `route_rel` (the vertex at or after the carrot, 6 m ahead on the route). `reason` is set on switches: `p_brake`, `clearance`, `clear`, `policy_timeout`.

### 4. ping (text, optional)

Send `{"type": "ping", "t_ms": ...}` and get back `{"type": "pong", ...}`. Use it to measure the round-trip time.

## Phone-side rules

**Mode `shadow`** (first flights):
- Never apply `sticks`.
- Fly the mission normally.
- Show `hazard`, `p_brake`, and `want` on screen.
- Beep when `want` becomes `policy`.

**Mode `policy` with `mission.obstacle_handoff == "policy"`:**
- When `want: "policy"` arrives, pause the mission runner and send the `sticks` as virtual-stick velocity commands. Hold the last command until the next cmd.
- If no cmd arrives for 300 ms while under the policy, or the socket drops: send zero velocity (hover) and report `flying_by: "hover"`.
- When `want: "mission"` arrives, resume the mission from `rejoin.route_idx`.
- When `want: "hover"` arrives: hover and alert.

**Mode `policy` with `obstacle_handoff == "human"`:** on `want: "policy"`, hover and alert the pilot. Never apply `sticks`.

**Always:**
- RC stick input wins, so report `flying_by: "rc"`. The server stops driving whenever `flying_by` is `rc` or `hover`.
- Keep the existing holds: GPS lost, satellites < 10, low battery.
- While under the policy, widen the off-route hold to about 8 m. The policy dodges sideways.
- Keep the ceiling and geofence in force.

## Logging

The server writes `runs/pilot/<stamp>/` on cradle:
- `mission.json`;
- `log.jsonl`, one line per frame: header, cmd, and pose in the scene frame;
- `frames/*.jpg`: every 5th frame, plus every frame with a hazard or under the policy.

These logs feed the replay and retraining of the hazard model on real frames.

## Testing without the drone

```
.venv/bin/python tools/pilot_sim_client.py --episodes 20
```

This plays the phone's role against a running server. It flies sim missions with an unmapped obstacle, sends rendered frames and synthetic telemetry in this exact protocol, and follows `want`/`sticks`/`rejoin`.
