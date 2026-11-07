# Quadrotor simulator survey for dji-rl

Written 2026-11-07. Repo dates, licenses and release tags come from the GitHub API on that day.
Throughput numbers are what each paper or README reports, on its own hardware. Treat them as
order-of-magnitude figures, not benchmarks we have run. "Unverified" marks claims I could not
confirm from a primary source.

## TL;DR

**Build a minimal custom sim in Python/PyTorch on top of gsplat. Do not adopt an existing simulator.**

Two reasons decide it. Neither depends on taste.

1. **No mainstream sim renders a Gaussian splat camera, batched, at RL speed.** The only
   general-purpose sim that renders splats out of the box is Isaac Sim/Isaac Lab, through
   NuRec on RTX. GaussGym's authors measure it at about 1 FPS per env at 640x480, against
   about 25 FPS per env for gsplat-in-the-loop. Every paper that trains drone policies in
   splats at scale (GRaD-Nav, ZJU's "Flying in Clutter", GaussGym, SOUS VIDE/FiGS) puts
   gsplat (or nerfstudio, which uses gsplat) beside its own physics.
2. **The physics we need is a black box we have to fit anyway.** The policy outputs DJI
   virtual-stick setpoints. DJI's flight controller closes the loops, so what matters is the
   setpoint-to-motion response: latency, acceleration and braking limits, yaw-rate lag.
   Motor and rotor fidelity (Flightmare, RotorPy, Isaac/PhysX, Genesis) does not help. We
   would still have to tune a velocity controller to imitate DJI's, using our own flight
   logs. A system-identified first-order model with limits is the right fidelity, and it
   is about 100 lines of batched torch.

So the custom sim is small: a torch dynamics module, gsplat batched rendering, an SDF voxel
lookup for collisions, and a BatchEnv API. Isaac Lab + NuRec stays as the fallback if our
renderer falls short (see Risks).

## What the project needs from a sim

| Need | Why it matters here |
|---|---|
| Splat-rendered RGB from the drone camera, batched on GPU | The policy sees VAE latents of camera frames. The scene is a Gaussian splat of the real neighbourhood. |
| Virtual-stick action space (pitch/roll/yaw/throttle as velocity or attitude setpoints) | That is what the MSDK v5 virtual stick accepts. In advanced mode, roll/pitch take ANGLE or VELOCITY. The API caps are 23 m/s horizontal, 6 m/s vertical and yaw max angle 180 ([DJI VirtualStickRange](https://developer.dji.com/api-reference-v5/android-api/Components/IVirtualStickManager/IVirtualStickManager_VirtualStickRange.html), [VirtualStickFlightControlParam](https://developer.dji.com/api-reference-v5/android-api/Components/IVirtualStickManager/Value_FlightController_Struct_VirtualStickFlightControlParam.html)). The Mini 3's real limits are lower and must come from flight logs. |
| Momentum, with braking limited by tilt and acceleration | Stopping from 8 m/s at about 4 m/s² takes about 8 m, plus about 1.6 m per 200 ms of latency. The policy has to learn to brake early. |
| Collisions against a mesh or voxel grid derived from the splat | Splats have no surface, so a separate geometry proxy is required in every system (GaussGym, GRaD-Nav, Isaac NuRec docs). |
| Downward height | Raycast down in the voxel grid, or read the splat depth at the nadir pixel. |
| Fits 2x RTX 3090 (24 GB each) on Linux | Rules out H100-scale assumptions. Isaac Sim's RTX path is memory-hungry. |
| Camera on a 3-axis gimbal | The Mini 3's camera is stabilised, so camera pose = position + yaw + gimbal pitch. Body roll and pitch do not tilt the image, which simplifies rendering. |

## Comparison table: drone simulators

"High-level ctrl" means whether a velocity or attitude setpoint interface like virtual sticks
exists or is easy to add. "Splat cam" means whether a Gaussian splat can be the camera view.

| Sim | Physics fidelity / high-level ctrl | Rendering & batching (reported speed) | Splat cam | Mesh / voxel collision | Maintenance (last default-branch commit; release) | License |
|---|---|---|---|---|---|---|
| **gym-pybullet-drones** | PyBullet rigid body, Crazyflie motor model, drag, ground effect, downwash. Action types include RPM, PID and **VEL** (velocity). | PyBullet TinyRenderer/OpenGL on CPU, one env at a time. Vision is slow. | No | Yes (PyBullet meshes) | Active: 2026-11-06 (now `learnsyslab/gym-pybullet-drones`); last tag v1.0.0, 2021 | MIT |
| **Flightmare** | C++ quad dynamics (motor and body-rate level), vectorised physics env. Inputs are collective thrust + body rates, or motor commands. | Unity renderer, decoupled from physics. The paper reports very fast physics and a few hundred Hz of rendering for a single instance (unverified detail). Not GPU-batched. | No | Unity colliders; point-cloud export | **Stale**: 2023-05-15; last release 0.0.5, 2020 | MIT (Unity engine binary not open) |
| **AirSim** | SimpleFlight FC, with `moveByVelocity` and `moveByRollPitchYawrateThrottle` APIs, which match virtual sticks almost exactly. PX4/ArduPilot SITL. | Unreal Engine 4, about one UE instance per few vehicles, tens of FPS. Not batched. | No (UE4) | Yes (UE colliders) | **Effectively ended**: Microsoft stopped development in 2022. Main still gets housekeeping commits (2026-09-15, CI pinning). Last release v1.8.1, 2022. | MIT |
| **Colosseum** (AirSim successor) | Same as AirSim, on UE 5.6 | Unreal Engine 5. Not batched. | Only through third-party UE5 3DGS plugins (not integrated) | Yes | **Archived** on GitHub. Last commit 2025-12-08 ("Remove Codex Labs as supporter"); v2.3.0, 2025-08 | MIT (inherited) |
| **Aerial Gym** (NTNU ARL) | Isaac Gym PhysX. GPU-parallel Lee geometric controllers: position, **velocity**, **attitude**, acceleration, rate. Arbitrary multirotor configs. | NVIDIA Warp raycasting for depth and segmentation, batched across thousands of envs. RGB only through Isaac Gym's slow per-env camera. | No | Yes (trimesh assets; Warp meshes for raycasting) | Active: v2.0.1, 2026-06-28. Built on **Isaac Gym, which NVIDIA has deprecated** (Python 3.8 era). The README says an Isaac Lab port is "under development". | BSD-3 |
| **Isaac Lab quadcopter** | `Isaac-Quadcopter-Direct-v0` (Isaac Lab 2.x, Crazyflie) takes thrust + body moments, state only. Isaac Lab `main` now also has `isaaclab_contrib` multirotor assets, Lee position/**velocity**/**attitude**/acceleration controllers, and `drone_arl` navigation tasks (the Aerial Gym port, touched 2026-10-07). | RTX tiled camera, batched in one large render. It is fast for depth/raycast and heavy for RGB. The Isaac Lab report notes the raycaster beats the tiled camera at low resolution ([arXiv 2511.04831](https://arxiv.org/pdf/2511.04831)). | **Yes**: NuRec/3DGRUT USDZ splats render in RTX alongside meshes (Isaac Sim 5.0+, 6.0). GaussGym reports about 1 FPS per env at 640x480. | Yes (PhysX mesh/SDF colliders). The splat needs its own collision mesh. | Very active: 2026-11-07 | Isaac Lab BSD-3. Isaac Sim source is open but depends on proprietary Omniverse/RTX components. |
| **OmniDrones** | Isaac Sim 4.1 PhysX. Four control modes (rate, attitude, velocity, position). | Isaac Sim cameras. Slow for RGB at scale. | Not integrated (old Isaac Sim) | Yes | **Abandoned by its author**: the README says it is "hard to maintain and update anymore". Last commit 2026-01-20 (README only). | MIT |
| **RotorPy** | The richest aerodynamics here: rotor drag, induced drag, wind models, motor dynamics. `QuadrotorEnv` supports control from **velocity commands** down to motor speeds. | Batched dynamics on CPU or GPU (torch). The README reports a 25x CPU speedup at more than 1000 drones. v3.0 adds `BatchedPinholeCamera`, which projects coloured 3D feature points (non-photoreal). | No. The third-party `hypersplat`/flysplat 0.0.1 (2026-10) wraps RotorPy + gsplat for a single drone. | Axis-aligned box "world blocks" only | Active: v3.0.0, 2026-09-07 | MIT |
| **Genesis** | Taichi GPU physics with a drone entity driven by propeller RPM. The hover tutorial uses 8192 envs. You write any velocity/attitude controller yourself. | Rasterizer, LuisaRender ray tracer, and `BatchRenderer` (gs-madrona: GPU-batched rasterizer or ray tracer across envs). No published drone-camera FPS. | No | Yes (mesh, convex decomposition / SDF) | Very active: 2026-11-07 (now `genesis-world`) | Apache-2.0 |
| **VisFly** (SJTU) | Differentiable torch dynamics with four controllers. `action_type` is bodyrate, thrust, **velocity** or position. | Modified Habitat-Sim. About 1e4 Hz of 64x64 RGB on an RTX 4090 (README). Batched. | No (mesh `.glb` scenes) | Nearest-obstacle distance from scene geometry | Active: 2026-05-09 | MIT |
| **HyperDrone** (RLtools, [arXiv 2610.28950](https://arxiv.org/abs/2610.28950)) | L2F multirotor dynamics, vectorised on CPU and CUDA. Motor/thrust level, from the learning-to-fly line. | HyperLux ray tracer (OptiX/Vulkan) with RGB, depth, segmentation and flow. It claims "millions of FPS" on consumer GPUs, more than 20x other batch quad sims. | No (mesh ray tracer) | Unclear from the docs | Brand new: 2026-10-31, `pip install hyperdrone` 0.1.10 | MIT (PyPI) |
| **FiGS / SOUS VIDE** (Stanford MSL) | Simple quad dynamics + ACADOS MPC expert, with body rate + thrust commands | gsplat through nerfstudio, single env. Up to about 130 fps (paper). | **Yes** | Not a focus | 2025-07-07 | **GPL-3.0** |
| **GRaD-Nav / GRaD-Nav++** (Stanford) | Differentiable torch quad dynamics: rigid body, PD attitude loop, motor delay, drag. Action = **body rates + normalised thrust**. | 3DGS through nerfstudio/gsplat, **128 drones** batched. 0.07 s per step on a 4090, about 1.8k drone-steps/s. Render is 56% of the step, dynamics 34%, collision 11%. | **Yes** | Point cloud from the 3DGS | 2026-07-07 | No SPDX license detected; check before reuse |
| **GaussGym** (legged, not drones) | Isaac Gym physics | gsplat batched: **more than 100k steps/s across 4096 envs on one RTX 4090** (RGB + depth; render likely decimated relative to physics) | **Yes** | Yes (mesh from the scan, synced with splats) | Code released 2025-10-21, last push 2025-10-30 | "Other" |

Also seen and judged not worth adopting: PX4 SITL with Gazebo, and the Crazyflie Webots/Gazebo
sims ([bitcraze/crazyflie-simulation](https://github.com/bitcraze/crazyflie-simulation)).
These are real-time SITL tools, not batched RL sims. FalconGym
([repo](https://github.com/IllinoisReliableAutonomyGroup/FalconGym)) is a course-oriented
gate-racing setup that moved from NeRF to GSplat with about 5 ms per render. It has no
license and no batching.

### Notes per simulator

- **gym-pybullet-drones.** The easiest install and well maintained, with a real velocity
  action type. Its CPU rendering and Crazyflie-scale physics are the wrong tools for
  camera RL in a splat. It works as a reference for action-type plumbing and little else.
- **Flightmare.** It was the vision-RL drone sim of 2020-2022, but it has had no
  meaningful commits since 2023, needs Unity binaries, and has no splat path. Skip.
- **AirSim / Colosseum.** AirSim has the closest API to virtual sticks
  (`moveByRollPitchYawrateThrottle`), but Unreal rendering is not batched and both projects
  are dead or archived. Getting tens of FPS per instance is about 100x too slow for PPO
  from pixels.
- **Aerial Gym.** It has the right controller set: its GPU Lee velocity and attitude
  controllers are what virtual sticks look like. Its rendering is depth and segmentation
  raycasting, not RGB, and its base, Isaac Gym, is deprecated. The useful parts (the
  controllers and the `drone_arl` tasks) are being ported into Isaac Lab contrib.
- **Isaac Lab (+ NuRec).** The only adoptable option that renders a splat camera, and it is
  under active development. Its costs: per-env RTX splat rendering is very slow compared
  with gsplat (GaussGym's 1 vs 25 FPS per env measurement), Omniverse install and VRAM
  weight on 24 GB cards, and the realism we would pay for (PhysX rotors) is irrelevant
  behind DJI's FC. This is the fallback, not the plan.
- **OmniDrones.** Abandoned by its author and pinned to Isaac Sim 4.1. Skip.
- **RotorPy.** Excellent, maintained aerodynamics with batched dynamics and velocity
  commands. Its camera is a feature-point projector, not a renderer, and its collision
  world is boxes. Worth reading for drag and wind terms if our sysid residuals call for
  them.
- **Genesis.** Fast GPU physics with a batch renderer, but drones are RPM-level, there is
  no splat support, and the API is still moving. It solves problems we do not have.
- **VisFly.** Fast and batched, with a velocity action type, but its renderer is
  Habitat-Sim (meshes). A splat is not a Habitat scene.
- **HyperDrone.** Very new and very fast, but it ray-traces meshes and works at motor level.
  Watch it, do not build on it.
- **GRaD-Nav.** The closest existing code to what we want: torch dynamics, batched splat
  rendering and point-cloud collisions. Its action space is body rate + thrust, it depends
  on nerfstudio, it focuses on differentiable RL, it runs only 128 envs, and its licensing
  is unclear. Borrow ideas from it, not code.

## Splat-in-the-loop work (2024-2026)

| Work | What it does | Renderer / sim | Action space | Scale & speed | Real-world result | Code |
|---|---|---|---|---|---|---|
| **Splat-Nav** ([arXiv 2403.02751](https://arxiv.org/abs/2403.02751)) | Safe planning (polytope corridors from ellipsoid splats) + Splat-Loc pose estimation. **Planning, not RL.** | nerfstudio splats | Trajectory | Replans at 2-5 Hz; localisation about 25 Hz | 100+ hardware flights | [chengine/splatnav](https://github.com/chengine/splatnav), MIT, 2025-04 |
| **SOUS VIDE / FiGS** ([arXiv 2412.16346](https://arxiv.org/abs/2412.16346)) | Distils an MPC expert into a visuomotor policy (SV-Net) trained only in splats from 2-3 min phone videos. Imitation, not RL. | gsplat via nerfstudio + simple dynamics | Body rate + thrust at 20 Hz | Up to about 130 fps | 105 flights, robust to mass, wind and lighting changes | [StanfordMSL/SousVide](https://github.com/StanfordMSL/SousVide), GPL-3.0 |
| **Liquid networks in GS** ([arXiv 2406.15149](https://arxiv.org/abs/2406.15149), CoRL 2024 / PMLR v270) | Imitation learning in a GS + quad-dynamics sim, transferred to real outdoor flights | GS + flight dynamics | Not stated | n/a | Single-scene training transferred to real flights | None found |
| **GRaD-Nav** ([arXiv 2503.03984](https://arxiv.org/abs/2503.03984), IROS 2025) and **GRaD-Nav++** ([arXiv 2506.14009](https://arxiv.org/abs/2506.14009), RA-L 2025) | Differentiable RL in 3DGS; ++ adds a VLA with a MoE head | torch dynamics + 3DGS | Body rate + thrust | 128 envs, about 1.8k steps/s on a 4090 | 6-7/10 real gate passes, zero-shot; ++ reaches 67% real on trained tasks | [Qianzhong-Chen/grad_nav](https://github.com/Qianzhong-Chen/grad_nav) |
| **FalconGym / 2.0** ([arXiv 2503.02198](https://arxiv.org/abs/2503.02198), [2510.02248](https://arxiv.org/abs/2510.02248)) | Gate racing. 2.0 adds an editable-GSplat Edit API and performance-guided refinement. | GSplat | Not stated | Track edits in ms; about 5 ms per render | 69/70 real gates, zero-shot | v1 course repo only |
| **Flying in Clutter** (ZJU, Fei Gao; [arXiv 2512.17349](https://arxiv.org/abs/2512.17349)) | **PPO from monocular RGB in 3DGS**, with an asymmetric critic using privileged depth and adversarial domain adaptation. **The closest match to our setup.** | gsplat (Speedy-Splat pruning) + Isaac Sim for depth and collision | **Normalised thrust + roll/pitch/yaw commands** (attitude-like, close to virtual sticks) | **1024 envs, about 30k FPS peak render at 60x80**, one L40, about 2 days | 80% real success. Failures came from foliage swaying in downwash. | Not stated |
| **Forest UAV nav, relightable 3DGS** ([arXiv 2602.07101](https://arxiv.org/abs/2602.07101)) | End-to-end RL from monocular RGB, relighting for robustness | Relightable 3DGS | Continuous control (unspecified) | n/a | Real forest flights up to 10 m/s | None |
| **GaussFly** ([arXiv 2604.05062](https://arxiv.org/abs/2604.05062)) | Contrastive encoder pretraining, then an RL policy on the latents. **The same latent-policy idea as our VAE.** | 3DGS + geometric constraints | Not stated | n/a | Zero-shot to unseen real scenes | None |
| **ReaDy-Go** ([arXiv 2602.11575](https://arxiv.org/abs/2602.11575), RA-L) | Dynamic 3DGS (moving obstacles), policies from expert planners | 3DGS | Not stated | n/a | Real-world gains, zero-shot | Project page only |
| **GaussGym** ([arXiv 2510.15352](https://arxiv.org/abs/2510.15352)) | Legged RL from pixels | Isaac Gym + batched gsplat | Joint targets | **More than 100k steps/s, 4096 envs, one RTX 4090** | Real stair climbing | [escontra/gauss_gym](https://github.com/escontra/gauss_gym) |
| **GS-Playground** (RSS 2025) | Batched GS rendering + MotrixSim physics | Batched GS | Manipulation and locomotion | 2048 envs at 640x480 (reported aggregate figures vary) | n/a | [gsplayground.github.io](https://gsplayground.github.io/) |
| Manipulation and indoor-nav relatives | SplatSim, RL-GSBridge ([arXiv 2409.20291](https://arxiv.org/abs/2409.20291)), EmbodiedSplat (ICCV 2025, Habitat nav) | Various | n/a | n/a | n/a | Mixed |

Infrastructure worth knowing:

- **gsplat** ([nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat), Apache-2.0,
  active 2026-09). `rasterization()` takes batched `viewmats`/`Ks`, so many cameras render in
  one call, and since v1.5.3 it also takes batches of scenes. It can return depth in the same
  pass. This is the renderer.
- **3DGRUT** ([nv-tlabs/3dgrut](https://github.com/nv-tlabs/3dgrut), Apache-2.0). Ray-traces
  Gaussian particles and exports USDZ for Isaac Sim NuRec. Relevant only on the Isaac
  fallback path.
- **splax** ([learnsyslab/splax](https://github.com/learnsyslab/splax), MIT). A batch-native
  JAX port, interesting only if we ever go JAX.

Takeaways for us:

1. The working recipe is always the same: **gsplat beside cheap custom dynamics, collisions
   against a separate point cloud, mesh or occupancy grid, and low-resolution frames
   (60x80 to 128x128).**
2. Expect around **5-15k rendered frames/s per 3090** at about 96x128 with a pruned splat
   (0.5-1 M Gaussians). This is scaled from the ZJU L40 figure of 30k FPS at 60x80 and
   GaussGym's 4090 numbers, not measured. At a 10 Hz policy rate that is plenty for PPO.
3. The most transferable tricks: privileged-depth critic (ZJU), appearance and lighting
   randomisation (forest relighting), and a pretrained encoder feeding the policy (GaussFly,
   our VAE).

## Recommendation: build a minimal custom sim

Adopting anything above means taking on a large stack (Omniverse, Isaac Gym, Unreal, Unity)
for physics we will replace with a fitted model, while still adding gsplat ourselves,
because only Isaac renders splats and it does so slowly. The custom version is small, every
part of it is either standard (gsplat) or fitted to our own data (dynamics), and it stays
fully on GPU from pose to VAE latent.

### Rust or Python? Python/PyTorch.

[`za-sim`](/home/arelius/Code/za-sim) is a Rust workspace: an engine crate, a PyO3 `zasim`
module exposing a rayon-parallel `BatchEnv`, and a CPU renderer. Its outputs are dict-of-arrays
(`frame [E,...]`, `reward`, `done`, `final_return`) with autoreset. That shape fits well
because there the game logic is the expensive part and is branchy CPU work.

Here the expensive part is rendering, which is CUDA (gsplat), and the frames go straight
into a GPU VAE. The dynamics are about 15 floats per env with no branching, a few
elementwise torch ops for thousands of envs. A Rust core would add a build system and a
CPU/GPU boundary in exchange for speed we do not need. So:

- **Keep za-sim's API pattern**: `BatchEnv(num_envs, seed, **cfg)`, `reset()` / `step(action)`
  returning a dict of batched tensors, autoreset with `final_*` keys, sim-only shaping signals
  kept apart from policy inputs, and a calibration report comparing sim to real logs (like
  `za_calib`).
- **Implement it in PyTorch + gsplat**, with everything as CUDA tensors and no host copies
  in the step loop.
- Rust stays an option for offline tooling only (log parsing, SDF baking) if Python ever
  becomes the bottleneck there. It will not.

### What the custom sim models

- **State per env:** position p (ENU, metres, aligned to the GPS waypoint frame), velocity v,
  yaw ψ, yaw rate r, gimbal pitch, a short action-delay ring buffer, and episode bookkeeping.
- **Action:** the 4 virtual-stick channels in velocity mode: forward and right body
  velocity, vertical velocity, and yaw rate, normalised to [-1, 1] and scaled by Mini 3
  limits. Angle mode (roll/pitch angles) is a config switch. In velocity mode DJI's FC
  brakes for us, but momentum still bites through the acceleration cap and latency.
- **Response:** each channel follows a first-order lag toward the commanded setpoint, with
  time constant τ and separate acceleration and deceleration caps, after a d-step command
  delay. Horizontal acceleration is capped near g·tan(θ_max). Linear drag and per-episode
  wind are added. The parameters (τ, caps, delay, speed limits) are fitted from our DJI Fly
  flight records (10 Hz, already landing in `data/raw/*/logs/`) and later from MSDK telemetry
  step tests. Starting guesses, to be replaced by the fit: horizontal speed of 10 m/s (Normal
  mode), yaw rate of about 100°/s, τ of about 0.3-0.5 s, latency of about 150-250 ms.
- **Camera:** pose = p + yaw + gimbal pitch (stabilised), with Mini 3-like horizontal FOV
  (about 82°), rendered at 96x128 or the VAE's native size. Also models observation latency
  (the video link), motion blur and compression artefacts, and exposure/white-balance jitter.
- **Collision:** an ESDF voxel grid (5-10 cm) baked from the splat. Render depth from the
  training poses, run TSDF fusion to get a mesh, then build the ESDF. Lookup is trilinear in
  torch. A collision is `sdf(p) < r_drone + margin` (r of about 0.15 m). The same SDF gives
  the privileged critic input and a clearance shaping term.
- **Downward height:** a ray march down the ESDF to the first surface, clipped to the sensor
  range, with noise. Masked out when "not available".

### Build plan

| # | Milestone | Deliverable | Exit criterion | Est. |
|---|---|---|---|---|
| M0 | Scene assets | Splat trained with gsplat from the Mini 3 + phone captures (`data/processed`), georeferenced to ENU using the SRT GPS poses. Collision mesh and ESDF in `data/splats/<name>/`. | Waypoint loop overlaid on the mesh looks right. ESDF query unit tests pass. Floaters near the course cleaned up. | 3-5 d |
| M1 | Dynamics + sysid | `dynamics.py` (batched torch) and `calib/` fitting τ, caps and delay from flight logs | Sim-vs-real velocity trace RMSE plotted for held-out flights. Braking distance at 5/8/10 m/s within about 15%. | 3-5 d |
| M2 | Batched renderer | `render.py` wrapping gsplat batched cameras, with pruning and fixed resolution | ≥5k frames/s at the target resolution with 1024 envs on one 3090, and VRAM fits beside the learner | 2-4 d |
| M3 | BatchEnv | `BatchEnv` with za-sim-style dict obs and autoreset. Course = waypoint loop with spawns sampled near obstacles. Reward = forward progress along the path − collision − jerk/yaw-rate penalties. Gymnasium `VectorEnv` adapter. | Random and scripted policies run. Collisions trigger where expected. Throughput is logged. | 2-3 d |
| M4 | VAE + PPO baseline | Image VAE trained on sim + real frames. PPO with the policy on (latent, height) and an asymmetric critic (privileged SDF/velocity). Sim on GPU0, learner on GPU1. | Policy clears the course in sim at a fixed speed without collisions on more than 90% of episodes | 4-7 d |
| M5 | Robustness / DR | Randomised sysid params, latency, wind, appearance (SH jitter, exposure, relighting-lite), video compression. Held-out eval routes. | Eval success holds under full randomisation. The sim-vs-real latent distribution gap is checked on real frames. | 3-5 d |
| M6 | Real bridge | MSDK v5 app path: live frames → encoder → policy → virtual sticks. Shadow mode first (policy predicts while a human or the waypoint mission flies), then a geofenced takeover near one obstacle. | Shadow-mode command agreement is reasonable, then the first supervised takeover | ongoing |

That is roughly 3-4 weeks to a sim-trained baseline (M0-M4). Start M2 and M1 in parallel:
the renderer benchmark is the main risk and should be retired first.

### Risks and fallbacks

- **Render throughput or VRAM falls short.** Prune harder (opacity/size culling, Speedy-Splat
  style), lower the resolution, crop the Gaussians to a radius around the course, or render
  every other physics step. If it is still short, fall back to Isaac Lab + NuRec with the
  `drone_arl` Lee velocity controller and accept lower FPS.
- **The splat-derived collision geometry misses thin things** (branches, wires, gutters). Inflate
  the margin, add hand-placed primitives for known hazards, and fly conservatively.
  ZJU's real failures were foliage, which splats get wrong in both look and motion.
- **The splat is good only near the capture poses.** The policy will see artefacts off the
  capture path. Capture along and around the actual course at flight altitude, randomise
  appearance, and terminate or penalise episodes that wander into poorly reconstructed space
  (track per-voxel view coverage).
- **DJI's response is not first-order** (mode switches, speed-dependent braking, or the
  sport/normal/cine mode). Add a second-order term or a speed-dependent cap only if the M1
  residuals show it.
- **Latency on the real link** (video + virtual-stick round trip) is probably the biggest
  sim-to-real gap for braking. Measure it in M6 and randomise it widely from M3 onward.

## Sources

Primary repos (GitHub API, 2026-11-07):
[gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones),
[Flightmare](https://github.com/uzh-rpg/flightmare),
[AirSim](https://github.com/microsoft/AirSim),
[Colosseum](https://github.com/CodexLabsLLC/Colosseum),
[Aerial Gym](https://github.com/ntnu-arl/aerial_gym_simulator) ([docs](https://ntnu-arl.github.io/aerial_gym_simulator/), [arXiv 2503.01471](https://arxiv.org/abs/2503.01471)),
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) ([tiled rendering](https://isaac-sim.github.io/IsaacLab/main/source/features/tiled_rendering.html)),
[Isaac Sim NuRec docs](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/assets/usd_assets_nurec.html),
[OmniDrones](https://github.com/btx0424/OmniDrones),
[RotorPy](https://github.com/spencerfolk/rotorpy),
[Genesis](https://github.com/Genesis-Embodied-AI/genesis-world) ([hover tutorial](https://genesis-world.readthedocs.io/en/latest/user_guide/policy_training/examples/hover_env.html), [gs-madrona](https://pypi.org/project/gs-madrona/)),
[VisFly](https://github.com/SJTU-ViSYS-team/VisFly) ([arXiv 2407.14783](https://arxiv.org/abs/2407.14783)),
[HyperDrone](https://github.com/rl-tools/hyperdrone),
[hypersplat/flysplat on PyPI](https://pypi.org/project/hypersplat/),
[SousVide](https://github.com/StanfordMSL/SousVide),
[grad_nav](https://github.com/Qianzhong-Chen/grad_nav),
[gauss_gym](https://github.com/escontra/gauss_gym),
[splatnav](https://github.com/chengine/splatnav),
[FalconGym](https://github.com/IllinoisReliableAutonomyGroup/FalconGym),
[gsplat](https://github.com/nerfstudio-project/gsplat),
[3dgrut](https://github.com/nv-tlabs/3dgrut),
[splax](https://github.com/learnsyslab/splax).

Papers: linked inline above. DJI virtual stick:
[VirtualStickRange](https://developer.dji.com/api-reference-v5/android-api/Components/IVirtualStickManager/IVirtualStickManager_VirtualStickRange.html),
[VirtualStickFlightControlParam](https://developer.dji.com/api-reference-v5/android-api/Components/IVirtualStickManager/Value_FlightController_Struct_VirtualStickFlightControlParam.html),
[DJI SDK forum: virtual stick chapter](https://sdk-forum.dji.net/hc/en-us/articles/8697869415577-Chapter-6-The-virtual-stick).
