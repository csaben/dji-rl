# dji-rl

A DJI Mini 3 flies a looped course of GPS waypoints around the house. Near obstacles, a camera
policy takes over. The policy trains with RL in a Gaussian-splat replica of the neighbourhood,
then flies the real one.

## Layout

```
data/            raw captures -> processed frames -> splats (see data/README.md)
docs/            worldlabs.md (World Labs API findings), sim-survey.md (why a custom gsplat sim)
src/djirl/       the sim + trainer
  scene.py       splat PLY io, ESDF collision grid (baked from the splat), scene dirs
  dynamics.py    batched virtual-stick response model (velocity mode; params are guesses until fitted to logs)
  render.py      batched gsplat camera
  encoder.py     frozen TAESD encoder: 128x96 frame -> 4x12x16 latent
  env.py         BatchEnv: N drones on GPU, waypoint-loop reward, collisions, autoreset
  ppo.py         PPO, actor on (latents, velocity, goal dir, height), critic also gets SDF probes
  viewer.py      browser flight viewer (same camera, dynamics, collisions)
  synthetic.py   procedural stand-in scene
tools/ingest.py              SRT parse + frame extraction for raw sessions
tools/pointcloud_to_splat.py coloured point cloud -> splat PLY (used for the MatrixCity sample)
```

## Use

```
uv sync                                    # Python 3.10, torch 2.4.1+cu124, prebuilt gsplat wheel
.venv/bin/djirl synth                      # data/splats/synthetic
.venv/bin/djirl bench --envs 1024          # ~5k env steps/s on one 3090 (render + encode + step)
.venv/bin/djirl train --scene matrixcity-block --envs 1024 --minutes 90 --name city1
.venv/bin/djirl rollout --scene matrixcity-block --ckpt runs/city1/policy.pt --out city1.mp4
.venv/bin/djirl import-ply splat.ply --name house --waypoints wps.json [--transform T.json]
```

TensorBoard: `.venv/bin/tensorboard --logdir runs --bind_all`.

## Viewer

`systemctl --user {status,restart} dji-rl-viewer` serves http://cradle.tailb1b51d.ts.net:8965
on GPU 1. You fly with WASD/QE/Space/Shift or a gamepad (Mode 2). Switch between HD and
"policy input" (the exact 128x96 frame the policy gets), the scene, and the pilot: manual,
random, or any `runs/*/policy.pt`.

## Scenes

- `synthetic`: procedural house, garage, trees and fence. 87k Gaussians.
- `matrixcity-block`: a 120x120 m slice of MatrixCity (a synthetic Unreal city; research
  dataset, `AIBluefihser/MatrixCity` on HF) converted from a dense coloured point cloud. These are
  isotropic Gaussians, not a trained splat, so it looks blobbier than a real capture will.

## Not done yet

- Reconstruction of real footage (COLMAP/GLOMAP + gsplat training, georeferenced with the SRT GPS):
  waiting on the first capture. COLMAP is not installed (`sudo apt install colmap`).
- Dynamics fit from flight logs: waiting on Mini 3 telemetry (step inputs, hard stops, yaw steps).
- Multi-GPU training: the env and learner both run on one device for now.
