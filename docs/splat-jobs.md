# Video -> splat jobs (viewer API)

The viewers on cradle (`dji-rl-viewer-poc` :8967, `dji-rl-viewer` :8965) take recorded videos and build a new
flyable scene. Jobs run one at a time, in order (`flock runs/splat/.lock`); each is a transient systemd unit
`djirl-splat-<job>` writing `runs/splat/<job>/{job.json,status.json,log.txt}`.

## Endpoints

```
POST /api/sim/splat/jobs
  {"videos": ["data/raw/2026-11-08/drone/flightlab_20261108_151952.stream.h264", ...],
   "name": "backyard-v2",                 # 2-48 chars [a-z0-9-], must not be an existing scene
   "method": "local" | "worldlabs",       # default local
   "telemetry": [],                       # optional, ignored for now: found automatically (see below)
   "fps": null, "steps": null,            # optional overrides (sampling fps, local training steps, default 15000)
   "prompt": null}                        # optional scene description for worldlabs
  -> 200 {"ok": true, "job": "s_20261108_172059_backyard-v2"}
     400 bad input / 409 name taken

GET  /api/sim/splat/jobs                  -> [status, ...] newest first
GET  /api/sim/splat/jobs/<job>[?log=60]   -> status (+ last N log lines)
POST /api/sim/splat/jobs/<job>/cancel     -> {"ok": true}
```

Status:

```json
{"job": "...", "name": "backyard-v2", "method": "local", "state": "queued|running|done|failed",
 "progress": 0.42, "stage": "queued|frames|sfm|train|pose|splat|align|import|phone|done",
 "message": "training splat (15000 steps)", "scene": null, "warnings": [],
 "sources": [{"tag": "drone", "video": "...", "frames": 139, "gps": 139}],
 "registered": "137/139", "psnr_db": 24.1, "georef_residual_m": 1.3}
```

When `state` is `done`, `scene` is set and the scene is listed in `GET /api/sim/scenes` (with `geo`), and the
on-phone export (`~/Code/dji-planner/tools/export_phone_scene.py <scene>`) has already been run (a failure
there is only a warning).

Video paths are relative to `~/Code/dji-rl` or `~/Code/dji-planner`, and must be under `data/raw/` or
`dji-planner/videos/`.

## What happens

- **Flight Lab streams** (`*.stream.h264/.h265`): frames are paired with `telemetry/<stamp>.video.csv` rows
  (the first undecodable rows before the first keyframe are skipped) and get GPS, altitude, attitude and
  gimbal interpolated from `telemetry/<stamp>.csv` at arrival time - 120 ms. A planner remux
  `videos/cache/flightlab_*.mp4` is mapped back to its raw stream automatically.
- **DJI .MP4**: GPS from the `.SRT` sidecar or the embedded DJI Fly subtitle track. **Phone / other video**:
  frames only, no GPS (fine as an extra source; at least one source needs GPS for a metric, georeferenced scene).
- Sampling aims for about 500 frames in total (1-5 fps), then drops near-duplicates (moved < 0.4 m and
  turned < 4 deg since the last kept frame: hovering, sitting on the ground).
- **local** (default; free, about 15-40 min): COLMAP/GLOMAP (`tools/sfm.py`) -> gsplat (`tools/train_splat.py`),
  Sim(3) to GPS ENU metres, ground levelled, cropped to the flown area + 30 m. Only what the camera saw.
- **worldlabs** (about $5 per job through the ac proxy, 15-30 min): 100 evenly spaced frames at 1280x720 ->
  Atlas `images2PosedRGBD` -> `posedImages2Splats` (refined) -> Sim(3) from the Atlas cameras to GPS. Looks
  cleaner from novel views, but it can invent unseen areas. If the proxy needs Clark's approval, the job
  fails with that message.
- Then `djirl import-ply` builds the collision ESDF into `data/splats/<name>/`.

## Capture tips

The live stream is only 1280x720. For a sharper scene, also record to the drone's SD card (4K + SRT) and
submit that MP4 alongside. Fly slow orbits and translate (pure yaw in place gives no depth). Keep the gimbal
between -10 and -40 deg. Several passes at different heights help a lot.
