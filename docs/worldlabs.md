# World Labs (Marble / Atlas beta) for the house splat

Researched 2026-11-07. No API calls were made. Sources: the `@worldlabsai/cli@0.3.0` and
`@worldlabsai/client@0.5.0` npm tarballs (read, never run), the public docs, and `~/Code/cavern-splat`.
Tags: **C** = confirmed (source given), **U** = unknown / not stated anywhere I could find.

## Summary

- **There are two different World Labs APIs.** Don't mix them up.
  1. **Marble World API** (`https://api.worldlabs.ai/marble/v1/worlds:generate`). This is generative:
     you give it one prompt (text, image, up to 8 images, or a video of 100 MB or less) and it builds a
     "world". It exports SPZ/PLY splats, a GLB collider mesh and a panorama, and reports a
     `metric_scale_factor`. Hidden areas "won't match a real floor plan" (C,
     [multi-image guide](https://docs.worldlabs.ai/marble/create/prompt-guides/multi-image-prompt.md)).
  2. **Atlas Developer API (beta)** (`https://api.atlas-beta.worldlabs.ai/api/v2/...`). This is what the CLI
     and our proxy key use. It is a set of tasks, and it includes a **real reconstruction chain**:
     `images2PosedRGBD` (unposed photos to cameras plus depth), then `posedImages2Splats` (SPZ), then
     `splats2Mesh` (GLB). It also has generative tasks (`atlasGenerate`, `atlasMasked`, `atlasChisel`,
     `atlasTextToImage`).
- **Fit:** the Atlas chain is our only World Labs option that is anchored to our footage. Even so, it is
  a learned feed-forward reconstruction. It works at 1280x720 at most, takes 512 frames at most, and its
  refiner uses diffusion clean-up ("render-difix") that can invent content where we didn't capture. The
  docs say plainly: "Atlas can invent uncaptured areas" and "estimated geometry is not a guarantee of
  measured dimensions" (C, [video guide](https://docs.atlas-beta.worldlabs.ai/using-video-footage)). It is
  tuned for **bounded indoor** scenes. Open space, sky, trees and windows degrade (C,
  [splat best practices](https://docs.atlas-beta.worldlabs.ai/splat-best-practices)). An outdoor house is
  outside its sweet spot.
- **Recommendation:** treat our own 3DGS as the ground truth. That means COLMAP/GLOMAP poses with SRT GPS
  priors, then gsplat (already in the cavern-splat venv) at full resolution, with a metric Sim(3) fit to
  the GPS. Run the Atlas chain on the same frames as a cheap second opinion ($0.50 draft, $5 refined),
  and use `splats2Mesh` as one possible collision-mesh source. Use Marble World API worlds only for
  augmentation or distractor scenes, never as the training replica.
- **cavern-splat warning:** it was never a trained 3DGS of real footage. Its dataset is 1 source image
  plus 62 *generated* 1280x720 views with depth. `export_fused.py` just writes the fused depth points out
  as fixed-size isotropic Gaussians. It has nothing for training or meshing. What we can reuse is the
  PLY/SPZ writer, the web viewer and the venv.

## 1. Atlas Developer API: endpoints (from the CLI/SDK source)

Auth is the `WLT-Api-Key: <key>` header (C, CLI `apiKeyAuth`). In the CLI the key comes from the
`WLT_API_KEY` env var or `~/.config/worldlabs/credentials/<sha256(origin)>.key`; `WORLDLABS_API_ORIGIN`
overrides the base URL (C, CLI README). Task submission sends an `Idempotency-Key` header (max 128
chars), and the client retries 429/502/503/504 up to 3 times. A `Retry-After` longer than 10 s rejects
immediately (C, SDK README). Several tasks are marked `@beta` and need entitlement.
`posedImages2Splats` needs "Marble 2 beta access" (C, [skills repo](https://github.com/worldlabsai/skills)).

| Method and path | Purpose |
|---|---|
| `GET /api/v2/health` | health |
| `POST /api/v2/tasks:{task}` (`?wait=true` optional) | submit; returns an Operation |
| `GET /api/v2/operations` (`task`, `status`=RUNNING/SUCCEEDED/FAILED, `createdAfter/Before`, `pageSize`, `pageToken`) | list operations |
| `GET /api/v2/operations/{id}` (`?wait=true`) | get / long-poll |
| `POST /api/v2/operations/{id}:wait` `{timeoutSecs}` (server cap 30 s) | block until done |
| `POST /api/v2/operations/{id}:cancel` `{reason, requestId}` | cancel |
| `GET /api/v2/operations/{id}:trace` | stored request + response |
| `GET /api/v2/operations/{id}/webhookDelivery` | webhook state |
| `GET/POST /api/v2/assets` (list filters: `state, session, applet, mimeType, operation, task, producer, groupBy, order`) | list / create asset |
| `GET/DELETE /api/v2/assets/{id}` | metadata / soft delete |
| `POST /api/v2/assets/{id}:createUploadUrl` `{contentType, contentLength}` returns `{uploadUrl, method, headers, expiresAt}` | signed PUT; upload **without** the API key |
| `POST /api/v2/assets/{id}:completeUpload` `{sizeBytes, checksum, width, height}` | mark READY |
| `POST /api/v2/assets/{id}:createReadUrl` returns `{readUrl, expiresAt}` | download |
| `POST /api/v2/assets/{id}:createUploadUrls` `{roles}` | re-mint part grants |
| `POST /api/v2/assets:createReadPrefix` returns `{baseUrl, signedQuery, expiresAt}` | read the whole library (treat as a secret) |

The **Operation** envelope is `{id, name, ownerId, done, error{code,message,details}, response,
metadata, createdAtSecs, expiresAtSecs}`. Check `error` before reading `response`.
A **TaskAsset** input or output is `{assetId | base64 (<10 MiB) | url}`. Output URLs expire; the
`assetId` does not. Uploads are capped at 1 GiB per part. The CLI maps these MIME types: jpg/png/webp,
`.exr`, `.mp4`, `.glb`, `.ply` (`model/x-ply`), `.spz` (`model/x-spz`). Every task also accepts
`webhookUrl` (signed, retried for 24 h) and `purgeContentOnOperationCompletion`.

**Camera** (`PinholeCamera`) is `{extrinsics: {position[3], quaternion[xyzw], coordinateSystem: "rub"|"rdf"},
intrinsics: {fx, fy, cx, cy, width, height}}`. The pose is **camera-to-world**. RUB is the
OpenGL/Three.js convention (looks along -Z, world Y-up); RDF is OpenCV. There are **no distortion
coefficients**, so undistort first (C,
[cameras guide](https://docs.atlas-beta.worldlabs.ai/cameras-and-posed-images)). Depth comes as a
linear EXR in world units (NaN means no geometry) plus an optional confidence PNG, or as a log-depth PNG
with `zMin/zMax`.

### The reconstruction tasks

- **`images2PosedRGBD`**: request `{frames: [{imageAsset}], canonicalizeToFirstFrame=false,
  targetResolution=[1280,720], refinePoses=false, base64Outputs}`. "A frame that carries a camera, a
  depth buffer or a mask is **rejected**." That means we **cannot pass SRT GPS or our own intrinsics**
  (C, SDK schema). The response is `frames[{imageAsset, camera, depth{depthAsset EXR, confidenceAsset},
  centroidWorld}]` plus `poseRefinementAccepted`. Poses are "good to about half a degree". `refinePoses`
  adds bundle adjustment and needs 2 or more frames. The output is gravity-levelled (C, docs). The grid
  must be one of 1280x720, 720x1280, 960x720 or 528x704. No maximum frame count is stated (U). `?wait`
  times out after 60 s at the load balancer (C, [guide](https://docs.atlas-beta.worldlabs.ai/images-to-posed-images)).
- **`posedImages2Splats`**: request `{posedFrames: [PosedRGBD x 2..512], prompt, enhancePrompt=true,
  refine=true, targetSplatCounts}`. All frames must share one grid from the list above, and depth is
  required on every frame. `refine=false` returns a **draft** fit at 2,000,000 splats (`draftSplat`).
  `refine=true` "places cameras, render-difixes them and trains" against those views and returns
  `refinedSplats{count: asset}`. Output is **SPZ only**. Price: refined 5,000 credits ($5), draft 500
  ($0.50); failed runs are free. Runtime is 10 to 20 min (C,
  [guide](https://docs.atlas-beta.worldlabs.ai/posed-images-to-splats)). The frames could come from our
  own COLMAP poses plus depth, since the schema only needs image + camera + depth EXR (C), but I have not
  tested this.
- **`splats2Mesh`**: request `{splatAsset (.spz), cameras: auto_anchors | manual_anchors{anchors} |
  exterior_sphere | [<=512 cameras, 64..1024 px], orientedBoundingBox{center, orientation, halfExtents},
  targetFaces, textureMode: image_texture|vertex_color|none, textureSize=4096, advancedOptions{renderHw=1024,
  renderHwSecondary=256, normalSource: depth|render, depthSource: render|stereo, resolution=512 (pow2),
  trimQuantile=0}}`. The response is `{mesh: GLB, anchorPositions}`. Coordinates are Three.js
  (+X right, +Y up, -Z forward). "Geometry no camera sees is not meshed" (C, SDK schema). The default
  grid of 512 across the longest side of a ~60 m scene gives cells of about 12 cm. Cost and runtime: U.

### The generative tasks (for completeness)

`atlasGenerate` takes posed RGBD `contextFrames` plus `targetCameras` and returns new views, optionally
with depth. `atlasMasked` does inpainting or edits from image + mask + camera. `atlasChisel` works from a
prompt plus depth layouts, with up to 32 targets at 1280x720 and
`3*context + source + targets <= 128`. `atlasTextToImage` takes a prompt and returns an image, with
optional "metric, single-view" depth. All of them **invent pixels**. cavern-splat was built this way.

### Scale, cost, limits (Atlas)

- **Scale:** "Camera positions establish the rig's scale". Since `images2PosedRGBD` estimates the
  cameras itself, it outputs "world units", and **the docs never say those are metres** (U). Plan on
  fitting a Sim(3) to the SRT GPS / relative altitude ourselves. A pure-rotation capture gives no scale.
- **Credits:** 1,000 credits = $1, and credits don't expire (C,
  [billing](https://docs.atlas-beta.worldlabs.ai/billing)). Only the `posedImages2Splats` price is
  published. `images2PosedRGBD` and `splats2Mesh` prices are U.
- **Rate limits:** per-task concurrency caps exist, but every number is "to be announced" (U,
  [rate limits](https://docs.atlas-beta.worldlabs.ai/rate-limits)).
- **Flow:** submit, store `operation.id`, then poll `:wait` or GET, or use a webhook. Generation takes
  10 to 20 min for splats; other durations are U.

## 2. Marble World API (generative)

- **Inputs:** `world_prompt.type` is `text`, `image` (jpg/png/webp, optional `is_pano`), `multi-image`
  (up to 4 with `azimuth`, or up to 8 with `reconstruct_images`), or `video` (mp4/webm/mov/avi,
  **100 MB max**). It accepts **one** input type per world, and **no cameras, intrinsics or GPS** (C,
  [generate ref](https://docs.worldlabs.ai/api/reference/worlds/generate.md)). In Auto Layout all images
  must share one resolution and aspect (C). Our 4K drone MP4 would need transcoding or trimming to fit
  under 100 MB. There is no way to choose which frames it uses (U / not exposed).
- **Video guidance:** a continuous 180 to 360 degree shot, rotation-dominant, no zoom, no exposure
  change (C, [video prompt](https://docs.worldlabs.ai/marble/create/prompt-guides/video-prompt.md)).
- **Outputs:** `assets.splats.spz_urls` (exports are SPZ/PLY at about 2M and about 500k splats),
  `assets.mesh.collider_mesh_url` (GLB, 100 to 200k triangles), `full_res_mesh_url` (about 1M triangles,
  vertex colour), `hq_mesh_url` (about 600k triangles, textured; up to 1 h; 4 per hour; 3,500 credits),
  `imagery.pano_url` (2560x1280 equirectangular PNG), `thumbnail_url`, `caption` (C,
  [world get](https://docs.worldlabs.ai/api/reference/worlds/get.md),
  [export specs](https://docs.worldlabs.ai/marble/export/specs.md)). There is no depth output (U).
- **Scale:** `semantics_metadata.metric_scale_factor` converts to metres (1.0 means "could not be
  inferred") and `ground_plane_offset` gives the ground height in metres (C). This is an **estimate**,
  not survey-grade.
- **Frame:** exports use "OpenCV convention" (the docs' wording is "+x left, +y down, +z forward"); flip
  Y and Z for OpenGL tools (C, export specs).
- **Cost and limits:** 1,250 credits = $1. Video or multi-image costs 250 (draft), 1,600
  (`marble-1.1`) or 1,600 to 3,100 (`marble-1.1-plus`) (C,
  [pricing](https://docs.worldlabs.ai/api/pricing.md)). Rate limits are about 3 starts/min and 60/h, and
  a world takes about 5 min (C, [rate limits](https://docs.worldlabs.ai/api/rate-limits.md)).
- **Fit:** **generative**. Use it for scene variety or domain randomisation, not as the house replica.

## 3. Fit for the drone RL replica

| Need | Marble World | Atlas chain | Own 3DGS (COLMAP + gsplat) |
|---|---|---|---|
| Faithful to footage | no, generative | mostly; refiner can invent unseen areas | yes, only what was seen |
| Input resolution used | U | 1280x720 max | full 4K / 12 MP |
| Frames | 1 video or 8 images | 2 to 512 | thousands |
| GPS / pose priors | no | no (rejected) | yes (pose priors / georegistration) |
| Metric scale | estimated factor | U (world units) | Sim(3) to SRT GPS, check with tape |
| Collision geometry | collider GLB | `splats2Mesh` GLB | TSDF from rendered depth, or 2DGS/PGSR mesh |
| Outdoor / trees / sky | degrades | degrades (documented) | handles sky with masking; foliage is fuzzy |

**Plan:**

1. Run `tools/ingest.py` to get frames and per-frame SRT poses.
2. Run COLMAP/GLOMAP with GPS priors (convert lat/lon/rel_alt to local ENU), then a Sim(3) fit to ENU.
   This makes the scene metric, with Z-up ENU as our canonical frame.
3. Train with gsplat, then extract a mesh (render depth, then TSDF) and a voxel grid at 0.25 m with
   inflation for collision.
4. In parallel, send about 300 to 500 of the same frames through `images2PosedRGBD` (`refinePoses`),
   then `posedImages2Splats` (draft first), then `splats2Mesh`. Sim(3)-align its camera centres to ours
   (the per-frame correspondence is exact) and compare. Keep it only where it beats ours, for example
   filling sparse areas. Mark any Atlas-only geometry as untrusted for collision.

**Combining:** we can feed our own COLMAP cameras plus depth into `posedImages2Splats`, which converts
camera to world in RUB (OpenCV RDF to RUB means flipping the camera Y and Z axes and converting the
world to Y-up). That way the World Labs splat lands in our metric frame directly. This is untested (U).
Marble worlds can be inserted as background or distractor scenes.

## 4. cavern-splat: what to reuse (read only)

- `dataset/transforms.json` is nerfstudio-style (`OPENCV`, fl 1306.3, 1280x720, c2w matrices, `depth/*.npy`,
  `init_points.ply` with 400k points). There is 1 `is_source_frame` (`base.png`); the rest are views
  named by yaw/pitch (`y45p16`, `l40`, ...) around a nearly fixed centre at y=2.59. So it is a
  **generated** world from a single image, not a capture. This JSON format is a good template for
  our gsplat dataset.
- `export_fused.py` does **not** train. It writes fused points as Gaussians with log-scale ln(0.018),
  opacity logit 2.197, identity rotation, SH0 from RGB, saves them with `plyfile` (3DGS PLY layout), and
  converts to SPZ with the Python `spz` package (`spz.GaussianSplat(...).save`). **Reuse this as our
  PLY-to-SPZ writer.**
- `viewer/index.html` loads `.ply` through three@0.186.0 + `@mkkellogg/gaussian-splats-3d@0.4.7` (a
  100k-splat preview file). We can reuse it to view our splats or the Atlas SPZ.
- `.venv` has torch, `gsplat`, `spz`, `plyfile` and numpy. That's enough to train 3DGS, but there's no
  training script in the repo.
- There is no mesh extraction or collision tooling. We'll have to write that, or use `splats2Mesh`.

## 5. Field checklist (first outing)

The Mini 3 (not Pro) has **no obstacle sensing**, so every pass is flown by line of sight with margins.
Map / GPS accuracy is about +/-2 ft, so don't trust SRT positions below about 0.6 m individually. Scale
comes from averaging many of them, plus a tape check.

**Before flying**
- [ ] Overcast or even light, low wind (foliage moves). Shoot every pass within about 45 min.
- [ ] Walk the site and note the tallest tree, wire and roof height (H). Set RTH altitude above all of them.
- [ ] Camera: 4K/30, **manual exposure locked** (shutter 1/250 or faster, ISO 100 to 400), fixed WB,
      Normal colour, **1x zoom (no digital zoom)**, SRT subtitles **on**.
- [ ] Set 2 or 3 tape-measured references on the ground (for example two cones 10.00 m apart, plus a
      wall length). Photograph them from the drone and the phone.

**Drone passes** (each orbit takes 2 min or more, so about 3 degrees/s; overlap between frames is high)
- [ ] O1, high: radius about 25 to 30 m (clear of trees), altitude about 2H (20 to 25 m), gimbal -40 degrees.
- [ ] O2, mid: radius about 15 to 20 m, altitude about H+3 m (10 to 12 m), gimbal -20 degrees.
- [ ] O3, low (only where clear): radius about 12 to 15 m, altitude 5 to 6 m, gimbal -5 to -10 degrees. Fly it slowly.
- [ ] Nadir/oblique grid over the roof and yard at 30 to 40 m: stills at gimbal -90 degrees and -60
      degrees, about 70% overlap.
- [ ] 12 MP JPEG stills every ~10 degrees on O2 (high-res anchors for COLMAP).
- [ ] Translate, don't just yaw in place. Pure rotation gives no depth or scale.

**Phone passes** (Samsung pro video: lock AE/AF/WB, 4K30 or 1080p60, landscape, 1x lens only)
- [ ] Loop 1 around the house at 3 to 5 m from the walls, chest height, slow walk (~0.5 m/s), camera
      angled about 15 degrees down so the ground shows.
- [ ] Loop 2 at about 10 m, camera level, phone held high.
- [ ] Short "bridge" clips that overlap what the drone saw on O3: entrances, corners, under eaves.
- [ ] Don't pivot on the spot. Keep people and cars out of frame.

**Extraction / limits**
- [ ] `ingest.py --fps 2` gives about 240 frames per 2-min orbit. That's fine for COLMAP (target 500 to
      1,500 total).
- [ ] For the Atlas chain, cap at **512 frames total**, downscaled to 1280x720 (choose by sharpness and
      spacing, about 50% overlap with neighbours).
- [ ] For a Marble World video, cut a clip under 100 MB (transcode to 1080p).
- [ ] Check `df -h ~` on cradle before pushing (about 0.5 to 1 GB/min of 4K).
