#!/usr/bin/env python3
"""One video -> splat job (driven by src/djirl/splat_jobs.py; run as a transient systemd unit under flock,
so jobs queue one at a time).

    .venv/bin/python tools/splat_job.py runs/splat/<job>

<job>/job.json: {"videos": [...], "name": "backyard-v2", "method": "local"|"worldlabs", "telemetry": [...],
                 "fps": null, "steps": 15000, "prompt": null, "device": "cuda:0"}
<job>/status.json: {"state": "queued|running|done|failed", "progress": 0..1, "stage", "message", "scene", ...}

Stages:
  frames     Flight Lab .stream.h264/.h265 -> ingest capture (frames + per-frame GPS/attitude/gimbal from
             telemetry/<stamp>.csv); DJI .MP4 -> ingest video (GPS from SRT/embedded subtitles);
             other video -> plain frames (no GPS). Near-duplicate frames (hovering, on the ground) are dropped.
  local      COLMAP/GLOMAP (tools/sfm.py) -> gsplat (tools/train_splat.py, GPS Sim(3) to ENU metres, levelled)
  worldlabs  <=100 frames at 1280x720 -> Atlas images2PosedRGBD -> posedImages2Splats (refined, ~$5) ->
             SPZ -> Sim(3) from Atlas camera centres to GPS, levelled, cropped
  import     djirl import-ply (collision ESDF) -> data/splats/<name>/ (shows up in /api/sim/scenes)
  phone      ~/Code/dji-planner/tools/export_phone_scene.py <name> (non-fatal)
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))
PY = str(REPO / ".venv" / "bin" / "python")
DJIRL = str(REPO / ".venv" / "bin" / "djirl")
PLANNER = Path.home() / "Code" / "dji-planner"
SRC = REPO / "data" / "splats" / "_src"

# progress bands per stage
BANDS = {"local": {"frames": (0.0, 0.05), "sfm": (0.05, 0.3), "train": (0.3, 0.88), "import": (0.88, 0.97), "phone": (0.97, 1.0)},
         "worldlabs": {"frames": (0.0, 0.05), "pose": (0.05, 0.2), "splat": (0.2, 0.8), "align": (0.8, 0.85),
                       "import": (0.85, 0.97), "phone": (0.97, 1.0)}}


class Job:
    def __init__(self, d: Path):
        self.d = d
        self.spec = json.loads((d / "job.json").read_text())
        self.method = self.spec.get("method", "local")
        self.name = self.spec["name"]
        self.dev = self.spec.get("device") or "cuda:0"
        self.logf = (d / "log.txt").open("a")
        self.warnings: list[str] = []

    def log(self, *a):
        line = " ".join(str(x) for x in a)
        self.logf.write(f"{time.strftime('%H:%M:%S')} {line}\n")
        self.logf.flush()

    def status(self, **kw):
        f = self.d / "status.json"
        st = json.loads(f.read_text()) if f.exists() else {}
        st.update(kw, updated=time.time(), warnings=self.warnings)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1))
        tmp.replace(f)

    def stage(self, name, frac=0.0, msg=None):
        lo, hi = BANDS[self.method][name]
        self.status(stage=name, progress=round(lo + (hi - lo) * min(max(frac, 0.0), 1.0), 3),
                    **({"message": msg} if msg else {}))

    def run(self, cmd, stage, parse=None):
        """run a subprocess, log its output, feed lines to parse(line) -> fraction|None for progress"""
        self.log("$", " ".join(cmd))
        p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail = []
        for line in p.stdout:
            line = line.rstrip()
            if re.match(r"^[IW]\d{8}", line) or "Warning" in line:
                continue
            self.log(line)
            tail = (tail + [line])[-20:]
            if parse:
                fr = parse(line)
                if fr is not None:
                    self.stage(stage, fr)
        if p.wait():
            raise RuntimeError(f"{Path(cmd[1] if cmd[0] == PY else cmd[0]).name} failed: " + " | ".join(tail[-4:]))
        return tail


# ---------------------------------------------------------------------------- frames
def _enu(rows):
    lat0, lon0 = rows[0]["latitude"], rows[0]["longitude"]
    return np.array([[(r["longitude"] - lon0) * 111320 * math.cos(math.radians(lat0)), (r["latitude"] - lat0) * 110574,
                      r.get("rel_alt", 0.0)] for r in rows])


def _dedupe(rows):
    """drop frames that add no new viewpoint (hovering / sitting on the ground): keep a frame when the drone
    moved >= 0.4 m or turned (yaw or gimbal) >= 4 deg since the last kept one"""
    if not rows or "latitude" not in rows[0]:
        return rows
    P = _enu(rows)
    keep, last = [0], 0
    for i in range(1, len(rows)):
        a, b = rows[last], rows[i]
        dyaw = abs((b.get("att_yaw", 0) - a.get("att_yaw", 0) + 180) % 360 - 180)
        dgim = abs(b.get("gimbal_pitch", 0) - a.get("gimbal_pitch", 0))
        if np.linalg.norm(P[i] - P[last]) >= 0.4 or dyaw >= 4 or dgim >= 4:
            keep.append(i)
            last = i
    return [rows[i] for i in keep]


def _resolve_video(v: str) -> Path:
    p = Path(v)
    if not p.is_absolute():
        p = (REPO / p) if (REPO / p).exists() else (PLANNER / p)
    p = p.resolve()
    # a planner playback remux of a Flight Lab stream: use the raw stream, it has the telemetry pairing
    if p.suffix.lower() == ".mp4" and p.stem.startswith("flightlab_"):
        stem = p.stem.split(".")[0]
        raw = sorted((REPO / "data" / "raw").glob(f"*/*/{stem}.stream.h26*"))
        if raw:
            return raw[0]
    if not p.exists():
        raise FileNotFoundError(v)
    return p


def _probe_duration(p: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 60.0


def stage_frames(job: Job):
    import ingest
    vids = [_resolve_video(v) for v in job.spec["videos"]]
    streams = [v for v in vids if ".stream." in v.name]
    # sampling rate: aim for ~500 frames before de-duplication (Flight Lab streams use telemetry time)
    dur = 0.0
    for v in vids:
        if v in streams:
            stamp = v.name.split(".")[0]
            vc = v.parent.parent / "telemetry" / f"{stamp}.video.csv"
            if vc.exists():
                ts = [int(l.split(",")[0]) for l in vc.read_text().splitlines()[1:] if l]
                dur += (ts[-1] - ts[0]) / 1e9 if ts else 0
        else:
            dur += _probe_duration(v)
    fps = float(job.spec.get("fps") or min(5.0, max(1.0, 500.0 / max(dur, 1.0))))
    job.log(f"{len(vids)} video(s), {dur:.0f} s total, sampling {fps:.2f} fps")
    srcs = []
    for k, v in enumerate(vids):
        job.stage("frames", k / len(vids), f"extracting frames from {v.name}")
        out = ingest.out_dir_for(v)
        if v in streams:
            try:
                ingest.do_capture(v, fps, None, float(job.spec.get("latency_ms", 120.0)))
            except FileNotFoundError as e:
                job.warnings.append(f"{v.name}: no telemetry ({Path(str(e)).name}); frames without GPS")
                ingest.extract_frames(_remux(v, job), out, fps, None, None)
        else:
            ingest.do_video(v, fps, 1920)
        rows = [json.loads(l) for l in (out / "frames.jsonl").open()]
        n0 = len(rows)
        rows = _dedupe(rows)
        kind = "drone" if v in streams or "drone" in v.parts else "phone" if "phone" in v.parts else "video"
        tag = kind if not any(s["tag"] == kind for s in srcs) else f"{kind}{k}"
        sd = job.d / "src" / tag
        (sd / "frames").mkdir(parents=True, exist_ok=True)
        for old in (sd / "frames").iterdir():
            old.unlink()
        for r in rows:
            (sd / r["file"]).symlink_to((out / r["file"]).resolve())
        (sd / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        gps = sum("latitude" in r for r in rows)
        job.log(f"{v.name}: {len(rows)}/{n0} frames kept after de-dup, {gps} with GPS -> src/{tag}")
        if not gps:
            job.warnings.append(f"{v.name}: no GPS, so it can't anchor scale or position")
        srcs.append({"tag": tag, "video": str(v), "frames": len(rows), "gps": gps})
    if not any(s["gps"] for s in srcs):
        job.warnings.append("no GPS in any video: the scene will not be in metres or georeferenced")
    if sum(s["frames"] for s in srcs) < 20:
        raise RuntimeError(f"only {sum(s['frames'] for s in srcs)} usable frames; need a longer or moving clip")
    job.status(sources=srcs)
    return srcs


def _remux(v: Path, job: Job) -> Path:
    out = job.d / f"{v.name.split('.')[0]}.mp4"
    fmt = "hevc" if v.suffix.lower() in (".h265", ".hevc") else "h264"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-r", "30", "-f", fmt, "-i", str(v), "-c", "copy", str(out)], check=True)
    return out


# ---------------------------------------------------------------------------- local
def stage_local(job: Job, srcs):
    recon = job.d / "recon"
    args = [a for s in srcs for a in ("--src", f"{s['tag']}={job.d / 'src' / s['tag'] / 'frames'}")]
    sfm_steps = ["features", "sequential matching", "cross/loop matching", "mapping"]
    job.stage("sfm", 0, "structure from motion (COLMAP)")

    def p_sfm(line):
        for i, k in enumerate(sfm_steps):
            if line.startswith(k):
                return (i + 1) / len(sfm_steps)
        m = re.match(r"model 0: (\d+)/(\d+) images", line)
        if m:
            job.status(registered=f"{m.group(1)}/{m.group(2)}")
        return None
    job.run([PY, "-I", "tools/sfm.py", str(recon), *args, "--max-size", "1600", "--cross-every", "4",
             "--mapper", "global"], "sfm", p_sfm)
    if not (recon / "sparse" / "0").exists():
        raise RuntimeError("structure from motion found no model (too little overlap or motion?)")
    steps = int(job.spec.get("steps") or 15000)
    job.stage("train", 0, f"training splat ({steps} steps)")

    def p_train(line):
        m = re.match(r"step (\d+) ", line)
        if m:
            return int(m.group(1)) / steps
        if line.startswith("georef: not enough"):
            job.warnings.append("georef: too few GPS frames registered; scene is not metric")
        m = re.search(r"held-out PSNR over \d+ views: ([\d.]+) dB", line)
        if m:
            job.status(psnr_db=float(m.group(1)))
        m = re.search(r"residual median ([\d.]+) m", line)
        if m:
            job.status(georef_residual_m=float(m.group(1)))
        return None
    job.run([PY, "-I", "tools/train_splat.py", str(recon), "--name", job.name, "--steps", str(steps),
             "--downscale", "1", "--max-side", "1920", "--holdout", "20", "--device", job.dev], "train", p_train)


# ---------------------------------------------------------------------------- worldlabs
def stage_worldlabs(job: Job, srcs):
    from PIL import Image

    from worldlabs import Atlas
    rows = []
    for s in srcs:
        sd = job.d / "src" / s["tag"]
        rows += [dict(json.loads(l), _path=str(sd / json.loads(l)["file"])) for l in (sd / "frames.jsonl").open()]
    k = min(100, len(rows))
    pick = [rows[int(i)] for i in np.linspace(0, len(rows) - 1, k).round()]
    wd = job.d / "wl"
    (wd / "frames").mkdir(parents=True, exist_ok=True)
    files = []
    for i, r in enumerate(pick):
        im = Image.open(r["_path"]).convert("RGB")
        w, h = im.size
        tw, th = (w, round(w * 9 / 16)) if w * 9 / 16 <= h else (round(h * 16 / 9), h)   # centre-crop to 16:9
        im = im.crop(((w - tw) // 2, (h - th) // 2, (w - tw) // 2 + tw, (h - th) // 2 + th)).resize((1280, 720), Image.LANCZOS)
        f = wd / "frames" / f"{i:04d}.webp"
        im.save(f, "WEBP", quality=70)
        files.append(f)
    (wd / "frames.jsonl").write_text("".join(json.dumps({k2: v for k2, v in r.items() if k2 != "_path"}) + "\n" for r in pick))
    atlas = Atlas(wd, log=job.log)
    job.stage("pose", 0, f"World Labs: posing {k} frames")
    posed = atlas.posed_rgbd(files, key=f"djirl-{job.d.name}-i2p", on_poll=lambda s: job.stage("pose", min(s / 300, 0.95)))
    job.stage("splat", 0, "World Labs: generating refined splat (10-20 min)")
    assets = atlas.splats(posed, job.spec.get("prompt") or f"Drone footage of {job.name.replace('-', ' ')}: outdoor scene, "
                          "real buildings, ground, vegetation and sky.", refine=True, key=f"djirl-{job.d.name}-p2s",
                          on_poll=lambda s: job.stage("splat", min(s / 1200, 0.95)))
    name, asset = max(assets, key=lambda a: int(a[0]) if str(a[0]).isdigit() else 0)
    spz = atlas.download(asset, wd / f"splat_{name}.spz")
    job.log(f"downloaded {spz.name} {spz.stat().st_size / 1e6:.1f} MB")
    job.stage("align", 0, "aligning to GPS")
    align_spz(job, spz, posed, pick)


def align_spz(job: Job, spz: Path, posed, rows):
    """SPZ (Atlas world frame) -> _src/<name>.ply + Sim(3) to ENU from camera centres vs GPS, levelled, cropped."""
    import torch

    from djirl.scene import Gaussians, save_ply
    from spzread import read_spz
    from train_splat import level_ground, robust_sim
    S = read_spz(spz)
    q = S["q_xyzw"][:, [3, 0, 1, 2]]
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    C = np.array([f["camera"]["extrinsics"]["position"] for f in posed])
    gi = [i for i, r in enumerate(rows) if "latitude" in r]
    T = np.eye(4)
    geo = None
    if len(gi) >= 5:
        enu = _enu([rows[i] for i in gi])
        s, R, t, err, keep = robust_sim(C[gi], enu)
        T[:3, :3], T[:3, 3] = s * R, t
        res = float(np.median(err[keep]))
        job.status(georef_residual_m=round(res, 2))
        job.log(f"sim3: scale {s:.4f}, {keep.sum()}/{len(gi)} inliers, residual median {res:.2f} m")
        geo = {"lat0": rows[gi[0]]["latitude"], "lon0": rows[gi[0]]["longitude"], "yaw_deg": 0.0, "scale": round(float(s), 5),
               "residual_m": round(res, 3), "gps_frames": int(keep.sum()), "gps_frames_total": len(gi),
               "note": "World Labs Atlas splat; ENU origin = first GPS fix of the job's frames; Sim(3) from Atlas camera centres"}
    else:
        job.warnings.append("too few GPS frames to align: scene left in World Labs units (not metres)")
    T = level_ground(T, S["means"], S["opacity"], (T[:3, :3] @ C.T).T + T[:3, 3])
    PE = (T[:3, :3] @ S["means"].T).T + T[:3, 3]
    camE = (T[:3, :3] @ C.T).T + T[:3, 3]
    lo, hi = camE.min(0) - 40, camE.max(0) + 40
    kc = (PE[:, 0] > lo[0]) & (PE[:, 0] < hi[0]) & (PE[:, 1] > lo[1]) & (PE[:, 1] < hi[1]) & (PE[:, 2] > -5) \
         & (PE[:, 2] < camE[:, 2].max() + 30) & (S["opacity"] > 0.02)
    g = Gaussians(*(torch.from_numpy(np.ascontiguousarray(x[kc])).float() for x in
                    (S["means"], q, np.exp(S["log_scales"]), S["opacity"].clip(1e-6, 1 - 1e-6), S["sh"])))
    SRC.mkdir(parents=True, exist_ok=True)
    save_ply(g, SRC / f"{job.name}.ply")
    (SRC / f"{job.name}.transform.json").write_text(json.dumps(T.tolist()))
    pick = np.linspace(0, len(camE) - 1, 7)[:-1].round().astype(int)
    (SRC / f"{job.name}.waypoints.json").write_text(json.dumps([[round(float(x), 2) for x in camE[i]] for i in pick]))
    if geo:
        (SRC / f"{job.name}.geo.json").write_text(json.dumps(geo, indent=2))
    job.log(f"kept {int(kc.sum())}/{len(kc)} gaussians inside the flown area + 40 m")


# ---------------------------------------------------------------------------- import / phone
def stage_import(job: Job):
    job.stage("import", 0, "building collision map")
    wp = json.loads((SRC / f"{job.name}.waypoints.json").read_text())
    ceiling = max(15.0, max(p[2] for p in wp) + 10.0)
    job.run([DJIRL, "import-ply", str(SRC / f"{job.name}.ply"), "--name", job.name, "--voxel", "0.1",
             "--max-height", f"{ceiling:.0f}", "--min-opacity", "0.5", "--min-neighbors", "4", "--max-scale", "1.0",
             "--transform", str(SRC / f"{job.name}.transform.json"), "--waypoints", str(SRC / f"{job.name}.waypoints.json"),
             "--device", job.dev], "import")
    if not (REPO / "data" / "splats" / job.name / "scene.json").exists():
        raise RuntimeError("import produced no scene")
    exp = PLANNER / "tools" / "export_phone_scene.py"
    if exp.exists():
        job.stage("phone", 0, "exporting the on-phone scene")
        try:
            job.run([PY, str(exp), job.name, "--device", job.dev], "phone")
        except Exception as e:  # noqa: BLE001
            job.warnings.append(f"phone export failed: {e}"[:300])


def main():
    d = Path(sys.argv[1]).resolve()
    job = Job(d)
    job.status(state="running", started=time.time(), message="starting")
    try:
        srcs = stage_frames(job)
        if job.method == "worldlabs":
            stage_worldlabs(job, srcs)
        else:
            stage_local(job, srcs)
        stage_import(job)
        job.status(state="done", progress=1.0, stage="done", scene=job.name, finished=time.time(),
                   message=f"scene {job.name} ready")
    except Exception as e:  # noqa: BLE001
        job.log(traceback.format_exc())
        msg = f"{type(e).__name__}: {e}"
        if type(e).__name__ == "Stop":
            msg = f"World Labs key needs Clark's approval in ac ({e})"
        job.status(state="failed", message=msg[:500], finished=time.time())
        sys.exit(1)


if __name__ == "__main__":
    main()
