"""Video -> splat jobs for the viewer API (tools/splat_job.py does the work).

Each job is a transient systemd unit `djirl-splat-<job>` that waits on runs/splat/.lock (flock), so jobs
run one at a time in submission order and show "queued" until their turn.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
JOBS = REPO / "runs" / "splat"
SPLATS = REPO / "data" / "splats"
PLANNER = Path.home() / "Code" / "dji-planner"
ALLOWED = [REPO / "data" / "raw", PLANNER / "videos"]
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,47}$")
VIDEO_EXT = (".h264", ".h265", ".hevc", ".mp4", ".mov", ".m4v")


def _unit(job):
    return f"djirl-splat-{job}"


def _active(job) -> bool:
    r = subprocess.run(["systemctl", "--user", "is-active", _unit(job)], capture_output=True, text=True)
    return r.stdout.strip() in ("active", "activating")


def status(job: str):
    if not re.match(r"^s_[0-9_a-z-]+$", job):
        return None
    d = JOBS / job
    f = d / "status.json"
    if not f.exists():
        return None
    st = json.loads(f.read_text())
    spec = json.loads((d / "job.json").read_text())
    st.update(job=job, name=spec["name"], method=spec["method"], videos=spec["videos"])
    if st.get("state") in ("queued", "running") and not _active(job):
        st.update(state="failed", message="job process exited unexpectedly (stopped or crashed); see log")
    return st


def list_jobs():
    return [s for s in (status(d.name) for d in sorted(JOBS.glob("s_*"), reverse=True)) if s]


def _resolve(v: str) -> Path | None:
    if not isinstance(v, str) or ".." in v:
        return None
    p = Path(v)
    cands = [p] if p.is_absolute() else [REPO / p, PLANNER / p]
    for c in cands:
        c = c.resolve()
        if c.exists() and c.is_file() and any(c.is_relative_to(a.resolve()) for a in ALLOWED) \
                and (c.suffix.lower() in VIDEO_EXT):
            return c
    return None


def start(body: dict):
    videos = body.get("videos") or []
    name = str(body.get("name", "")).strip().lower()
    method = body.get("method", "local")
    if not isinstance(videos, list) or not videos:
        return 400, {"ok": False, "error": "videos: list of paths under data/raw/ or dji-planner/videos/"}
    bad = [v for v in videos if _resolve(v) is None]
    if bad:
        return 400, {"ok": False, "error": f"not found or not allowed: {bad}"}
    if not NAME_RE.match(name):
        return 400, {"ok": False, "error": "name: 2-48 chars, lowercase letters, digits and '-'"}
    if (SPLATS / name).exists() or any(s.get("name") == name and s.get("state") in ("queued", "running") for s in list_jobs()):
        return 409, {"ok": False, "error": f"scene {name!r} already exists or is being built"}
    if method not in ("local", "worldlabs"):
        return 400, {"ok": False, "error": "method: 'local' or 'worldlabs'"}
    job = f"s_{time.strftime('%Y%m%d_%H%M%S')}_{name}"[:64]
    d = JOBS / job
    d.mkdir(parents=True)
    spec = {"videos": [str(_resolve(v).relative_to(REPO)) if _resolve(v).is_relative_to(REPO) else str(_resolve(v)) for v in videos],
            "name": name, "method": method, "telemetry": body.get("telemetry") or [],
            "fps": body.get("fps"), "steps": body.get("steps"), "prompt": body.get("prompt"),
            "latency_ms": body.get("latency_ms", 120.0), "device": "cuda:0", "submitted": time.time()}
    (d / "job.json").write_text(json.dumps(spec, indent=1))
    (d / "status.json").write_text(json.dumps({"state": "queued", "progress": 0.0, "stage": "queued",
                                               "message": "waiting for the GPU", "scene": None, "warnings": []}))
    r = subprocess.run(["systemd-run", "--user", "--unit", _unit(job), "--collect", f"--working-directory={REPO}",
                        "-E", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                        "flock", str(JOBS / ".lock"), str(REPO / ".venv" / "bin" / "python"), "tools/splat_job.py", str(d)],
                       capture_output=True, text=True)
    if r.returncode:
        (d / "status.json").write_text(json.dumps({"state": "failed", "progress": 0.0, "message": r.stderr[-300:]}))
        return 500, {"ok": False, "error": r.stderr[-300:]}
    return 200, {"ok": True, "job": job}


def cancel(job: str):
    st = status(job)
    if st is None:
        return 404, {"ok": False, "error": "unknown job"}
    subprocess.run(["systemctl", "--user", "stop", _unit(job)], capture_output=True)
    f = JOBS / job / "status.json"
    s = json.loads(f.read_text())
    if s.get("state") in ("queued", "running"):
        s.update(state="failed", message="cancelled")
        f.write_text(json.dumps(s))
    return 200, {"ok": True}


def log_tail(job: str, n=60):
    f = JOBS / job / "log.txt"
    return f.read_text().splitlines()[-n:] if re.match(r"^s_[0-9_a-z-]+$", job) and f.exists() else []
