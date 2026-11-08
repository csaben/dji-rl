"""Race training jobs for the viewer API: each job is a transient systemd unit running
`djirl race-train` into runs/race/<job>/ (status.json, ghost.json, policy.pt). One job at a time."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from .race import course_hash

REPO = Path(__file__).resolve().parents[2]
JOBS = REPO / "runs" / "race"
MAX_MIN = 60.0
PLANNER_RACES = REPO.parent / "dji-planner" / "library" / "races"


def _unit(job):
    return f"djirl-race-{job}"


def _active(job) -> bool:
    return subprocess.run(["systemctl", "--user", "is-active", "-q", _unit(job)]).returncode == 0


def status(job: str):
    if not job or "/" in job or ".." in job:
        return None
    f = JOBS / job / "status.json"
    if not (JOBS / job / "course.json").exists():
        return None
    st = json.loads(f.read_text()) if f.exists() else {"status": "starting"}
    if st.get("status") in ("running", "starting") and not _active(job):
        if not f.exists() and time.time() - (JOBS / job / "course.json").stat().st_mtime < 60:
            pass  # unit still spinning up
        else:
            st["status"] = "failed"
            st.setdefault("error", "training process exited (see journalctl --user -u " + _unit(job) + ")")
    st["job"] = job
    st["has_ghost"] = (JOBS / job / "ghost.json").exists()
    return st


def list_jobs():
    out = []
    for d in sorted(JOBS.glob("r_*"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
        st = status(d.name)
        if st:
            c = json.loads((d / "course.json").read_text())
            out.append({"job": d.name, "name": c.get("name"), "course_hash": st.get("course_hash"),
                        "status": st["status"], "best_time_s": st.get("best_time_s")})
    return out


def ghost(job=None, course=None):
    if job:
        if "/" in job or ".." in job:
            return None
        f = JOBS / job / "ghost.json"
        return json.loads(f.read_text()) if f.exists() else None
    if course:
        lib = PLANNER_RACES / f"{course}.json"
        if "/" not in course and ".." not in course and lib.exists():   # a planner library slug
            course = course_hash(json.loads(lib.read_text()))
        best = None
        for f in JOBS.glob("r_*/ghost.json"):
            g = json.loads(f.read_text())
            if g.get("course_hash") == course and (best is None or g["time_s"] < best["time_s"]):
                best = g
        return best
    return None


def start(course, minutes: float):
    if isinstance(course, str):                                        # planner library slug
        lib = PLANNER_RACES / f"{course}.json"
        if "/" in course or ".." in course or not lib.exists():
            return 404, {"ok": False, "error": f"no race {course!r} in the planner library"}
        course = json.loads(lib.read_text())
    scene = course.get("scene", "")
    if not scene or "/" in scene or not (REPO / "data" / "splats" / scene / "esdf.npz").exists():
        return 400, {"ok": False, "error": f"unknown scene {scene!r}"}
    gates = course.get("gates") or []
    if not gates:
        return 400, {"ok": False, "error": "course has no gates"}
    for g in gates:
        for k in ("x", "y", "z", "yaw_deg"):
            float(g[k])
    running = [j for j in list_jobs() if j["status"] in ("running", "starting")]
    if running:
        return 409, {"ok": False, "error": "a race job is already training", "running": running[0]}
    minutes = max(1.0, min(MAX_MIN, minutes))
    h = course_hash(course)
    job = f"r_{time.strftime('%Y%m%d_%H%M%S')}_{h[:6]}"
    d = JOBS / job
    d.mkdir(parents=True)
    (d / "course.json").write_text(json.dumps(course))
    subprocess.run(["systemd-run", "--user", "--unit", _unit(job), "--working-directory", str(REPO),
                    "--setenv", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                    str(REPO / ".venv" / "bin" / "djirl"), "race-train", "--course", str(d / "course.json"),
                    "--out", str(d), "--minutes", str(minutes), "--device", "cuda:1"], check=True,
                   capture_output=True)
    return 200, {"ok": True, "job": job, "course_hash": h, "minutes": minutes,
                 "status_url": f"/api/sim/race/jobs/{job}", "ghost_url": f"/api/sim/race/ghost?job={job}"}


_COURSES: dict = {}


def register(course: dict) -> str:
    for g in course["gates"]:
        for k in ("x", "y", "z", "yaw_deg"):
            float(g[k])
    h = course_hash(course)
    _COURSES[h] = course
    d = JOBS / "courses"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{h}.json").write_text(json.dumps(course))
    return h


def course_by_key(key: str):
    """course hash (registered, or from any job) or planner library slug -> course dict (cached)."""
    if key in _COURSES:
        return _COURSES[key]
    if "/" in key or ".." in key:
        return None
    for f in [JOBS / "courses" / f"{key}.json", PLANNER_RACES / f"{key}.json"]:
        if f.exists():
            c = json.loads(f.read_text())
            if f.parent == PLANNER_RACES:      # slugs can be edited: don't cache by slug
                return c
            _COURSES[key] = c
            return c
    for f in JOBS.glob("r_*/course.json"):
        c = json.loads(f.read_text())
        if course_hash(c) == key:
            _COURSES[key] = c
            return c
    return None
