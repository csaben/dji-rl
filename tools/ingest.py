#!/usr/bin/env python3
"""Ingest a raw capture session: parse DJI .SRT sidecars and extract frames.

Stdlib only (plus ffmpeg/ffprobe on PATH).

    # one SRT -> per-frame JSONL
    python3 tools/ingest.py srt data/raw/2026-11-07/drone/DJI_0001.SRT -o out.jsonl

    # one video -> frames at N fps (+ poses joined from its .SRT if present)
    python3 tools/ingest.py frames data/raw/2026-11-07/drone/DJI_0001.MP4 --fps 2

    # a whole session: every video under drone/ and phone/
    python3 tools/ingest.py session data/raw/2026-11-07 --fps 2 --width 1920

Output goes to data/processed/<session>/<source>/<video-stem>/:
    srt.jsonl     one record per SRT entry (every video frame for DJI)
    frames/       extracted JPEGs, 000001.jpg ...
    frames.jsonl  one record per extracted frame: file, t, plus the nearest SRT pose
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROCESSED = REPO / "data" / "processed"
VIDEO_EXT = {".mp4", ".mov", ".m4v"}

_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
_DATETIME = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[.,]\d+)?")
_BRACKET = re.compile(r"\[([^\]]*)\]")
_PAIR = re.compile(r"([A-Za-z_]+)\s*:\s*([^\s\]]+)")
_FRAMECNT = re.compile(r"FrameCnt\s*:\s*(\d+)", re.I)
_DIFFTIME = re.compile(r"DiffTime\s*:\s*(\d+)\s*ms", re.I)
# DJI Fly app recordings (embedded DJI.Subtitle track, 1 Hz):
#   F/1.7, SS 113.01, ISO 100, EV 0, DZOOM 1.000, GPS (-122.4194, 37.7749, 22), D 25.95m, H 5.50m, H.S 2.51m/s, V.S -0.00m/s
# GPS is (lon, lat, satellite count); values can be "n/a" before a fix.
_FLY_GPS = re.compile(r"GPS\s*\(\s*([^,]+),\s*([^,]+),\s*([^)]+)\)")
_FLY_KV = {"F": "fnum", "SS": "shutter", "ISO": "iso", "EV": "ev", "DZOOM": "dzoom",
           "D": "home_dist", "H": "rel_alt", "H.S": "h_speed", "V.S": "v_speed"}


def _secs(h, m, s, frac) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(frac) / 10 ** len(frac)


def _num(v: str):
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in v else f
    except ValueError:
        return v


def parse_srt(path: Path) -> list[dict]:
    """Parse a DJI SRT into records with t_start/t_end (s) and every key: value field."""
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"<[^>]+>", " ", text)  # drop <font ...> tags
    records = []
    for block in re.split(r"\n\s*\n", text.replace("\r", "")):
        m = _TIME.search(block)
        if not m:
            continue
        g = m.groups()
        rec: dict = {"t_start": _secs(*g[:4]), "t_end": _secs(*g[4:])}
        body = block[m.end():]
        if fm := _FRAMECNT.search(body):
            rec["frame"] = int(fm.group(1))
        if dm := _DIFFTIME.search(body):
            rec["diff_ms"] = int(dm.group(1))
        if dt := _DATETIME.search(body):
            rec["datetime"] = dt.group(0).replace(",", ".")
        for inner in _BRACKET.findall(body):
            for k, v in _PAIR.findall(inner):
                rec[k.lower()] = _num(v)
        if "latitude" not in rec and (gm := _FLY_GPS.search(body)):
            lon, lat, sats = (x.strip() for x in gm.groups())
            if lon != "n/a" and lat != "n/a":
                rec["longitude"], rec["latitude"] = float(lon), float(lat)
            if sats != "n/a":
                rec["gps_sats"] = int(float(sats))
            rec["format"] = "dji_fly"
            for tok in _FLY_GPS.sub("", body).split(","):
                tok = tok.strip()
                if tok.startswith("F/"):
                    k, v = "F", tok[2:]
                elif " " in tok:
                    k, v = tok.split(None, 1)
                else:
                    continue
                if k in _FLY_KV:
                    v = re.sub(r"(m/s|m)$", "", v.strip())
                    if v != "n/a":
                        rec[_FLY_KV[k]] = _num(v)
        records.append(rec)
    return records


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def find_srt(video: Path, out_dir: Path) -> Path | None:
    """Sidecar .SRT next to the video, else the embedded DJI.Subtitle track (DJI Fly recordings)."""
    for ext in (".SRT", ".srt"):
        p = video.with_suffix(ext)
        if p.exists():
            return p
    out_dir.mkdir(parents=True, exist_ok=True)
    emb = out_dir / "embedded.srt"
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-map", "0:s:0", "-f", "srt", str(emb)],
                       capture_output=True)
    return emb if r.returncode == 0 and emb.stat().st_size > 0 else None


def extract_frames(video: Path, out_dir: Path, fps: float, width: int | None,
                   srt: list[dict] | None) -> int:
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.jpg"):
        old.unlink()
    vf = f"fps={fps}"
    if width:
        vf += f",scale={width}:-2"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
         "-vf", vf, "-q:v", "2", str(frames_dir / "%06d.jpg")],
        check=True,
    )
    files = sorted(frames_dir.glob("*.jpg"))
    starts = [r["t_start"] for r in srt] if srt else []
    rows = []
    for i, f in enumerate(files):
        t = i / fps  # fps filter emits frame i at ~i/fps from stream start
        row = {"file": f"frames/{f.name}", "t": round(t, 4)}
        if srt:
            j = max(0, bisect.bisect_right(starts, t) - 1)
            pose = {k: v for k, v in srt[j].items() if k not in ("t_start", "t_end")}
            row.update(pose)
            row["srt_index"] = j
        rows.append(row)
    write_jsonl(out_dir / "frames.jsonl", rows)
    return len(files)


def out_dir_for(video: Path) -> Path:
    """data/raw/<session>/<source>/X.MP4 -> data/processed/<session>/<source>/X/"""
    parts = video.resolve().parts
    try:
        i = parts.index("raw")
        session, source = parts[i + 1], parts[i + 2]
    except (ValueError, IndexError):
        session, source = "misc", video.parent.name
    return PROCESSED / session / source / video.stem


def do_video(video: Path, fps: float, width: int | None) -> None:
    out = out_dir_for(video)
    srt_path = find_srt(video, out)
    srt = parse_srt(srt_path) if srt_path else None
    if srt is not None:
        write_jsonl(out / "srt.jsonl", srt)
    n = extract_frames(video, out, fps, width, srt)
    tag = f"{len(srt)} srt records" if srt is not None else "no srt"
    print(f"{video} -> {out.relative_to(REPO)}: {n} frames, {tag}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("srt", help="parse one .SRT to JSONL")
    s.add_argument("srt", type=Path)
    s.add_argument("-o", "--out", type=Path, help="default: stdout")

    for name in ("frames", "session"):
        p = sub.add_parser(name, help="extract frames from one video" if name == "frames"
                           else "process every video in a raw session dir")
        p.add_argument("path", type=Path)
        p.add_argument("--fps", type=float, default=2.0)
        p.add_argument("--width", type=int, default=None, help="resize width (keeps aspect)")

    a = ap.parse_args()
    if a.cmd == "srt":
        recs = parse_srt(a.srt)
        if a.out:
            write_jsonl(a.out, recs)
            print(f"{len(recs)} records -> {a.out}")
        else:
            for r in recs:
                print(json.dumps(r))
    elif a.cmd == "frames":
        do_video(a.path, a.fps, a.width)
    else:
        videos = sorted(p for src in ("drone", "phone") for p in (a.path / src).rglob("*")
                        if p.suffix.lower() in VIDEO_EXT)
        if not videos:
            print(f"no videos under {a.path}/drone or {a.path}/phone", file=sys.stderr)
            return 1
        for v in videos:
            do_video(v, a.fps, a.width)
    return 0


if __name__ == "__main__":
    sys.exit(main())
