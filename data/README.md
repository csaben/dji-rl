# data/

```
data/
  raw/<YYYY-MM-DD>/          one capture session, pushed from the Mac/phone as-is
    drone/                   SD card: DJI_*.MP4 + matching DJI_*.SRT, DJI_*.JPG stills
    phone/                   Samsung handheld videos/photos (ground level)
    logs/                    DJI Fly flight records (.txt, ~10 Hz GPS/attitude)
    telemetry/               our MSDK app's per-frame CSV/JSONL (later)
    NOTES.md                 optional: what was flown, weather, time, anything odd
  processed/<YYYY-MM-DD>/<source>/<video-stem>/
    srt.jsonl                every SRT entry (per video frame): t_start, t_end, frame,
                             datetime, latitude, longitude, rel_alt, abs_alt, iso, shutter, ...
    frames/000001.jpg ...    frames extracted at --fps
    frames.jsonl             one row per extracted frame: file, t, + nearest SRT pose
  splats/<name>/             splat outputs (.ply/.spz), collision mesh/voxels, alignment notes
```

Keep the file names from the SD card so each video stays paired with its `.SRT`.
`raw/` is treated as read-only; everything in `processed/` can be regenerated:

```
python3 tools/ingest.py session data/raw/2026-11-07 --fps 2 --width 1920
python3 tools/ingest.py srt data/raw/2026-11-07/drone/DJI_0001.SRT -o /tmp/x.jsonl
```

Disk: cradle's / is ~98% full (about 47 GB free on 2026-11-07). 4K drone video runs
about 0.5-1 GB/min, so check `df -h ~` before a big push.
