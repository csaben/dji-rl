"""Fit the Flight Lab video latency from a reconstruction: re-time the 50 Hz telemetry at
t_arrival - L for a sweep of L and see which L makes the telemetry agree best with the SfM cameras.

  heading  circular spread of (SfM camera heading in ENU) - (telemetry heading), after removing the
           constant offset; yaw changes fast, so this is the sharpest latency signal
  position median residual of the robust GPS sim3 fit (weak at 4-decimal GPS, useful at full precision)

    .venv/bin/python tools/fit_latency.py --sparse data/recon/<name>/sparse/0 \
        --frames data/processed/<date>/drone/flightlab_<stamp>/frames.jsonl \
        --telemetry data/raw/<date>/telemetry/flightlab_<stamp>.csv --src drone

frames.jsonl must come from `ingest.py capture` (rows carry t_arrival_ns). The telemetry heading is
gimbal_yaw when present (the camera's compass yaw), else att_yaw; DJI yaw is compass degrees
(0 = north, clockwise), ENU heading = 90 - yaw.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def load_telemetry(csv: Path):
    from ingest import _col, _read_csv
    h, rows = _read_csv(csv)
    c = {k: _col(h, *v) for k, v in {"t": ("t_ns",), "lat": ("lat", "latitude"), "lon": ("lon", "longitude", "lng"),
                                       "alt": ("alt_rel", "rel_alt", "relative_altitude"),
                                       "gyaw": ("gimbal_yaw",), "ayaw": ("att_yaw", "yaw")}.items()}
    yk = "gyaw" if c["gyaw"] is not None else "ayaw"
    out = {k: [] for k in ("t", "lat", "lon", "alt", "yaw")}
    for r in rows:
        try:
            vals = [float(r[c[k]]) for k in ("t", "lat", "lon", "alt", yk)]
        except (ValueError, IndexError, TypeError):
            continue
        if all(v == v for v in vals):
            for k, v in zip(out, vals):
                out[k].append(v)
    out = {k: np.array(v) for k, v in out.items()}
    out["yaw"] = np.degrees(np.unwrap(np.radians(out["yaw"])))
    return out, yk


def sweep(centers, fwd, t_arr, tel, lats=np.arange(0, 401, 10)):
    """centers/fwd [N,3] SfM camera centre and optical axis; t_arr [N] ns; tel from load_telemetry.
    -> list of {latency_ms, heading_spread_deg, pos_residual_m, n}."""
    from train_splat import robust_sim
    lat0, lon0 = tel["lat"][0], tel["lon"][0]
    res = []
    for L in lats:
        t = t_arr - L * 1e6
        ok = (t > tel["t"][0]) & (t < tel["t"][-1])
        if ok.sum() < 10:
            continue
        g = lambda k: np.interp(t[ok], tel["t"], tel[k])
        enu = np.stack([(g("lon") - lon0) * 111320 * math.cos(math.radians(lat0)), (g("lat") - lat0) * 110574, g("alt")], -1)
        s, R, tr, err, keep = robust_sim(centers[ok], enu)
        f = (R @ fwd[ok].T).T
        cam_h = np.arctan2(f[:, 1], f[:, 0])
        tel_h = np.radians(90 - g("yaw"))
        flat = np.hypot(f[:, 0], f[:, 1]) > 0.3          # skip near-nadir views (heading undefined)
        d = np.angle(np.exp(1j * (cam_h - tel_h)))[flat]
        d = np.angle(np.exp(1j * (d - np.angle(np.exp(1j * d).mean()))))
        res.append({"latency_ms": int(L), "heading_spread_deg": round(float(np.degrees(np.sqrt((d ** 2).mean()))), 3),
                    "pos_residual_m": round(float(np.median(err[keep])), 3), "n": int(ok.sum())})
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sparse", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--telemetry", type=Path, required=True)
    ap.add_argument("--src", default="drone", help="image name prefix of this capture in the reconstruction")
    a = ap.parse_args()
    import pycolmap
    rows = {Path(r["file"]).name: r for r in map(json.loads, a.frames.open()) if "t_arrival_ns" in r}
    if not rows:
        sys.exit(f"{a.frames}: no t_arrival_ns (re-run ingest.py capture)")
    rec = pycolmap.Reconstruction(a.sparse)
    C, Fw, T = [], [], []
    for img in rec.images.values():
        p = Path(img.name)
        if p.parts[0] != a.src or p.name not in rows:
            continue
        X = img.cam_from_world() if callable(getattr(img, "cam_from_world", None)) else img.cam_from_world
        M = X.matrix()
        R, t = M[:3, :3], M[:3, 3]
        C.append(-R.T @ t)
        Fw.append(R.T @ np.array([0, 0, 1.0]))
        T.append(rows[p.name]["t_arrival_ns"])
    if len(C) < 10:
        sys.exit(f"only {len(C)} registered '{a.src}/' images matched {a.frames}")
    tel, yk = load_telemetry(a.telemetry)
    res = sweep(np.array(C), np.array(Fw), np.array(T, dtype=np.float64), tel)
    for r in res:
        print(json.dumps(r))
    bh = min(res, key=lambda r: r["heading_spread_deg"])
    bp = min(res, key=lambda r: r["pos_residual_m"])
    print(f"best by heading ({yk}): {bh['latency_ms']} ms ({bh['heading_spread_deg']} deg); "
          f"best by position: {bp['latency_ms']} ms ({bp['pos_residual_m']} m); {len(C)} frames")


if __name__ == "__main__":
    main()
