#!/usr/bin/env python3
"""Structure-from-motion for a capture: frames -> camera poses + sparse points (COLMAP via pycolmap).

    .venv/bin/python tools/sfm.py data/recon/poc \
        --src drone=data/processed/2026-11-07/drone/<stem>/frames [--src phone=...] [--every 1]

Creates <out>/images/<name>/ (symlinks), <out>/database.db, <out>/sparse/<k>/ (one model per
connected component; 0 is the largest). One camera per source folder. Sequential matching with
loop detection off (video), plus exhaustive matching between sources when there are several."""
import argparse
import shutil
import time
from pathlib import Path

import pycolmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--src", action="append", required=True, help="name=frames_dir")
    ap.add_argument("--every", type=int, default=1, help="keep every Nth frame")
    ap.add_argument("--max-size", type=int, default=1600, help="max image side for SIFT")
    ap.add_argument("--max-features", type=int, default=8192)
    ap.add_argument("--overlap", type=int, default=15)
    ap.add_argument("--camera-model", default="SIMPLE_RADIAL")
    ap.add_argument("--cross-every", type=int, default=4, help="cross-source/loop pairs use every Nth frame")
    ap.add_argument("--mapper", choices=["global", "incremental"], default="global")
    a = ap.parse_args()

    out = a.out
    if out.exists():
        shutil.rmtree(out)
    img_root = out / "images"
    names = []
    for spec in a.src:
        name, d = spec.split("=", 1)
        (img_root / name).mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(sorted(Path(d).glob("*.jpg"))):
            if i % a.every:
                continue
            (img_root / name / f.name).symlink_to(f.resolve())
            names.append(f"{name}/{f.name}")
    print(f"{len(names)} images from {len(a.src)} source(s)", flush=True)
    db = out / "database.db"

    t = time.time()
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = a.max_size
    ext.sift.max_num_features = a.max_features
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = a.camera_model
    pycolmap.extract_features(db, img_root, image_names=names, camera_mode=pycolmap.CameraMode.PER_FOLDER,
                              reader_options=reader, extraction_options=ext)
    print(f"features: {time.time() - t:.0f}s", flush=True)

    t = time.time()
    seq = pycolmap.SequentialPairingOptions()
    seq.overlap = a.overlap
    seq.quadratic_overlap = True
    pycolmap.match_sequential(db, pairing_options=seq)
    print(f"sequential matching: {time.time() - t:.0f}s", flush=True)
    # extra pairs on a thinned subset: every source against every other (drone<->phone), plus
    # long-range pairs within a source for loop closure (the drone orbits back over itself)
    t = time.time()
    by_src = {}
    for n in names:
        by_src.setdefault(n.split("/")[0], []).append(n)
    thin = {k: v[::a.cross_every] for k, v in by_src.items()}
    pairs = set()
    ks = list(thin)
    for i, ka in enumerate(ks):
        for kb in ks[i:]:
            for x in thin[ka]:
                for y in thin[kb]:
                    if x < y and (ka != kb or abs(by_src[ka].index(x) - by_src[ka].index(y)) > a.overlap):
                        pairs.add((x, y))
    pf = out / "pairs.txt"
    pf.write_text("".join(f"{x} {y}\n" for x, y in sorted(pairs)))
    imp = pycolmap.ImportedPairingOptions()
    imp.match_list_path = pf
    pycolmap.match_image_pairs(db, pairing_options=imp)
    print(f"cross/loop matching ({len(pairs)} pairs): {time.time() - t:.0f}s", flush=True)

    t = time.time()
    sparse = out / "sparse"
    sparse.mkdir(exist_ok=True)
    if a.mapper == "global":
        maps = pycolmap.global_mapping(db, img_root, sparse)
    else:
        maps = pycolmap.incremental_mapping(db, img_root, sparse)
    print(f"mapping: {time.time() - t:.0f}s", flush=True)
    for k, m in sorted(maps.items(), key=lambda kv: -kv[1].num_reg_images()):
        print(f"model {k}: {m.num_reg_images()}/{len(names)} images, {m.num_points3D()} points, "
              f"reproj err {m.compute_mean_reprojection_error():.2f}px")


if __name__ == "__main__":
    main()
