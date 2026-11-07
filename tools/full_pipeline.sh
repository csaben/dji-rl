#!/usr/bin/env bash
# Frames -> SfM (drone + phone, GLOMAP) -> full-res splat -> scene -> viewer.
# Run detached:  systemd-run --user --unit=djirl-full --collect tools/full_pipeline.sh house-full
set -euo pipefail
NAME=${1:-house-full}
SESSION=${SESSION:-2026-11-07}
STEPS=${STEPS:-30000}
cd "$(dirname "$0")/.."
LOG=runs/$NAME.log
mkdir -p runs
exec > >(tee -a "$LOG") 2>&1
echo "== $(date -Is) start $NAME"
DRONE=$(ls -d data/processed/$SESSION/drone/*/frames | head -1)
PHONE=$(ls -d data/processed/$SESSION/phone/*/frames | head -1)

echo "== $(date -Is) sfm"
.venv/bin/python -I tools/sfm.py data/recon/$NAME --src drone=$DRONE --src phone=$PHONE \
    --max-size 1600 --cross-every 4 --mapper global 2>&1 | grep -vE '^[IW][0-9]{8}' || true
test -d data/recon/$NAME/sparse/0

echo "== $(date -Is) train"
.venv/bin/python -I tools/train_splat.py data/recon/$NAME --name $NAME --steps $STEPS \
    --downscale 1 --max-side 1920 --holdout 20 --device cuda:0 2>&1 | grep -vE '^[IW][0-9]{8}|Warning'

echo "== $(date -Is) import"
S=data/splats/_src
.venv/bin/djirl import-ply $S/$NAME.ply --name $NAME --voxel 0.1 --max-height 15 \
    --min-opacity 0.5 --min-neighbors 4 --max-scale 1.0 \
    --transform $S/$NAME.transform.json --waypoints $S/$NAME.waypoints.json --device cuda:0
.venv/bin/djirl preview --scene $NAME --device cuda:0

echo "== $(date -Is) viewer"
UNIT=~/.config/systemd/user/dji-rl-viewer-poc.service
sed -i "s/--scene [^ ]*/--scene $NAME/" "$UNIT"
systemctl --user daemon-reload
systemctl --user restart dji-rl-viewer-poc
echo "== $(date -Is) done"
