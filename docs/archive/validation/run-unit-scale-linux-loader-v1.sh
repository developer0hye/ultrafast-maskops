#!/usr/bin/env bash
# Two sequential full matrices: 60 measured processes total, no pilot subsets.
set -euo pipefail
ROOT=/home/yonghye/ultrafast-vision-build
SOURCE=/home/yonghye/ultrafast-maskops-unit-scale-v1
PYTHON="$ROOT/mask-unit-scale-linux-clean-v1/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 YOLO_OFFLINE=true
cd "$SOURCE"
for MODE in yes no; do
  "$PYTHON" bench/coco_loader.py \
    --corpus "$ROOT/coco-val2017-yolo/segment" \
    --workers 0 2 8 --rounds 5 --batch 8 --imgsz 640 --mask-ratio 4 \
    --overlap "$MODE" \
    --out "$ROOT/mask-unit-scale-linux-loader-overlap-$MODE-v1.json" \
    > "$ROOT/mask-unit-scale-linux-loader-overlap-$MODE-v1.log" 2>&1
done
