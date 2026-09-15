"""Capture the inputs of Format's overlap-mask call from the augmented COCO pipeline.

    python bench/capture_format_inputs.py --images /path/to/coco/segment/images/val2017 \
        --calls 2000 --out coco-format-inputs.npz

Records the float32 (N, 1000, 2) segment arrays exactly as
`polygons2masks_overlap` receives them after mosaic, random perspective and
resampling, together with the image size and mask ratio of each call. The
output is the input of bench/mask_stage.py and of tests/test_sampled.py's
`full-capture` case (MASKOPS_FORMAT_INPUTS), and tests/fixtures/coco-format-inputs.npz
is a 43-call subset of such a capture.

The dataset is the unmodified reference YOLODataset with training defaults
and fixed seeds, so the same command reproduces the same calls. Format skips
images without instances, so `--calls` counts calls, not samples.
"""

import argparse
import copy
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data import augment, dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=2000)
    args = parser.parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    cv2.setNumThreads(0)
    ds = dataset.YOLODataset(
        img_path=args.images,
        imgsz=640,
        batch_size=16,
        augment=True,
        hyp=copy.deepcopy(DEFAULT_CFG),
        task="segment",
        data={"names": dict(enumerate(map(str, range(80))))},
    )
    calls = []
    original = augment.polygons2masks_overlap

    def record(imgsz, segments, downsample_ratio=1):
        # Copy: Format reorders `instances[sorted_idx]` afterwards, and the
        # capture must hold the array as the call saw it.
        calls.append(
            (np.array(segments, dtype=np.float32, copy=True), tuple(int(v) for v in imgsz), int(downsample_ratio))
        )
        return original(imgsz, segments, downsample_ratio)

    # Format looks the function up as a module global at call time.
    augment.polygons2masks_overlap = record
    try:
        index = 0
        while len(calls) < args.calls:
            ds[index % len(ds)]
            index += 1
    finally:
        augment.polygons2masks_overlap = original
    calls = calls[: args.calls]
    counts = np.array([len(seg) for seg, _, _ in calls], dtype=np.int64)
    vertices = np.array([seg.shape[1] for seg, _, _ in calls], dtype=np.int64)
    offsets = np.zeros(len(calls) + 1, dtype=np.int64)
    np.cumsum(counts * vertices, out=offsets[1:])
    np.savez(
        args.out,
        points=np.concatenate([seg.reshape(-1, 2) for seg, _, _ in calls]).astype(np.float32),
        offsets=offsets,
        instances=counts,
        vertices=vertices,
        hw=np.array([hw for _, hw, _ in calls], dtype=np.int64),
        ratio=np.array([r for _, _, r in calls], dtype=np.int64),
        overlap=np.ones(len(calls), dtype=bool),
        source_index=np.arange(len(calls), dtype=np.int64),
    )
    print(f"{len(calls)} calls, {int(counts.sum())} instances, {index} samples -> {args.out}")


if __name__ == "__main__":
    main()
