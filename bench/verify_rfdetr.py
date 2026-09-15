"""Byte parity of the RF-DETR adapter over a whole COCO split, several seeds each.

Every image is drawn through RF-DETR's own dataset and through the accelerated
one under the same seed; images and every target tensor must match exactly.
Also fuzzes the kernel against pycocotools + torchvision on random polygons.

usage: verify_rfdetr.py <images_dir> <instances_json> [--seeds 2] [--fuzz 2000] [--out report.json]
"""

import argparse
import hashlib
import json
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])  # repository root, for tests/
from tests.test_rfdetr import apply_chain, random_ops, random_polygon, reference_masks  # noqa: E402
from ultrafast_maskops import _native, _pycocotools  # noqa: E402
from ultrafast_maskops._chain import index_maps  # noqa: E402


def digest(image, target):
    h = hashlib.sha256(image.numpy().tobytes())
    for key in sorted(target):
        h.update(key.encode())
        h.update(target[key].numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images")
    parser.add_argument("annotations")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--fuzz", type=int, default=2000)
    parser.add_argument("--resolution", type=int, default=560)
    parser.add_argument("--out")
    parser.add_argument("--fuzz-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)

    # Kernel fuzz: larger canvases than the unit test, many chains.
    fused = _pycocotools.fused()
    print(f"pycocotools arithmetic: {_pycocotools.arithmetic()}", flush=True)
    fuzz_start = time.perf_counter()
    fuzz_pixels = 0
    for seed in range(args.fuzz):
        rng = random.Random(10_000 + seed)
        h, w = rng.randint(4, 400), rng.randint(4, 400)
        instances = [[random_polygon(rng, h, w) for _ in range(rng.randint(0, 3))] for _ in range(rng.randint(0, 4))]
        instances = [polys for polys in instances if not polys or len(polys[0]) > 4]
        expected = reference_masks(instances, h, w)
        for _ in range(3):
            ops = random_ops(rng, h, w)
            rows, cols = index_maps(ops, h, w)
            got = _native.rfdetr_masks([[np.asarray(p) for p in polys] for polys in instances], h, w, rows, cols, fused)
            want = apply_chain(expected, ops)
            if got.shape != want.shape or not np.array_equal(got, want):
                print(f"FUZZ MISMATCH seed {seed} ops {ops}")
                sys.exit(1)
            fuzz_pixels += got.size
    print(f"fuzz: {args.fuzz} seeds, {fuzz_pixels / 1e6:.0f} M output pixels, no mismatch, {time.perf_counter() - fuzz_start:.0f} s", flush=True)
    if args.fuzz_only:
        return

    from rfdetr.datasets import coco as rf_coco

    from ultrafast_maskops.rfdetr import accelerate_dataset

    def build():
        transforms = rf_coco.make_coco_transforms_square_div_64("train", args.resolution, multi_scale=True)
        return rf_coco.CocoDetection(args.images, args.annotations, transforms, include_masks=True, remap_category_ids=True)

    reference, accelerated = build(), accelerate_dataset(build())
    n = len(reference)
    mismatches, instances, fast_path = 0, 0, 0
    start = time.perf_counter()
    for seed in range(args.seeds):
        for i in range(n):
            torch.manual_seed(seed * n + i)
            a = reference[i]
            torch.manual_seed(seed * n + i)
            b = accelerated[i]
            instances += int(b[1]["masks"].shape[0])
            if digest(*a) != digest(*b):
                mismatches += 1
                print(f"MISMATCH image index {i} seed {seed}", flush=True)
        print(f"seed {seed}: {n} images checked, {mismatches} mismatches so far, {time.perf_counter() - start:.0f} s", flush=True)
    report = {
        "images": n,
        "seeds": args.seeds,
        "instances": instances,
        "mismatches": mismatches,
        "fuzz_seeds": args.fuzz,
        "fuzz_output_pixels": fuzz_pixels,
        "pycocotools_arithmetic": _pycocotools.arithmetic(),
        "resolution": args.resolution,
    }
    print(json.dumps(report))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
