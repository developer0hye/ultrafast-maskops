"""RF-DETR segmentation data loading: RF-DETR's own path vs the maskops adapter.

Builds two RF-DETR CocoDetection datasets (include_masks=True) with the default
training transforms (square_resize_div_64, multi_scale), checks that both give
byte-identical samples under the same seed, then times __getitem__ in one
process (with the mask work split out) and an N-worker DataLoader.

usage: rfdetr_loader.py <images_dir> <instances_json> [--samples 400]
                        [--resolution 560] [--workers 8] [--out results.json]
"""

import argparse
import functools
import hashlib
import json
import platform
import sys
import time

import torch
from rfdetr.datasets import coco as rf_coco
from rfdetr.utilities.tensors import collate_fn
from torch.utils.data import DataLoader

from ultrafast_maskops.rfdetr import MaterializeMasks, accelerate_dataset


def build(args):
    transforms = rf_coco.make_coco_transforms_square_div_64("train", args.resolution, multi_scale=True)
    return rf_coco.CocoDetection(args.images, args.annotations, transforms, include_masks=True, remap_category_ids=True)


def digest(image, target):
    h = hashlib.sha256(image.numpy().tobytes())
    for key in sorted(target):
        h.update(key.encode())
        h.update(target[key].numpy().tobytes())
    return h.hexdigest()


def timed(dataset, indices, seed_base=0):
    """Mean ms per sample over indices, each drawn under a fixed seed."""
    start = time.perf_counter()
    for i in indices:
        torch.manual_seed(seed_base + i)
        dataset[i]
    return 1e3 * (time.perf_counter() - start) / len(indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images")
    parser.add_argument("annotations")
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--resolution", type=int, default=560)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out")
    args = parser.parse_args()
    torch.set_num_threads(1)

    reference, accelerated = build(args), accelerate_dataset(build(args))
    n = min(args.samples, len(reference))
    print(f"dataset {len(reference)} images, resolution {args.resolution}, {n} samples", flush=True)

    # Parity: same seed, same bytes.
    mismatches = 0
    instances = 0
    for i in range(n):
        torch.manual_seed(i)
        a = reference[i]
        torch.manual_seed(i)
        b = accelerated[i]
        instances += int(b[1]["masks"].shape[0])
        if digest(*a) != digest(*b):
            mismatches += 1
    print(f"parity: {n} samples, {instances} instances, {mismatches} mismatches", flush=True)

    # Stage split: prepare and transforms for both, materialize for the adapter.
    stages = {"reference": {"prepare": 0.0, "transforms": 0.0}, "accelerated": {"prepare": 0.0, "transforms": 0.0, "materialize": 0.0}}

    def wrap(store, name, fn):
        @functools.wraps(fn)
        def inner(*a, **k):
            s = time.perf_counter()
            r = fn(*a, **k)
            store[name] += time.perf_counter() - s
            return r

        return inner

    originals = {}
    for name, dataset in (("reference", reference), ("accelerated", accelerated)):
        originals[name] = (dataset.prepare, dataset._transforms)
        if name == "accelerated":
            assert isinstance(dataset._transforms.transforms[-1], MaterializeMasks)
            dataset._transforms.transforms[-1] = wrap(stages[name], "materialize", dataset._transforms.transforms[-1])
        dataset.prepare = wrap(stages[name], "prepare", dataset.prepare)
        dataset._transforms = wrap(stages[name], "transforms", dataset._transforms)

    indices = list(range(n))
    results = {"reference_ms": [], "accelerated_ms": []}
    for _ in range(2):  # warm the page cache and allocators
        timed(reference, indices[:40]), timed(accelerated, indices[:40])
    for store in stages.values():
        for k in store:
            store[k] = 0.0
    for round_ in range(args.rounds):
        results["reference_ms"].append(timed(reference, indices, 1000 * round_))
        results["accelerated_ms"].append(timed(accelerated, indices, 1000 * round_))
        print(f"round {round_}: reference {results['reference_ms'][-1]:.2f} ms, accelerated {results['accelerated_ms'][-1]:.2f} ms", flush=True)
    per = n * args.rounds
    ref_stages = {k: 1e3 * v / per for k, v in stages["reference"].items()}
    acc = stages["accelerated"]
    acc_stages = {"prepare": 1e3 * acc["prepare"] / per, "transforms": 1e3 * (acc["transforms"] - acc["materialize"]) / per, "materialize": 1e3 * acc["materialize"] / per}
    for name, dataset in (("reference", reference), ("accelerated", accelerated)):
        dataset.prepare, dataset._transforms = originals[name]

    ref_ms = min(results["reference_ms"])
    acc_ms = min(results["accelerated_ms"])
    print(f"\nsingle process, best of {args.rounds}: reference {ref_ms:.2f} ms/sample, accelerated {acc_ms:.2f} ms/sample, {ref_ms / acc_ms:.2f}x")
    print(f"  reference   prepare {ref_stages['prepare']:6.2f} ms  transforms {ref_stages['transforms']:6.2f} ms")
    print(f"  accelerated prepare {acc_stages['prepare']:6.2f} ms  transforms {acc_stages['transforms']:6.2f} ms  materialize {acc_stages['materialize']:6.2f} ms")

    # DataLoader throughput.
    loader_results = {}
    for name, dataset in (("reference", reference), ("accelerated", accelerated)):
        loader = DataLoader(dataset, batch_size=8, num_workers=args.workers, shuffle=False, collate_fn=collate_fn)
        seen, start = 0, None
        for batch in loader:
            if start is None:
                start = time.perf_counter()  # after worker startup
                seen = 0
            seen += len(batch[1])
            if seen >= 4 * n:
                break
        elapsed = time.perf_counter() - start
        loader_results[name] = seen / elapsed
        print(f"DataLoader {args.workers} workers, batch 8, {name}: {seen / elapsed:.1f} img/s", flush=True)
    print(f"DataLoader speedup {loader_results['accelerated'] / loader_results['reference']:.2f}x")

    if args.out:
        json.dump(
            {
                "host": platform.node(),
                "cpu": platform.processor() or platform.machine(),
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "resolution": args.resolution,
                "samples": n,
                "instances": instances,
                "parity_mismatches": mismatches,
                "single_process": {
                    "reference_ms": results["reference_ms"],
                    "accelerated_ms": results["accelerated_ms"],
                    "reference_stages_ms": ref_stages,
                    "accelerated_stages_ms": acc_stages,
                },
                "dataloader_img_per_s": {**loader_results, "workers": args.workers, "batch": 8},
            },
            open(args.out, "w"),
            indent=1,
        )


if __name__ == "__main__":
    main()
