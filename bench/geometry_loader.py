"""Augmented segmentation __getitem__ throughput: reference vs maskops backends.

Each backend runs in fresh processes with identical seeds, so every backend
must produce the same sample sequence; a full-output digest (image, classes,
boxes, masks, metadata) is compared outside all timers. Pairs alternate order.
"""

import argparse
import copy
import functools
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
from ultrafast_maskops import geometry
from ultrafast_maskops.ultralytics import accelerate_dataset
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data import augment, dataset

BACKENDS = ("reference", "masks", "geometry+masks")


def digest(value, h):
    if isinstance(value, dict):
        for key in sorted(value):
            h.update(key.encode() + b"\0")
            digest(value[key], h)
    elif isinstance(value, (list, tuple)):
        h.update(f"{type(value).__name__}:{len(value)}".encode())
        for part in value:
            digest(part, h)
    elif isinstance(value, (np.ndarray, torch.Tensor)):
        array = value.numpy() if isinstance(value, torch.Tensor) else value
        h.update(f"{array.dtype}:{array.shape}".encode() + np.ascontiguousarray(array).tobytes())
    else:
        h.update(f"{type(value).__name__}:{value!r}".encode() + b"\0")


def stage_timers():
    values = {}

    def wrap(owner, name, label):
        original = getattr(owner, name)

        @functools.wraps(original)
        def measured(*a, **k):
            started = time.perf_counter()
            try:
                return original(*a, **k)
            finally:
                values[label] = values.get(label, 0.0) + time.perf_counter() - started

        setattr(owner, name, measured)

    wrap(augment.RandomPerspective, "apply_segments", "apply_segments")
    wrap(geometry.FastRandomPerspective, "apply_segments", "apply_segments")
    wrap(augment.Format, "_format_segments", "masks")
    # Both are looked up as module globals at call time by update_labels_info.
    wrap(dataset, "resample_segments", "resample")
    wrap(geometry, "resample_stack", "resample")
    return values


def worker(args, backend):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
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
    if backend in ("masks", "geometry+masks"):
        accelerate_dataset(ds)
    if backend == "geometry+masks":
        geometry.accelerate_geometry(ds)
        geometry.interp_fused()  # one-time calibration outside timing
    for i in range(args.warmup):
        ds[i]
    timers = stage_timers()
    h = hashlib.sha256()
    elapsed = []
    for i in range(args.warmup, args.warmup + args.samples):
        started = time.perf_counter()
        sample = ds[i % len(ds)]
        elapsed.append(time.perf_counter() - started)
        digest(sample, h)
    return {
        "backend": backend,
        "samples": args.samples,
        "total_s": float(np.sum(elapsed)),
        "median_ms": float(np.median(elapsed) * 1e3),
        "mean_ms": float(np.mean(elapsed) * 1e3),
        "stages_s": timers,
        "output_sha256": h.hexdigest(),
        "rss_bytes": psutil.Process().memory_info().rss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--worker", choices=BACKENDS)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker:
        args.result.write_text(json.dumps(worker(args, args.worker)))
        return
    run_dir = args.out.with_suffix(".runs")
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "single-process augmented YOLODataset.__getitem__ (training defaults), identical seeds",
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version,
        "numpy": np.__version__,
        "cv2": cv2.__version__,
        "torch": torch.__version__,
        "interp_fused": geometry.interp_fused(),
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "results": [],
    }
    expected = None
    for round_ in range(args.rounds):
        order = args.backends if round_ % 2 == 0 else args.backends[::-1]
        for backend in order:
            output = run_dir / f"{round_}-{backend.replace('+', '_')}.json"
            command = [sys.executable, __file__, "--images", args.images, "--worker", backend, "--result", str(output)]
            command += ["--samples", str(args.samples), "--warmup", str(args.warmup)]
            subprocess.run(command, check=True)
            value = json.loads(output.read_text())
            expected = expected or value["output_sha256"]
            assert value["output_sha256"] == expected, f"output mismatch: {backend}"
            value["round"] = round_
            report["results"].append(value)
            args.out.write_text(json.dumps(report, indent=2))
            print(round_, backend, f"{value['mean_ms']:.3f} ms/sample", flush=True)
    summary = {}
    for backend in args.backends:
        means = [r["mean_ms"] for r in report["results"] if r["backend"] == backend]
        summary[backend] = {"mean_ms_median": float(np.median(means))}
    for backend in args.backends:
        summary[backend]["reference_over_backend"] = summary["reference"]["mean_ms_median"] / summary[backend]["mean_ms_median"]
    report["summary"] = summary
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
