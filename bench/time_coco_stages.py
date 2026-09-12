"""Direct wall-clock stage probes, independent of cProfile call-stack accounting."""

import argparse
import copy
import functools
import hashlib
import inspect
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from coco_loader import (
    COCO_FINGERPRINT,
    DEFAULT_CFG,
    DataLoader,
    YOLODataset,
    _native,
    accelerate_dataset,
    digest_value,
    fingerprint,
    seed_worker,
    sha,
    source_hashes,
    validate_profile,
)
from torch.utils.data import Subset
from ultrafast_maskops import PackedPolygons, Rasterizer
from ultrafast_maskops.ultralytics import FastFormat
from ultralytics.data.augment import Format
from ultralytics.data.base import BaseDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--backend", choices=["reference", "native"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    validate_profile()
    report = json.loads(args.benchmark.read_text())
    assert "summary" in report and report["source_sha256"] == source_hashes()
    extension_sha256 = sha(Path(_native.__file__).read_bytes())
    assert extension_sha256 == report["extension_sha256"]
    config = report["results"][0]
    sample_count = config["images"]
    assert 1 <= sample_count <= 5000
    assert fingerprint(args.corpus.resolve()) == COCO_FINGERPRINT
    work = args.out.with_suffix(".runs")
    work.mkdir(parents=True, exist_ok=False)
    seed_worker(0)
    random.seed(912)
    np.random.seed(912)
    torch.manual_seed(912)
    hyp = copy.deepcopy(DEFAULT_CFG)
    hyp.mask_ratio, hyp.overlap_mask = config["mask_ratio"], config["overlap"]
    dataset = YOLODataset(
        img_path=str(args.corpus.resolve() / "images/val2017"),
        imgsz=config["imgsz"],
        batch_size=config["batch_size"],
        augment=False,
        cache=False,
        rect=False,
        hyp=hyp,
        task="segment",
        data={"names": dict(enumerate(map(str, range(80)))), "nc": 80},
    )
    assert len(dataset) == 5000
    if args.backend == "native":
        assert accelerate_dataset(dataset) == 1
    selected_dataset = Subset(dataset, range(sample_count)) if sample_count != 5000 else dataset
    loader = DataLoader(
        selected_dataset,
        batch_size=config["batch_size"],
        num_workers=0,
        shuffle=False,
        collate_fn=YOLODataset.collate_fn,
        generator=torch.Generator().manual_seed(912),
    )
    assert sys.getprofile() is None
    samples, originals = {}, []
    targets = [
        (BaseDataset, "load_image"),
        (YOLODataset, "update_labels_info"),
        (Format, "_format_img"),
        (Format, "apply_instances"),
        (Format, "_format_segments"),
        (FastFormat, "_format_segments"),
        (PackedPolygons, "from_segments"),
        (Rasterizer, "overlap"),
        (Rasterizer, "masks"),
        (YOLODataset, "collate_fn"),
    ]

    def probe(function, key):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            began = time.perf_counter_ns()
            try:
                return function(*args, **kwargs)
            finally:
                samples[key].append(time.perf_counter_ns() - began)

        return measured

    for owner, name in targets:
        descriptor = inspect.getattr_static(owner, name)
        originals.append((owner, name, descriptor))
        key = owner.__name__ + "." + name
        samples[key] = []
        if isinstance(descriptor, (staticmethod, classmethod)):
            replacement = type(descriptor)(probe(descriptor.__func__, key))
        else:
            replacement = probe(descriptor, key)
        setattr(owner, name, replacement)
    loader.collate_fn = YOLODataset.collate_fn
    seen = 0
    start = time.perf_counter()
    try:
        for batch in loader:
            seen += batch["img"].shape[0]
            del batch
    finally:
        elapsed = time.perf_counter() - start
        for owner, name, descriptor in originals:
            setattr(owner, name, descriptor)
        loader.collate_fn = YOLODataset.collate_fn
    assert seen == sample_count
    nonempty = sum(bool(len(label["cls"])) for label in dataset.labels[:sample_count])
    expected = {
        "BaseDataset.load_image": sample_count,
        "YOLODataset.update_labels_info": sample_count,
        "Format._format_img": sample_count,
        "Format.apply_instances": sample_count,
        "YOLODataset.collate_fn": (sample_count + config["batch_size"] - 1) // config["batch_size"],
    }
    expected["Format._format_segments"] = nonempty if args.backend == "reference" else 0
    for name in ["FastFormat._format_segments", "PackedPolygons.from_segments"]:
        expected[name] = nonempty if args.backend == "native" else 0
    expected["Rasterizer.overlap"] = nonempty if args.backend == "native" and config["overlap"] else 0
    expected["Rasterizer.masks"] = nonempty if args.backend == "native" and not config["overlap"] else 0
    counts = {k: len(v) for k, v in samples.items()}
    assert counts == expected, (counts, expected)
    (work / "stage-samples.json").write_text(json.dumps(samples) + "\n")
    stats = {
        k: {
            "calls": len(v),
            "inclusive_s": sum(v) / 1e9,
            "median_ns": float(np.median(v)) if v else None,
            "p95_ns": float(np.quantile(v, 0.95)) if v else None,
        }
        for k, v in samples.items()
    }
    digest = hashlib.sha256()
    verified = 0
    for batch in loader:
        digest_value(batch, digest)
        verified += batch["img"].shape[0]
        del batch
    assert verified == sample_count and all(r["output_sha256"] == digest.hexdigest() for r in report["results"])
    assert fingerprint(args.corpus.resolve()) == COCO_FINGERPRINT
    output = {
        "scope": "single workers=0 epoch with direct perf_counter_ns probes; nested times are inclusive and must not be added; instrumentation adds overhead, not a repeated performance comparison",
        "extension_sha256": extension_sha256,
        "backend": args.backend,
        "images": seen,
        "instrumented_epoch_s": elapsed,
        "source_sha256": source_hashes(),
        "profile_script_sha256": sha(Path(__file__).read_bytes()),
        "benchmark_sha256": sha(args.benchmark.read_bytes()),
        "output_sha256": digest.hexdigest(),
        "parity": True,
        "stats": stats,
        "expected_call_counts": expected,
        "raw_samples_sha256": sha((work / "stage-samples.json").read_bytes()),
    }
    with args.out.open("x") as stream:
        json.dump(output, stream, indent=2)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
