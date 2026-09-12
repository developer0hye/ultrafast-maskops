"""Diagnostic cProfile pass of the same full COCO loader, never a speed sample."""

import argparse
import copy
import cProfile
import hashlib
import json
import random
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
    assert config["images"] == 5000
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
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=0,
        shuffle=False,
        collate_fn=YOLODataset.collate_fn,
        generator=torch.Generator().manual_seed(912),
    )
    profile = cProfile.Profile()
    seen = 0
    start = time.perf_counter()
    profile.enable()
    for batch in loader:
        seen += batch["img"].shape[0]
        del batch
    profile.disable()
    elapsed = time.perf_counter() - start
    assert seen == 5000
    profile.create_stats()
    profile.dump_stats(str(work / "epoch.pstats"))
    stats = [
        {
            "file": file,
            "line": line,
            "function": function,
            "primitive_calls": cc,
            "calls": nc,
            "exclusive_s": tt,
            "inclusive_s": ct,
        }
        for (file, line, function), (cc, nc, tt, ct, _) in profile.stats.items()
    ]
    stats.sort(key=lambda row: row["inclusive_s"], reverse=True)
    digest = hashlib.sha256()
    verified = 0
    for batch in loader:
        digest_value(batch, digest)
        verified += batch["img"].shape[0]
        del batch
    assert verified == 5000 and all(r["output_sha256"] == digest.hexdigest() for r in report["results"])
    assert fingerprint(args.corpus.resolve()) == COCO_FINGERPRINT
    output = {
        "scope": "single instrumented workers=0 epoch for bottleneck diagnosis; profiler adds overhead, not a performance comparison",
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
    }
    with args.out.open("x") as stream:
        json.dump(output, stream, indent=2)
    print(
        json.dumps(
            [
                r
                for r in stats
                if r["function"]
                in {
                    "_format_segments",
                    "_format_img",
                    "load_image",
                    "update_labels_info",
                    "collate_fn",
                    "apply_instances",
                }
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
