"""Verify real augmented COCO targets in independent reference/reference/native processes.

Correctness only: hashing occurs in the loader loop and there is no performance
claim. Run after the bound full-loader benchmark and fresh-cache check finish.
Reference replay must agree before the candidate is accepted. Each worker count
is its own RNG experiment; streams need not agree across different worker counts.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import random
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from coco_loader import (
    COCO_FINGERPRINT,
    DEFAULT_CFG,
    UPSTREAM_COMMIT,
    UPSTREAM_FILES,
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

# Fixed stress profile. These are verification settings, not training advice or
# a claim that this is the default Ultralytics training configuration.
STRESS = {
    "mosaic": 1.0,
    "mixup": 1.0,
    "copy_paste": 1.0,
    "copy_paste_mode": "flip",
    "cutmix": 0.0,
    "degrees": 15.0,
    "translate": 0.15,
    "scale": 0.5,
    "shear": 5.0,
    "perspective": 0.0005,
    "flipud": 0.5,
    "fliplr": 0.5,
    "bgr": 0.5,
}


def dataset_profile(candidate):
    if candidate == "mask":
        return None
    import ultrafast_yolo_dataset as package
    from ultrafast_yolo_dataset.ultralytics import check_profile

    check_profile()
    root = Path(package.__file__).parent
    paths = sorted(root.glob("*.py")) + [root / "_ultralytics_profile.json"]
    return {
        "version": package.__version__,
        "python_sha256": {p.name: sha(p.read_bytes()) for p in paths},
        "extension_sha256": sha(Path(package._native.__file__).read_bytes()),
    }


def environment_versions():
    versions = {}
    for name in [
        "numpy",
        "pillow",
        "opencv-python",
        "torch",
        "torchvision",
        "ultralytics",
        "albumentations",
        "pi-heif",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def validate_binding(args):
    validate_profile()
    report = json.loads(args.benchmark.read_text())
    fresh = json.loads(args.fresh_check.read_text())
    assert "summary" in report, "wait for the full benchmark to finish"
    assert len(report["results"]) >= 10 and all(r["images"] == 5000 for r in report["results"])
    assert fresh["benchmark_sha256"] == sha(args.benchmark.read_bytes())
    assert fresh["all_benchmark_outputs_match_fresh_reference"]
    assert fresh["matched_runs"] == len(report["results"])
    assert report["upstream_commit"] == UPSTREAM_COMMIT and report["upstream_files"] == UPSTREAM_FILES
    assert all(r["output_sha256"] == fresh["fresh_reference"]["output_sha256"] for r in report["results"])
    assert fresh["fresh_reference"]["source_sha256"] == report["source_sha256"]
    assert fresh["fresh_reference"]["extension_sha256"] == report["extension_sha256"]
    sources = source_hashes()
    extension = sha(Path(_native.__file__).read_bytes())
    if not args.reference_only_baseline:
        assert sources == report["source_sha256"]
        assert extension == report["extension_sha256"]
    build = _native.build_profile()
    for key, path in (
        ("bindings_sha256", "src/bindings.cpp"),
        ("cmake_sha256", "CMakeLists.txt"),
        ("template_sha256", "src/build_profile.h.in"),
    ):
        assert build[key] == sources[path], f"compiled source mismatch: {path}"
    root = args.corpus.resolve()
    assert fingerprint(root) == COCO_FINGERPRINT
    assert sha((root / "labels/val2017.cache").read_bytes()) == fresh["fresh_cache_sha256"]
    return sources, extension, fresh["fresh_cache_sha256"]


def worker(args):
    sources, extension, cache_sha = validate_binding(args)
    dataset_artifact = dataset_profile(args.candidate)
    seed_worker(0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    hyp = copy.deepcopy(DEFAULT_CFG)
    for name, value in STRESS.items():
        setattr(hyp, name, value)
    hyp.mask_ratio, hyp.overlap_mask = args.mask_ratio, args.overlap == "yes"
    dataset_type, native_options = YOLODataset, {}
    if args.backend == "native" and args.candidate == "both":
        from ultrafast_yolo_dataset.ultralytics import FastYOLODataset

        dataset_type = FastYOLODataset
        native_options = {"annotation_cache": "native", "cache_dir": args.result.parent / "native-cache"}
    dataset = dataset_type(
        img_path=str(args.corpus.resolve() / "images/val2017"),
        imgsz=args.imgsz,
        batch_size=args.batch,
        augment=True,
        cache=False,
        rect=False,
        hyp=hyp,
        task="segment",
        data={"names": dict(enumerate(map(str, range(80)))), "nc": 80},
        **native_options,
    )
    assert len(dataset) == 5000 and dataset.augment and not dataset.cache and not dataset.rect
    if args.backend == "native":
        assert accelerate_dataset(dataset) == 1
    count = args.limit or len(dataset)
    assert 1 <= count <= len(dataset)
    selected = Subset(dataset, range(count)) if args.limit else dataset
    extra = (
        {"multiprocessing_context": "spawn", "persistent_workers": False, "prefetch_factor": 2}
        if args.worker_count
        else {}
    )
    loader = DataLoader(
        selected,
        batch_size=args.batch,
        num_workers=args.worker_count,
        collate_fn=YOLODataset.collate_fn,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed),
        **extra,
    )
    digest, batches, images, objects = hashlib.sha256(), [], 0, 0
    fields = set()
    object_counts, mask_dtypes = [], set()
    for index, batch in enumerate(loader):
        required = {"img", "cls", "bboxes", "batch_idx", "masks", "sem_masks", "im_file"}
        assert required <= batch.keys(), batch.keys()
        fields.update(batch)
        batch_digest = hashlib.sha256()
        digest_value(batch, batch_digest)
        digest_value(batch, digest)
        n = int(batch["img"].shape[0])
        instances = int(batch["cls"].shape[0])
        assert len(batch["bboxes"]) == len(batch["batch_idx"]) == instances
        mask_dtypes.add(str(batch["masks"].dtype))
        object_counts.extend(torch.bincount(batch["batch_idx"].to(torch.int64), minlength=n).tolist())
        batches.append({"batch": index, "images": n, "instances": instances, "sha256": batch_digest.hexdigest()})
        images += n
        objects += instances
        del batch
    assert images == count and len(batches) == (count + args.batch - 1) // args.batch
    assert sources == source_hashes() and extension == sha(Path(_native.__file__).read_bytes())
    assert fingerprint(args.corpus.resolve()) == COCO_FINGERPRINT
    assert sha((args.corpus.resolve() / "labels/val2017.cache").read_bytes()) == cache_sha
    assert dataset_profile(args.candidate) == dataset_artifact
    output = {
        "backend": args.backend,
        "candidate": args.candidate,
        "dataset_artifact": dataset_artifact,
        "dataset_class": type(dataset).__name__,
        "annotation_cache_hit": getattr(dataset, "annotation_cache_hit", None),
        "workers": args.worker_count,
        "seed": args.seed,
        "images": images,
        "augmented_instances": objects,
        "objects_per_image": {
            "min": min(object_counts),
            "max": max(object_counts),
            "median": float(np.median(object_counts)),
        },
        "mask_dtypes": sorted(mask_dtypes),
        "fields": sorted(fields),
        "output_sha256": digest.hexdigest(),
        "batches": batches,
        "source_sha256": sources,
        "extension_sha256": extension,
        "cache_sha256": cache_sha,
        "script_sha256": sha(Path(__file__).read_bytes()),
        "hyp": vars(hyp),
        "packages": environment_versions(),
    }
    args.result.write_text(json.dumps(output, indent=2) + "\n")


def run_child(command, log_path):
    with log_path.open("x") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait()
            if code:
                raise RuntimeError(f"verification process exited {code}: {log_path}")
        finally:
            if child.poll() is None:
                if os.name == "posix":
                    os.killpg(child.pid, signal.SIGTERM)
                else:
                    child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(child.pid, signal.SIGKILL)
                    else:
                        child.kill()
                    child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--fresh-check", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 2, 8])
    parser.add_argument("--overlap", choices=["yes", "no"], default="yes")
    parser.add_argument("--candidate", choices=["mask", "both"], default="mask")
    parser.add_argument(
        "--reference-only-baseline",
        action="store_true",
        help="use an older benchmark only for original-cache provenance; current candidate still needs matching compiled source hashes",
    )
    parser.add_argument("--mask-ratio", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=912)
    parser.add_argument(
        "--limit", type=int, default=0, help="first N output samples for a pilot; mixing can read all 5000 images"
    )
    parser.add_argument("--backend", choices=["reference", "native"])
    parser.add_argument("--worker-count", type=int, default=0)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.backend:
        worker(args)
        return
    assert args.out and not args.out.exists() and all(w >= 0 for w in args.workers)
    sources, extension, cache_sha = validate_binding(args)
    work = args.out.with_suffix(".runs")
    work.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "augmented real COCO target parity pilot" if args.limit else "full augmented real COCO target parity",
        "method": "independent reference, reference replay, native processes for each worker count; hashing inside iteration, no performance claim; RNG streams may differ between worker counts",
        "stress_overrides": STRESS,
        "imgsz": args.imgsz,
        "batch_size": args.batch,
        "mask_ratio": args.mask_ratio,
        "overlap": args.overlap == "yes",
        "candidate": args.candidate,
        "dataset_artifact": dataset_profile(args.candidate),
        "upstream_commit": UPSTREAM_COMMIT,
        "benchmark_sha256": sha(args.benchmark.read_bytes()),
        "fresh_check_sha256": sha(args.fresh_check.read_bytes()),
        "benchmark_role": "reference cache provenance only; not candidate performance evidence"
        if args.reference_only_baseline
        else "exact candidate benchmark and reference cache provenance",
        "source_sha256": sources,
        "extension_sha256": extension,
        "script_sha256": sha(Path(__file__).read_bytes()),
        "cache_sha256": cache_sha,
        "packages": environment_versions(),
        "results": [],
    }
    for workers in args.workers:
        expected = None
        for trial, backend in enumerate(["reference", "reference", "native"]):
            stem = f"w{workers}-{trial}-{backend}"
            result = work / f"{stem}.json"
            command = [sys.executable, str(Path(__file__).resolve())]
            for name, value in {
                "corpus": args.corpus.resolve(),
                "benchmark": args.benchmark.resolve(),
                "fresh-check": args.fresh_check.resolve(),
                "backend": backend,
                "worker-count": workers,
                "result": result,
                "overlap": args.overlap,
                "mask-ratio": args.mask_ratio,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "seed": args.seed,
                "limit": args.limit,
                "candidate": args.candidate,
            }.items():
                command.extend(["--" + name, str(value)])
            if args.reference_only_baseline:
                command.append("--reference-only-baseline")
            run_child(command, work / f"{stem}.log")
            current = json.loads(result.read_text())
            assert current["source_sha256"] == sources and current["extension_sha256"] == extension
            assert current["script_sha256"] == report["script_sha256"]
            assert current["dataset_artifact"] == report["dataset_artifact"]
            assert current["packages"] == report["packages"]
            if expected is None:
                expected = current
            else:
                assert current["hyp"] == expected["hyp"]
                assert current["batches"] == expected["batches"], f"augmented target mismatch; inspect {work}"
                assert current["output_sha256"] == expected["output_sha256"]
            current["trial"] = trial
            current["matches_reference"] = True
            report["results"].append(current)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(f"workers={workers} {backend} trial={trial}: {current['images']} augmented outputs match", flush=True)
    report["complete"] = True
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
