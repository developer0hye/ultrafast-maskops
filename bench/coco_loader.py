"""Full COCO CPU DataLoader epochs, with separate untimed output verification.

Every sample is a fresh process. The ordinary YOLODataset is used on both sides;
its final Format is replaced, and explicit persistent mode also enables the
instance shared collator. No training, dataset-parser acceleration, image RAM
cache, or augmentation is included in this experiment.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import resource
import signal
import subprocess
import sys
import time
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
import ultralytics
from torch.utils.data import DataLoader, Subset
from ultrafast_maskops import _native, backend_info
from ultrafast_maskops._shared_collate import shared_collate_fn
from ultrafast_maskops.ultralytics import accelerate_dataset, check_profile
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data.dataset import YOLODataset

# File hashes are checked against the immutable upstream commit, not its version string.
UPSTREAM_COMMIT = "795a556942a12fe0124cf767888194a1d0b83e2e"
UPSTREAM_FILES = {
    "data/base.py": "863018d66387a9904d61368c5058e2ce1cfe830b6fd99a399fe9df159d4b0555",
    "data/dataset.py": "6c30496c691aefeb0646bea23a1519d2960c927b71dae10c370afcd9765f0e0e",
    "data/augment.py": "3631af29958e17bc442cbaa49a6bd32b660615b1bb1171a4e8e8692bace1f428",
    "data/utils.py": "6135197e734e4dba2b85caca69a7757979f5ac4c8cb08223feb22888fe9c0ade",
    "utils/ops.py": "1a8034386c970fb779ef18f7117083dede6d138d4dbc5f66ed5db76624d36158",
    "utils/instance.py": "609e24de399439afe3eeb453334d1e700afa53047c07701e3f25c3e97bdd2b98",
}
COCO_FINGERPRINT = {
    "files": 9952,
    "bytes": 830426850,
    "sha256": "8b079ad2d8472ff17b6cca9b0250e831472f12ef500f2b065125f6f39406c208",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(root):
    digest, size = hashlib.sha256(), 0
    files = sorted((root / "images").rglob("*.jpg")) + sorted((root / "labels").rglob("*.txt"))
    for path in files:
        data = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + data)
        size += len(data)
    return {"files": len(files), "bytes": size, "sha256": digest.hexdigest()}


def validate_profile():
    check_profile()
    assert len(UPSTREAM_FILES) == 6, "freeze the complete upstream file profile"
    root = Path(ultralytics.__file__).parent
    for name, expected in UPSTREAM_FILES.items():
        assert sha((root / name).read_bytes()) == expected, f"upstream source mismatch: {name}"
    runtime_sources = Path(__file__).resolve().parents[1] / "python/ultrafast_maskops"
    installed = Path(_native.__file__).parent
    for path in runtime_sources.glob("*.py"):
        assert path.read_bytes() == (installed / path.name).read_bytes(), f"installed runtime mismatch: {path.name}"


def digest_value(value, digest):
    # Length/type boundaries prevent ambiguous concatenations. Paths are kept
    # verbatim: compare runs on the same host/fixture, not different root paths.
    if isinstance(value, dict):
        digest.update(f"dict:{len(value)}:".encode())
        for key in sorted(value):
            digest_value(key, digest)
            digest_value(value[key], digest)
    elif isinstance(value, (list, tuple)):
        digest.update(f"{type(value).__name__}:{len(value)}:".encode())
        for item in value:
            digest_value(item, digest)
    elif isinstance(value, (torch.Tensor, np.ndarray)):
        array = value.numpy() if isinstance(value, torch.Tensor) else value
        digest.update(f"{type(value).__name__}:{array.dtype}:{array.shape}:".encode())
        digest.update(array.tobytes())
    else:
        encoded = repr(value).encode()
        digest.update(f"{type(value).__name__}:{len(encoded)}:".encode() + encoded)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


def peak_rss():
    if sys.platform.startswith("linux"):
        return int(Path("/proc/self/status").read_text().split("VmHWM:")[1].split()[0]) * 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ["src/bindings.cpp", "src/build_profile.h.in", "CMakeLists.txt", "pyproject.toml", "bench/coco_loader.py"]
    names.extend(str(p.relative_to(root)) for p in (root / "python/ultrafast_maskops").glob("*.py"))
    return {n: sha((root / n).read_bytes()) for n in sorted(names)}


def distribution(values):
    return {
        "min": int(min(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": int(max(values)),
    }


def worker(args):
    validate_profile()
    root = args.corpus.resolve()
    manifest = json.loads((root / "coco.json").read_text())
    assert manifest["task"] == "segment" and manifest["count"] == 5000
    assert manifest["fingerprint"] == COCO_FINGERPRINT == fingerprint(root)
    seed_worker(0)
    random.seed(912)
    np.random.seed(912)
    torch.manual_seed(912)
    hyp = copy.deepcopy(DEFAULT_CFG)
    hyp.mask_ratio, hyp.overlap_mask = args.mask_ratio, args.overlap == "yes"
    started = time.perf_counter()
    dataset = YOLODataset(
        img_path=str(root / "images/val2017"),
        imgsz=args.imgsz,
        batch_size=args.batch,
        augment=False,
        cache=False,
        rect=False,
        hyp=hyp,
        task="segment",
        data={"names": dict(enumerate(map(str, range(80)))), "nc": 80},
    )
    constructor_s = time.perf_counter() - started
    assert len(dataset) == 5000
    assert dataset.augment is False and not dataset.cache
    replaced = accelerate_dataset(dataset, persistent=args.persistent_mask) if args.backend == "native" else 0
    if args.backend == "native":
        assert replaced == 1
    expected_collator = (
        shared_collate_fn if args.backend == "native" and args.persistent_mask else YOLODataset.collate_fn
    )
    assert dataset.collate_fn is expected_collator
    count = args.limit or len(dataset)
    assert 1 <= count <= len(dataset)
    selected = dataset.labels[:count]
    object_counts = [len(v["cls"]) for v in selected]
    raw_vertices = [len(s) for v in selected for s in v["segments"]]
    # This is the upstream per-image resampling rule, not an alternative transform.
    resampled_vertices = [
        max(1000, max(map(len, v["segments"])) + 1) if max(map(len, v["segments"])) > 1000 else 1000
        for v in selected
        if v["segments"]
    ]
    topology = {
        "images": count,
        "instances": sum(object_counts),
        "objects_per_image": distribution(object_counts),
        "raw_vertices_per_polygon": distribution(raw_vertices),
        "resampled_vertices_per_nonempty_image": distribution(resampled_vertices),
    }
    data = Subset(dataset, range(count)) if args.limit else dataset
    kwargs = (
        {"multiprocessing_context": "spawn", "persistent_workers": True, "prefetch_factor": 2}
        if args.worker_count
        else {}
    )
    loader = DataLoader(
        data,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.worker_count,
        collate_fn=dataset.collate_fn,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(912),
        **kwargs,
    )
    assert loader.collate_fn is expected_collator
    before = psutil.Process().memory_info().rss
    available_before = psutil.virtual_memory().available
    swap_before = psutil.swap_memory()
    load_before = os.getloadavg()
    # The parent samples process-family RSS only while this marker is present.
    # Worker launch and queue fill belong to the first epoch's elapsed time.
    args.marker.write_text("measure")
    start = last = time.perf_counter()
    arrivals, seen, batches = [], 0, 0
    first_batch_s = None
    for batch in loader:
        now = time.perf_counter()
        arrivals.append(now - last)
        if first_batch_s is None:
            first_batch_s = now - start
        seen += batch["img"].shape[0]
        batches += 1
        last = now
        del batch
    elapsed = time.perf_counter() - start
    parent_peak = peak_rss()
    swap_after = psutil.swap_memory()
    args.marker.write_text("validate")
    assert seen == count
    # Full second pass on the same deterministic workload, outside timing/RSS.
    # No hashing, equality comparison or target tensor reduction occurred above.
    digest, verified, verification_batches = hashlib.sha256(), 0, 0
    for batch in loader:
        digest_value(batch, digest)
        verified += batch["img"].shape[0]
        verification_batches += 1
        del batch
    assert verified == count and verification_batches == batches
    # Outside all timers and resource-sampling markers. Preserve failures from
    # either backend: a returned epoch alone does not prove clean worker exit.
    iterator = getattr(loader, "_iterator", None)
    workers = list(getattr(iterator, "_workers", ()))
    worker_pids = [worker.pid for worker in workers]
    if iterator is not None:
        iterator._shutdown_workers()
    worker_exitcodes = [worker.exitcode for worker in workers]
    assert len(workers) == args.worker_count
    assert all(not worker.is_alive() and worker.exitcode == 0 for worker in workers), worker_exitcodes
    assert fingerprint(root) == manifest["fingerprint"]
    result = {
        "backend": args.backend,
        "workers": args.worker_count,
        "persistent_mask": args.persistent_mask,
        "collator": f"{loader.collate_fn.__module__}.{loader.collate_fn.__qualname__}",
        "worker_pids": worker_pids,
        "worker_exitcodes": worker_exitcodes,
        "images": count,
        "batch_size": args.batch,
        "imgsz": args.imgsz,
        "mask_ratio": args.mask_ratio,
        "overlap": args.overlap == "yes",
        "augment": False,
        "image_cache": False,
        "constructor_s_untimed": constructor_s,
        "epoch_s": elapsed,
        "first_batch_s": first_batch_s,
        "images_per_s": count / elapsed,
        "after_first_batch_s": elapsed - first_batch_s,
        "after_first_batch_images_per_s": (count - min(args.batch, count)) / (elapsed - first_batch_s),
        "batch_arrivals_s": arrivals,
        "batch_arrival_p50_s": float(np.median(arrivals[1:] or arrivals)),
        "batch_arrival_p95_s": float(np.quantile(arrivals[1:] or arrivals, 0.95)),
        "parent_rss_before_epoch_bytes": before,
        "parent_peak_rss_through_epoch_bytes": parent_peak,
        "output_sha256": digest.hexdigest(),
        "verification_images": verified,
        "verification_batches": verification_batches,
        "topology": topology,
        "loadavg_before_epoch": load_before,
        "available_ram_before_epoch_bytes": available_before,
        "host_swap_in_delta_epoch_bytes": swap_after.sin - swap_before.sin,
        "host_swap_out_delta_epoch_bytes": swap_after.sout - swap_before.sout,
        "loadavg": os.getloadavg(),
        "available_ram_bytes": psutil.virtual_memory().available,
        "source_sha256": source_hashes(),
        "extension_sha256": sha(Path(_native.__file__).read_bytes()),
    }
    with args.result.open("x") as stream:
        json.dump(result, stream, indent=2)
    args.marker.write_text("done")


def run_sample(args, run_dir, backend, workers, round_):
    stem = f"{workers}-{round_}-{backend}"
    result, marker = run_dir / (stem + ".json"), run_dir / (stem + ".phase")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--corpus",
        str(args.corpus.resolve()),
        "--backend",
        backend,
        "--worker-count",
        str(workers),
        "--result",
        str(result),
        "--marker",
        str(marker),
        "--batch",
        str(args.batch),
        "--imgsz",
        str(args.imgsz),
        "--mask-ratio",
        str(args.mask_ratio),
        "--overlap",
        args.overlap,
        "--limit",
        str(args.limit),
    ]
    if args.persistent_mask:
        command.append("--persistent-mask")
    samples = []
    with (run_dir / (stem + ".log")).open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            monitored = psutil.Process(process.pid)
            while process.poll() is None:
                if marker.exists() and marker.read_text() == "measure":
                    total, count = 0, 0
                    try:
                        family = [monitored, *monitored.children(recursive=True)]
                    except psutil.NoSuchProcess:
                        family = []
                    for child in family:
                        try:
                            total += child.memory_info().rss
                            count += 1
                        except psutil.NoSuchProcess:
                            pass
                    if count:
                        samples.append([time.monotonic(), total, count])
                time.sleep(0.05)
            if process.returncode:
                raise RuntimeError(f"worker exited {process.returncode}: {run_dir / (stem + '.log')}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    value = json.loads(result.read_text())
    assert samples, "epoch finished without a memory sample; increase pilot size"
    value.update(
        round=round_,
        sampled_family_peak_rss_bytes=max(s[1] for s in samples),
        memory_sample_count=len(samples),
        max_sampled_processes=max(s[2] for s in samples),
        memory_sample_max_gap_s=max((b[0] - a[0] for a, b in pairwise(samples)), default=None),
    )
    (run_dir / (stem + ".memory.json")).write_text(json.dumps(samples) + "\n")
    return value


def summarize(runs):
    summary = {}
    for workers in sorted({r["workers"] for r in runs}):
        summary[str(workers)] = {}
        for metric in [
            "epoch_s",
            "first_batch_s",
            "after_first_batch_s",
            "images_per_s",
            "sampled_family_peak_rss_bytes",
        ]:
            values = {
                b: np.array([r[metric] for r in runs if r["workers"] == workers and r["backend"] == b])
                for b in ["reference", "native"]
            }
            n = len(values["reference"])
            assert n == len(values["native"]) and n > 0
            indices = np.random.default_rng(912).integers(0, n, (10000, n))
            ratios = np.median(values["reference"][indices], axis=1) / np.median(values["native"][indices], axis=1)
            summary[str(workers)][metric] = {
                **{b: {"median": float(np.median(v)), "p95": float(np.quantile(v, 0.95))} for b, v in values.items()},
                "reference_over_native_ci95": np.quantile(ratios, [0.025, 0.975]).tolist(),
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 2, 8])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--mask-ratio", type=int, default=4)
    parser.add_argument("--overlap", choices=["yes", "no"], default="yes")
    parser.add_argument("--persistent-mask", action="store_true", help="native instance factory and shared collator")
    parser.add_argument("--limit", type=int, default=0, help="pilot only: first N images; zero means all 5000")
    parser.add_argument("--backend", choices=["reference", "native"])
    parser.add_argument("--worker-count", type=int, default=0)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--marker", type=Path)
    args = parser.parse_args()
    if args.backend:
        worker(args)
        return
    assert args.out and args.rounds > 0 and all(w >= 0 for w in args.workers)
    validate_profile()
    run_dir = args.out.with_suffix(".runs")
    run_dir.mkdir(parents=True, exist_ok=False)
    assert fingerprint(args.corpus.resolve()) == COCO_FINGERPRINT
    cpu = (
        subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if sys.platform == "darwin"
        else next(
            s.split(":", 1)[1].strip()
            for s in Path("/proc/cpuinfo").read_text().splitlines()
            if s.startswith("model name")
        )
    )
    report = {
        "complete": False,
        "rounds": args.rounds,
        "worker_counts": args.workers,
        "scope": "pilot" if args.limit or args.rounds < 5 else "full COCO non-augmented CPU DataLoader",
        "method": "five or specified alternating fresh processes per backend/worker count; whole first epoch and first-batch-excluded remainder; spawn persistent workers, prefetch=2, no pinning; ordinary YOLODataset on both sides; original reference collator, optional native shared collator; strict zero worker exit checks after untimed verification",
        "persistent_mask": args.persistent_mask,
        "validation": "full second pass hashes every batch key/type/dtype/shape/value outside timers; input fingerprint before/after each process pre-reads OS cache; no cold-storage claim",
        "memory": "external parent samples benchmark process plus recursive children every 50 ms during timed epoch; RSS sum counts shared pages more than once, is not unique memory or allocation; sampling can miss brief peaks; parent high-water mark also includes dataset construction",
        "platform": platform.platform(),
        "cpu": cpu,
        "physical_cpus": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "python": sys.version,
        "packages": {
            n: importlib.metadata.version(n)
            for n in ["numpy", "pillow", "opencv-python", "torch", "torchvision", "ultralytics"]
        },
        "native": backend_info(),
        "cv2_build": cv2.getBuildInformation(),
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_files": UPSTREAM_FILES,
        "source_sha256": source_hashes(),
        "extension_sha256": sha(Path(_native.__file__).read_bytes()),
        "corpus": json.loads((args.corpus / "coco.json").read_text()),
        "results": [],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    expected = None
    for workers in args.workers:
        for round_ in range(args.rounds):
            for backend in ["reference", "native"] if round_ % 2 == 0 else ["native", "reference"]:
                value = run_sample(args, run_dir, backend, workers, round_)
                assert (
                    value["source_sha256"] == report["source_sha256"]
                    and value["extension_sha256"] == report["extension_sha256"]
                )
                expected = expected or value["output_sha256"]
                assert value["output_sha256"] == expected, "complete epoch parity mismatch"
                value["parity"] = True
                report["results"].append(value)
                args.out.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    workers,
                    round_,
                    backend,
                    f"{value['epoch_s']:.3f} s, {value['images_per_s']:.1f} images/s",
                    flush=True,
                )
    report["summary"] = summarize(report["results"])
    report["complete"] = True
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
