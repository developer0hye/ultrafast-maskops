"""One full augmented training epoch over a COCO-scale segmentation dataset.

Both sides build their data as Ultralytics' trainer does for mode="train":
init_seeds(0), the dataset arguments of build_yolo_dataset with the default
configuration, and build_dataloader (seeded shuffle, seed_worker, persistent
InfiniteDataLoader workers) on CPU.

- reference: the unmodified pinned Ultralytics YOLODataset.
- accelerated: ultrafast_yolo_dataset.FastYOLODataset (annotation_cache="fast")
  with ultrafast_maskops accelerate_dataset and accelerate_geometry
  (persistent=True, shared collator).
- maskops: the unmodified YOLODataset with the same ultrafast_maskops calls.
- fast-dataset: FastYOLODataset alone.

The first of --backends is the baseline for the reported ratios.

Every measurement is a fresh process timing the constructor (label cache hit
or miss), the first batch (worker start-up included) and the whole epoch.
Before timing, the first --verify-batches batches of each backend are digested
in separate untimed processes and must be identical.
"""

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data.build import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import init_seeds

BACKENDS = ("reference", "accelerated", "maskops", "fast-dataset")
FAST_DATASET = ("accelerated", "fast-dataset")
MASKOPS = ("accelerated", "maskops")
TIMED = ("constructor_s", "first_batch_s", "epoch_s", "total_s", "samples_per_s", "peak_memory_bytes")


def cache_paths(images):
    labels = Path(str(Path(images).absolute()).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))
    return {b: labels.with_suffix(".uydfast" if b in FAST_DATASET else ".cache") for b in BACKENDS}


def make_dataset(backend, args):
    # build_yolo_dataset(cfg, img_path, batch, data, mode="train") with the default configuration.
    cfg = copy.deepcopy(DEFAULT_CFG)
    cfg.task, cfg.imgsz = "segment", args.imgsz
    kwargs = dict(
        img_path=str(args.images),
        imgsz=cfg.imgsz,
        batch_size=args.batch,
        augment=True,
        hyp=copy.copy(cfg),
        rect=False,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix=colorstr("train: "),
        task="segment",
        classes=None,
        data={"names": {i: str(i) for i in range(80)}, "channels": 3},
        fraction=args.fraction,
    )
    if backend in FAST_DATASET:
        from ultrafast_yolo_dataset.ultralytics import FastYOLODataset

        dataset = FastYOLODataset(**kwargs, annotation_cache="fast")
    else:
        dataset = YOLODataset(**kwargs)
    if backend in MASKOPS:
        from ultrafast_maskops import geometry
        from ultrafast_maskops.ultralytics import accelerate_dataset

        accelerate_dataset(dataset, persistent=True)
        geometry.accelerate_geometry(dataset, persistent=True)
    return dataset


def digest_value(value, digest):
    if isinstance(value, dict):
        for key in sorted(value):
            digest.update(str(key).encode() + b"\0")
            digest_value(value[key], digest)
    elif isinstance(value, (list, tuple)):
        digest.update(f"{type(value).__name__}:{len(value)}".encode())
        for item in value:
            digest_value(item, digest)
    elif isinstance(value, (np.ndarray, torch.Tensor)):
        array = value.numpy() if isinstance(value, torch.Tensor) else value
        digest.update(f"{array.dtype}:{array.shape}".encode() + np.ascontiguousarray(array).tobytes())
    else:
        digest.update(f"{type(value).__name__}:{value!r}".encode() + b"\0")


class PeakMemory:
    """Largest total memory of this process and its DataLoader workers.

    PSS where the platform reports it (shared pages split among processes),
    otherwise RSS summed over processes, which counts shared pages repeatedly.
    """

    def __init__(self, interval=1.0):
        self.value, self.kind = 0, None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(interval,), daemon=True)
        self._thread.start()

    def _sample(self, process):
        try:
            info = process.memory_full_info()
            if getattr(info, "pss", None) is not None:
                self.kind = "pss"
                return info.pss
        except (psutil.Error, AttributeError):
            pass
        try:
            self.kind = self.kind or "rss"
            return process.memory_info().rss
        except psutil.Error:
            return 0

    def _run(self, interval):
        me = psutil.Process()
        while not self._stop.wait(interval):
            try:
                children = me.children(recursive=True)
            except psutil.Error:
                children = []
            self.value = max(self.value, self._sample(me) + sum(self._sample(c) for c in children))

    def stop(self):
        self._stop.set()
        self._thread.join()
        return self.value


def worker(args):
    caches = cache_paths(args.images)
    own = caches[args.backend]
    if args.mode == "miss" and not args.verify_batches:
        own.unlink(missing_ok=True)
    before = own.stat().st_mtime_ns if own.exists() else None
    init_seeds(0)
    peak = PeakMemory()
    load_before = os.getloadavg()
    started = time.perf_counter()
    dataset = make_dataset(args.backend, args)
    constructed = time.perf_counter()
    after = own.stat().st_mtime_ns if own.exists() else None
    result = {
        "backend": args.backend,
        "mode": args.mode,
        "images": len(dataset),
        "constructor_s": constructed - started,
        "cache_hit": before is not None and before == after,
        "loadavg_before": load_before,
    }
    if args.prime:
        peak.stop()
        return result
    loader = build_dataloader(
        dataset, args.batch, args.workers, shuffle=True, rank=-1, drop_last=False, pin_memory=False, device="cpu"
    )
    digest = hashlib.sha256()
    batches = samples = 0
    first = None
    for batch in loader:
        if first is None:
            first = time.perf_counter()
        batches += 1
        samples += len(batch["im_file"])
        if args.verify_batches:
            digest_value(batch, digest)
            if batches == args.verify_batches:
                break
    finished = time.perf_counter()
    workers = loader.num_workers
    del loader, batch
    gc.collect()
    result.update(
        {
            "workers": workers,
            "batches": batches,
            "samples": samples,
            "first_batch_s": first - constructed,
            "epoch_s": finished - constructed,
            "total_s": finished - started,
            "samples_per_s": samples / (finished - constructed),
            "peak_memory_bytes": peak.stop(),
            "peak_memory_kind": peak.kind,
            "loadavg_after": os.getloadavg(),
        }
    )
    if args.verify_batches:
        result["output_sha256"] = digest.hexdigest()
    return result


def run(args, run_dir, backend, name, *extra):
    output = run_dir / f"{name}-{backend}.json"
    command = [sys.executable, __file__, "--images", str(args.images), "--backend", backend, "--mode", args.mode]
    command += ["--result", str(output), "--workers", str(args.workers), "--batch", str(args.batch)]
    # Timed and priming runs read the whole epoch; verification runs override this.
    command += ["--imgsz", str(args.imgsz), "--fraction", str(args.fraction), "--verify-batches", "0", *extra]
    env = {**os.environ, "YOLO_OFFLINE": "true"}
    with (run_dir / f"{name}-{backend}.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    return json.loads(output.read_text())


def versions():
    found = {}
    for name in ("ultralytics", "ultrafast-maskops", "ultrafast-yolo-dataset", "torch", "numpy", "opencv-python"):
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def summarize(runs, backends):
    summary = {}
    for backend in backends:
        mine = [r for r in runs if r["backend"] == backend]
        summary[backend] = {
            key: {"median": statistics.median(r[key] for r in mine), "rounds": [r[key] for r in mine]} for key in TIMED
        }
    base = backends[0]
    for backend in backends[1:]:
        summary[f"{base}_over_{backend}"] = {
            key: summary[base][key]["median"] / summary[backend][key]["median"]
            for key in ("constructor_s", "first_batch_s", "epoch_s", "total_s", "peak_memory_bytes")
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True, help="images directory of the COCO-scale split")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--mode", choices=["hit", "miss"], default="hit")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--fraction", type=float, default=1.0, help="smoke tests only")
    parser.add_argument("--verify-batches", type=int, default=64)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=["reference", "accelerated"])
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--prime", action="store_true")
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.result:
        value = worker(args)
        with args.result.open("x") as stream:
            json.dump(value, stream)
        return
    if args.out is None:
        parser.error("--out required")
    run_dir = args.out.with_suffix(".runs")
    run_dir.mkdir(parents=True, exist_ok=False)
    images = sorted(os.scandir(args.images), key=lambda e: e.name)
    report = {
        "scope": "one augmented training epoch, constructor to last batch, Ultralytics train-mode dataset and loader",
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "python": sys.version,
        "versions": versions(),
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "corpus": {"images": len(images), "bytes": sum(e.stat().st_size for e in images)},
        "verification": {},
        "results": [],
    }
    backends = list(dict.fromkeys(args.backends))
    if args.mode == "hit":
        for backend in backends:
            run(args, run_dir, backend, "prime", "--prime")
            print("primed", backend, flush=True)
    if args.verify_batches:
        for backend in backends:
            value = run(args, run_dir, backend, "verify", "--verify-batches", str(args.verify_batches))
            report["verification"][backend] = value["output_sha256"]
            print("verified", backend, value["output_sha256"][:16], flush=True)
        if len(set(report["verification"].values())) != 1:
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            raise SystemExit("verification digests differ; no timing was run")
    for round_ in range(args.rounds):
        for backend in backends if round_ % 2 == 0 else backends[::-1]:
            value = run(args, run_dir, backend, str(round_))
            value["round"] = round_
            report["results"].append(value)
            print(round_, backend, f"total {value['total_s']:.1f}s epoch {value['epoch_s']:.1f}s", flush=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
    report["summary"] = summarize(report["results"], backends)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report["summary"].items() if "_over_" in k}, indent=2))


if __name__ == "__main__":
    main()
