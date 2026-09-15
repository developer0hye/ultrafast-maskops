"""End-to-end GPU training: the unmodified SegmentationTrainer against FastSegmentationTrainer.

    python bench/gpu_training.py --corpus /path/to/coco/segment --out series.json \
        --workers 0 2 8 --repeats 3 --epochs 2 --batch 16

Every job is a fresh process that trains YOLO11n-seg from random weights on the
5,000-image COCO val2017 segmentation fixture (a performance fixture, not an
accuracy experiment; validation is off except Ultralytics' final one). Per
epoch it records the wall time between the trainer's epoch callbacks with CUDA
synchronized at both ends, every batch's loss items, and finally a hash of the
model weights. Backends alternate order across repeats. The parent checks that
every job of a worker count produced the same loss vectors and weights, which
is what "the training run does not change" means here.

This harness does not depend on bench/coco_loader.py, so it also runs on
Windows; it is lighter than bench/train_coco_gpu.py (no fixture fingerprints,
process-family RSS sampling or resume cases).
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

BACKENDS = ("reference", "fast")


def job(args):
    os.environ.setdefault("YOLO_OFFLINE", "true")  # no font, weight or AMP-check downloads
    import cv2
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.segment import SegmentationTrainer
    from ultralytics.utils import YAML

    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    if args.backend == "fast":
        from ultrafast_maskops.training import FastSegmentationTrainer as base
    else:
        base = SegmentationTrainer

    record = {"epochs": [], "losses": [], "batches": []}
    state = {}

    def epoch_start(trainer):
        torch.cuda.synchronize()
        state["start"] = time.perf_counter()
        record["losses"].append([])

    def batch_end(trainer):
        items = trainer.loss_items  # a name -> tensor dict in this Ultralytics version
        values = items.values() if isinstance(items, dict) else items
        record["losses"][-1].append([float(v) for v in values])

    def epoch_end(trainer):
        torch.cuda.synchronize()
        record["epochs"].append(time.perf_counter() - state["start"])
        record["batches"].append(len(record["losses"][-1]))
        print(f"    epoch {len(record['epochs'])}: {record['epochs'][-1]:.1f} s", flush=True)

    class Timed(base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.add_callback("on_train_epoch_start", epoch_start)
            self.add_callback("on_train_batch_end", batch_end)
            self.add_callback("on_train_epoch_end", epoch_end)

    data = args.out.with_suffix(".data.yaml")
    names = {i: str(i) for i in range(80)}
    # Validation is off during training; Ultralytics still validates once at the
    # end, so --val may point at a small subset to keep that step short.
    val = str(args.val) if args.val else "images/val2017"
    YAML.save(data, {"path": str(args.corpus), "train": "images/val2017", "val": val, "names": names})
    started = time.perf_counter()
    model = YOLO("yolo11n-seg.yaml")
    model.train(
        trainer=Timed,
        data=str(data),
        epochs=args.epochs,
        close_mosaic=0,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=0,
        seed=912,
        deterministic=True,
        amp=args.amp,
        plots=False,
        val=False,
        pretrained=False,
        fraction=args.fraction,
        exist_ok=True,
        project=str(args.out.parent / "runs"),
        name=f"{args.backend}-w{args.workers}-r{args.repeat}",
        verbose=False,
    )
    whole = time.perf_counter() - started
    weights = hashlib.sha256()
    for key, tensor in sorted(model.trainer.model.state_dict().items()):
        weights.update(key.encode() + tensor.detach().cpu().contiguous().numpy().tobytes())
    losses = hashlib.sha256(json.dumps(record["losses"]).encode()).hexdigest()
    compose = model.trainer.train_loader.dataset.transforms
    report = {
        "backend": args.backend,
        "workers": args.workers,
        "repeat": args.repeat,
        "epochs_s": record["epochs"],
        "batches": record["batches"],
        "whole_job_s": whole,
        "loss_sha256": losses,
        "losses": record["losses"],
        "weights_sha256": weights.hexdigest(),
        "transforms": sorted({type(t).__name__ for t in compose.transforms}),
        "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    args.out.write_text(json.dumps(report))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--fraction", type=float, default=1.0, help="pilot only: fraction of the training images")
    parser.add_argument(
        "--val", type=Path, help="validation image directory for the final validation (default: the corpus)"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="train with Ultralytics' default mixed precision instead of the FP32 of the earlier GPU protocol; the AMP "
        "check loads yolo26n.pt from the weights directory, so download it beforehand when running offline",
    )
    parser.add_argument("--backend", choices=BACKENDS, help="internal: run one job")
    parser.add_argument("--repeat", type=int, default=0, help="internal")
    args = parser.parse_args()
    if args.backend:
        args.workers = args.workers[0]
        job(args)
        return
    run_dir = args.out.with_suffix(".runs")
    run_dir.mkdir(parents=True, exist_ok=False)
    jobs = []
    for workers in args.workers:
        for repeat in range(args.repeats):
            order = BACKENDS if repeat % 2 == 0 else BACKENDS[::-1]
            jobs += [(workers, repeat, backend) for backend in order]
    report = {
        "scope": "YOLO11n-seg from random weights on the COCO val2017 segmentation fixture; fresh process per job",
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version,
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "results": [],
    }
    print(
        f"{len(jobs)} jobs = {len(args.workers)} worker counts x {args.repeats} repeats x 2 backends, "
        f"{args.epochs} epochs each, {'AMP' if args.amp else 'FP32'}",
        flush=True,
    )
    t0, durations = time.time(), []
    for k, (workers, repeat, backend) in enumerate(jobs, 1):
        out = run_dir / f"w{workers}-r{repeat}-{backend}.json"
        print(f"[{k}/{len(jobs)}] workers={workers} repeat={repeat} {backend}", flush=True)
        ts = time.time()
        command = [sys.executable, __file__, "--corpus", str(args.corpus), "--out", str(out), "--backend", backend]
        command += ["--workers", str(workers), "--repeat", str(repeat), "--epochs", str(args.epochs)]
        command += ["--batch", str(args.batch), "--imgsz", str(args.imgsz), "--fraction", str(args.fraction)]
        command += ["--val", str(args.val)] if args.val else []
        command += ["--amp"] if args.amp else []
        subprocess.run(command, check=True)
        value = json.loads(out.read_text())
        value.pop("losses")  # kept in the per-job file; the hash is compared here
        report["results"].append(value)
        args.out.write_text(json.dumps(report, indent=2))
        durations.append(time.time() - ts)
        avg = statistics.mean(durations)
        eta = avg * (len(jobs) - k)
        print(
            f"    job {durations[-1]:.0f} s, epochs {[round(e, 1) for e in value['epochs_s']]}  elapsed {time.time() - t0:.0f} s"
            f"  avg {avg:.0f} s  ETA {eta / 60:.1f} min",
            flush=True,
        )
        if k == 1:
            print(f"  -> whole series about {avg * len(jobs) / 60:.0f} min", flush=True)
    summary = {}
    for workers in args.workers:
        mine = [r for r in report["results"] if r["workers"] == workers]
        losses = {r["loss_sha256"] for r in mine}
        weights = {r["weights_sha256"] for r in mine}
        entry = {"identical_losses": len(losses) == 1, "identical_weights": len(weights) == 1}
        for backend in BACKENDS:
            runs = [r for r in mine if r["backend"] == backend]
            entry[backend] = {
                "epoch_s_median": [statistics.median(r["epochs_s"][e] for r in runs) for e in range(args.epochs)],
                "whole_job_s_median": statistics.median(r["whole_job_s"] for r in runs),
            }
        entry["reference_over_fast"] = [
            entry["reference"]["epoch_s_median"][e] / entry["fast"]["epoch_s_median"][e] for e in range(args.epochs)
        ]
        summary[str(workers)] = entry
    report["summary"] = summary
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
