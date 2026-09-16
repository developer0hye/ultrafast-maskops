"""Ultralytics segmentation training throughput with the default configuration.

Fresh process per measurement: ``YOLO("yolo11n-seg.yaml")`` (random
initialization) trained with Ultralytics' defaults (batch 16, AMP, imgsz 640,
mosaic and every default augmentation) for one epoch, either with the stock
``SegmentationTrainer`` (reference) or with
``ultrafast_maskops.training.FastSegmentationTrainer`` (accelerated). The
batches both trainers produce are identical (tests/test_trainer.py).

Throughput is measured from batch --warmup to the last full batch with CUDA
synchronized at both marks. The run stops at the end of the first epoch by
raising from a callback, so validation, plots and checkpoints never run.

usage: ultralytics_training.py --data DATA.yaml --out report.json
       [--workers 2 4 8] [--rounds 2] [--batch 16] [--warmup 30] [--model yolo11n-seg.yaml]
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path


class Stop(Exception):
    """Raised by the callback at the end of the measured epoch."""


def worker(args):
    import torch
    from ultralytics import YOLO

    record = {"losses": []}
    state = {"batches": 0}

    def on_train_batch_end(trainer):
        state["batches"] += 1
        full = len(trainer.train_loader)  # batches per epoch
        if len(record["losses"]) < 3:
            # tloss is a dict of named running losses in this Ultralytics build and a tensor in others.
            tloss = getattr(trainer, "tloss", None)
            if isinstance(tloss, dict):
                record["losses"].append({k: float(v) for k, v in tloss.items()})
            elif tloss is not None:
                record["losses"].append([float(v) for v in torch.as_tensor(tloss).flatten()])
        if state["batches"] == args.warmup:
            torch.cuda.synchronize()
            record["t0"] = time.perf_counter()
            record["b0"] = state["batches"]
        if state["batches"] == full - 1:  # the last batch may be partial; stop at the last full one
            torch.cuda.synchronize()
            record["t1"] = time.perf_counter()
            record["b1"] = state["batches"]
            record["batch"] = trainer.batch_size
            record["amp"] = bool(trainer.amp)
            record["workers"] = trainer.train_loader.num_workers

    def on_train_epoch_end(trainer):
        raise Stop()

    model = YOLO(args.model)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    kwargs = {}
    if args.arm == "accelerated":
        from ultrafast_maskops.training import FastSegmentationTrainer

        kwargs["trainer"] = FastSegmentationTrainer
    load_before = os.getloadavg()
    torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory() as project:
        try:
            model.train(data=args.data, epochs=1, batch=args.batch, imgsz=640, workers=args.num_workers, device=0,
                        val=False, plots=False, seed=0, project=project, name="run", exist_ok=True, verbose=False,
                        **kwargs)
        except Stop:
            pass
    if "t1" not in record:
        raise RuntimeError("the epoch ended before the measured batches")
    images = (record["b1"] - record["b0"]) * record["batch"]
    elapsed = record["t1"] - record["t0"]
    return {
        "arm": args.arm,
        "workers": record["workers"],
        "batch": record["batch"],
        "amp": record["amp"],
        "images": images,
        "elapsed_s": elapsed,
        "img_per_s": images / elapsed,
        "first_losses": record["losses"],
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "loadavg_before": load_before,
        "loadavg_after": os.getloadavg(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", default="yolo11n-seg.yaml")
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--arm", choices=["reference", "accelerated"])
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.result:
        args.result.write_text(json.dumps(worker(args)))
        return
    if args.out is None:
        parser.error("--out required")
    run_dir = args.out.with_suffix(".runs")
    run_dir.mkdir(parents=True, exist_ok=True)
    import torch

    report = {
        "scope": "Ultralytics segmentation training img/s, default configuration, reference vs FastSegmentationTrainer",
        "host": platform.node(),
        "processor": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "results": [],
    }
    arms = ["reference", "accelerated"]
    for round_ in range(args.rounds):
        for workers in args.workers:
            for arm in arms if round_ % 2 == 0 else arms[::-1]:
                output = run_dir / f"{round_}-{workers}-{arm}.json"
                command = [sys.executable, __file__, "--data", args.data, "--model", args.model, "--arm", arm,
                           "--num-workers", str(workers), "--batch", str(args.batch), "--warmup", str(args.warmup),
                           "--result", str(output)]
                with output.with_suffix(".log").open("w") as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                value = json.loads(output.read_text())
                value["round"] = round_
                report["results"].append(value)
                print(f"round {round_} workers {workers:2d} {arm:11s} {value['img_per_s']:6.1f} img/s  amp {value['amp']}  "
                      f"load {value['loadavg_before'][0]:.2f}", flush=True)
                args.out.write_text(json.dumps(report, indent=1))
    summary = {}
    for workers in args.workers:
        row = {arm: max(r["img_per_s"] for r in report["results"] if r["workers"] == workers and r["arm"] == arm)
               for arm in arms}
        row["speedup"] = row["accelerated"] / row["reference"]
        summary[str(workers)] = row
    report["summary_best_img_per_s"] = summary
    args.out.write_text(json.dumps(report, indent=1))
    for workers, row in summary.items():
        print(f"workers {workers:>2}: " + "  ".join(f"{k} {v:.2f}" for k, v in row.items()))


if __name__ == "__main__":
    main()
