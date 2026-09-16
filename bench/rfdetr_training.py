"""RF-DETR segmentation training throughput: does the faster loader shorten training?

Runs RF-DETR's own training entry point (``RFDETRSeg*.train``: PyTorch
Lightning, EMA, AMP, the default training configuration) for a fixed number of
optimizer steps, in a fresh process per measurement, with one of three
training datasets:

- reference: RF-DETR's CocoDetection as RF-DETR builds it;
- accelerated: the same dataset after ultrafast_maskops.rfdetr.accelerate_dataset;
- cached: 64 samples of the accelerated dataset prepared up front and served
  in a cycle, so the loader costs almost nothing: the GPU-bound ceiling.

The harness wraps ``rfdetr.training.module_data.build_dataset`` (to swap the
training dataset) and ``rfdetr.training.build_trainer`` (to add a timing
callback); RF-DETR itself is unmodified. Throughput is measured between step
--warmup and step --steps with CUDA synchronized at both marks. The run stops
by raising from the callback, so validation never runs. The first losses are
recorded so the arms can be checked to train on the same batches.

usage: rfdetr_training.py --dataset-dir COCO_ROOT --out report.json
       [--model seg-small] [--workers 2 4 8] [--arms reference accelerated cached]
       [--rounds 2] [--steps 300] [--warmup 40] [--batch 4]
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

MODELS = {"seg-nano": "RFDETRSegNano", "seg-small": "RFDETRSegSmall", "seg-medium": "RFDETRSegMedium"}


class Stop(Exception):
    """Raised by the timing callback once the measured steps are done."""


def worker(args):
    import torch

    try:
        import pytorch_lightning as pl
    except ImportError:  # pragma: no cover - the lightning namespace
        import lightning.pytorch as pl
    import rfdetr
    import rfdetr.training as rf_training
    import rfdetr.training.module_data as module_data

    from ultrafast_maskops.rfdetr import accelerate_dataset

    class Cached(torch.utils.data.Dataset):
        def __init__(self, base, count):
            self.base = base
            self.items = []
            for i in range(count):
                torch.manual_seed(i)
                self.items.append(base[i])

        def __len__(self):
            return len(self.base)

        def __getitem__(self, index):
            image, target = self.items[index % len(self.items)]
            return image.clone(), {k: (v.clone() if torch.is_tensor(v) else v) for k, v in target.items()}

        def __getattr__(self, name):
            if name in ("base", "items"):
                raise AttributeError(name)
            return getattr(self.base, name)

    original_build = module_data.build_dataset

    def build_dataset(image_set, ns, resolution):
        dataset = original_build(image_set, ns, resolution)
        if image_set == "train" and args.arm in ("accelerated", "cached"):
            accelerate_dataset(dataset)
            if args.arm == "cached":
                dataset = Cached(dataset, 64)
        return dataset

    module_data.build_dataset = build_dataset

    record = {"losses": []}

    class Timer(pl.Callback):
        def __init__(self):
            self.batches = 0
            self.images = 0

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            self.batches += 1
            if len(record["losses"]) < 5 and isinstance(outputs, dict) and "loss" in outputs:
                record["losses"].append(float(outputs["loss"]))
            if self.batches == args.warmup:
                torch.cuda.synchronize()
                record["t0"] = time.perf_counter()
            elif self.batches > args.warmup:
                self.images += len(batch[1])
            if self.batches == args.steps:
                torch.cuda.synchronize()
                record["t1"] = time.perf_counter()
                record["images"] = self.images
                raise Stop()

    original_trainer = rf_training.build_trainer

    def build_trainer(*a, **k):
        trainer = original_trainer(*a, **k)
        trainer.callbacks.append(Timer())
        return trainer

    rf_training.build_trainer = build_trainer

    load_before = os.getloadavg()
    model = getattr(rfdetr, MODELS[args.model])()
    torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory() as output_dir:
        try:
            model.train(
                dataset_file="coco",
                dataset_dir=args.dataset_dir,
                output_dir=output_dir,
                epochs=1,
                batch_size=args.batch,
                grad_accum_steps=1,
                num_workers=args.num_workers,
                tensorboard=False,
                progress_bar=None,
                seed=0,
                device="cuda",
            )
        except Stop:
            pass
        except Exception as exc:  # Lightning re-raises the callback's exception, possibly wrapped
            if "Stop" not in type(exc).__name__ and not isinstance(exc.__cause__, Stop):
                raise
    if "t1" not in record:
        raise RuntimeError("training ended before the measured steps")
    elapsed = record["t1"] - record["t0"]
    return {
        "arm": args.arm,
        "workers": args.num_workers,
        "model": args.model,
        "batch": args.batch,
        "steps": args.steps,
        "warmup": args.warmup,
        "images": record["images"],
        "elapsed_s": elapsed,
        "img_per_s": record["images"] / elapsed,
        "first_losses": record["losses"],
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "loadavg_before": load_before,
        "loadavg_after": os.getloadavg(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, help="COCO root with train2017/ and annotations/instances_train2017.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", choices=sorted(MODELS), default="seg-small")
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--arms", nargs="+", choices=["reference", "accelerated", "cached"],
                        default=["reference", "accelerated", "cached"])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--arm")
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
        "scope": "RF-DETR segmentation training steps/s with the reference, accelerated and cached training datasets",
        "host": platform.node(),
        "processor": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "results": [],
    }
    for round_ in range(args.rounds):
        for workers in args.workers:
            arms = args.arms if round_ % 2 == 0 else args.arms[::-1]
            for arm in arms:
                output = run_dir / f"{round_}-{workers}-{arm}.json"
                command = [sys.executable, __file__, "--dataset-dir", args.dataset_dir, "--model", args.model,
                           "--arm", arm, "--num-workers", str(workers), "--steps", str(args.steps),
                           "--warmup", str(args.warmup), "--batch", str(args.batch), "--result", str(output)]
                with output.with_suffix(".log").open("w") as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                value = json.loads(output.read_text())
                value["round"] = round_
                report["results"].append(value)
                print(f"round {round_} workers {workers:2d} {arm:11s} {value['img_per_s']:6.1f} img/s  "
                      f"loss0 {value['first_losses'][:1]}  load {value['loadavg_before'][0]:.2f}", flush=True)
                args.out.write_text(json.dumps(report, indent=1))
    summary = {}
    for workers in args.workers:
        row = {}
        for arm in args.arms:
            mine = [r["img_per_s"] for r in report["results"] if r["workers"] == workers and r["arm"] == arm]
            row[arm] = max(mine)
        if "reference" in row and "accelerated" in row:
            row["speedup"] = row["accelerated"] / row["reference"]
        summary[str(workers)] = row
    report["summary_best_img_per_s"] = summary
    args.out.write_text(json.dumps(report, indent=1))
    for workers, row in summary.items():
        print(f"workers {workers:>2}: " + "  ".join(f"{k} {v:.2f}" for k, v in row.items()))


if __name__ == "__main__":
    main()
