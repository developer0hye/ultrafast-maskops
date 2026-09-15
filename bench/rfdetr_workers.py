"""RF-DETR DataLoader throughput against num_workers: RF-DETR's own dataset vs the adapter.

Every measurement is a fresh process (arm, workers, round) that builds the
dataset with RF-DETR's default training transforms, iterates one pass of the
split through a DataLoader with RF-DETR's collate, and reports images per
second after the first batch (worker start-up excluded) and over the whole
pass. Arms alternate within each round.

usage: rfdetr_workers.py <images_dir> <instances_json> --out report.json
                         [--workers 0 1 2 4 6 8 12] [--rounds 2] [--batch 8] [--resolution 560]
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def build(args, accelerate):
    import torch
    from rfdetr.datasets import coco as rf_coco

    transforms = rf_coco.make_coco_transforms_square_div_64("train", args.resolution, multi_scale=True)
    dataset = rf_coco.CocoDetection(args.images, args.annotations, transforms, include_masks=True, remap_category_ids=True)
    if accelerate:
        from ultrafast_maskops.rfdetr import accelerate_dataset

        accelerate_dataset(dataset)
    torch.set_num_threads(1)
    return dataset


def worker(args):
    import torch
    from rfdetr.utilities.tensors import collate_fn
    from torch.utils.data import DataLoader

    load_before = os.getloadavg()
    started = time.perf_counter()
    dataset = build(args, args.arm == "accelerated")
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(
        dataset, batch_size=args.batch, num_workers=args.num_workers, shuffle=True, generator=generator, collate_fn=collate_fn
    )
    constructed = time.perf_counter()
    samples = first = first_samples = None
    samples = 0
    for batch in loader:
        n = len(batch[1])
        if first is None:
            first, first_samples = time.perf_counter(), n
        samples += n
    finished = time.perf_counter()
    return {
        "arm": args.arm,
        "workers": args.num_workers,
        "batch": args.batch,
        "samples": samples,
        "constructor_s": constructed - started,
        "first_batch_s": first - constructed,
        "pass_s": finished - constructed,
        "img_per_s_total": samples / (finished - constructed),
        "img_per_s_steady": (samples - first_samples) / (finished - first),
        "loadavg_before": load_before,
        "loadavg_after": os.getloadavg(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images")
    parser.add_argument("annotations")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 1, 2, 4, 6, 8, 12])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=560)
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
        "scope": "one pass over the split, RF-DETR default training transforms, fresh process per measurement",
        "host": platform.node(),
        "processor": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "results": [],
    }
    arms = ["reference", "accelerated"]
    for round_ in range(args.rounds):
        for workers in args.workers:
            for arm in arms if round_ % 2 == 0 else arms[::-1]:
                output = run_dir / f"{round_}-{workers}-{arm}.json"
                command = [sys.executable, __file__, args.images, args.annotations, "--arm", arm, "--num-workers", str(workers),
                           "--batch", str(args.batch), "--resolution", str(args.resolution), "--result", str(output)]
                with output.with_suffix(".log").open("w") as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                value = json.loads(output.read_text())
                value["round"] = round_
                report["results"].append(value)
                print(f"round {round_} workers {workers:2d} {arm:11s} {value['img_per_s_steady']:6.1f} img/s steady, "
                      f"{value['img_per_s_total']:6.1f} total, load {value['loadavg_before'][0]:.2f}", flush=True)
                args.out.write_text(json.dumps(report, indent=1))
    summary = {}
    for workers in args.workers:
        summary[str(workers)] = {}
        for arm in arms:
            mine = [r for r in report["results"] if r["workers"] == workers and r["arm"] == arm]
            summary[str(workers)][arm] = max(r["img_per_s_steady"] for r in mine)
        summary[str(workers)]["speedup"] = summary[str(workers)]["accelerated"] / summary[str(workers)]["reference"]
    report["summary_best_steady_img_per_s"] = summary
    args.out.write_text(json.dumps(report, indent=1))
    for workers, row in summary.items():
        print(f"workers {workers:>2}: reference {row['reference']:6.1f}  accelerated {row['accelerated']:6.1f}  {row['speedup']:.2f}x")


if __name__ == "__main__":
    main()
