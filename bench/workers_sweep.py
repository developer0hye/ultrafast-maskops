"""DataLoader worker sweep for both frameworks, run one after the other.

RF-DETR: bench/rfdetr_workers.py (reference vs accelerated). Ultralytics:
bench/coco_epoch.py per worker count (reference vs maskops on the unmodified
YOLODataset). Each framework needs its own interpreter.

usage: workers_sweep.py --rfdetr-python P --ultralytics-python P
                        --rfdetr-images DIR --rfdetr-annotations JSON --ultralytics-images DIR
                        --out-dir DIR [--workers 0 1 2 4 6 8 12] [--rounds 2]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rfdetr-python", required=True)
    parser.add_argument("--ultralytics-python", required=True)
    parser.add_argument("--rfdetr-images", required=True)
    parser.add_argument("--rfdetr-annotations", required=True)
    parser.add_argument("--ultralytics-images", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 1, 2, 4, 6, 8, 12])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--skip-rfdetr", action="store_true")
    parser.add_argument("--skip-ultralytics", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("loadavg at start", os.getloadavg(), flush=True)

    if not args.skip_rfdetr:
        subprocess.run(
            [args.rfdetr_python, str(HERE / "rfdetr_workers.py"), args.rfdetr_images, args.rfdetr_annotations, "--out",
             str(args.out_dir / "rfdetr-workers.json"), "--workers", *map(str, args.workers), "--rounds", str(args.rounds)],
            check=True, stdout=sys.stdout, stderr=subprocess.STDOUT,
        )

    if not args.skip_ultralytics:
        summary = {}
        for workers in args.workers:
            out = args.out_dir / f"ultralytics-workers-{workers}.json"
            subprocess.run(
                [args.ultralytics_python, str(HERE / "coco_epoch.py"), "--images", args.ultralytics_images, "--out", str(out),
                 "--workers", str(workers), "--rounds", str(args.rounds), "--verify-batches", "16", "--backends", "reference", "maskops"],
                check=True, stdout=sys.stdout, stderr=subprocess.STDOUT, env={**os.environ, "YOLO_OFFLINE": "true"},
            )
            report = json.loads(out.read_text())
            row = {}
            for backend in ("reference", "maskops"):
                mine = [r for r in report["results"] if r["backend"] == backend]
                # Steady-state throughput: after the first batch, worker start-up excluded.
                row[backend] = max((r["samples"] - 16) / (r["epoch_s"] - r["first_batch_s"]) for r in mine)
            row["speedup"] = row["maskops"] / row["reference"]
            summary[str(workers)] = row
            print(f"ultralytics workers {workers:>2}: reference {row['reference']:6.1f}  maskops {row['maskops']:6.1f}  {row['speedup']:.2f}x", flush=True)
        (args.out_dir / "ultralytics-workers.json").write_text(json.dumps({"summary_best_steady_img_per_s": summary}, indent=1))


if __name__ == "__main__":
    main()
