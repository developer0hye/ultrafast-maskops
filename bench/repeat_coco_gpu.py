"""Counterbalanced fresh-process GPU training comparisons; retain every trial.

Each repetition contains reference, mask-only, and combined replacement runs.
The default is 45 full training jobs (three worker counts, five repetitions,
three backends), with two epochs per job. Preparation pilots are excluded.
"""

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ratio_summary(pairs):
    reference, native = zip(*pairs)
    rng = random.Random(912)
    bootstrap = []
    for _ in range(10000):
        selected = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        bootstrap.append(statistics.median(p[0] for p in selected) / statistics.median(p[1] for p in selected))
    bootstrap.sort()
    return {
        "pairs": len(pairs),
        "reference_median_s": statistics.median(reference),
        "candidate_median_s": statistics.median(native),
        "reference_over_candidate_median_ratio": statistics.median(reference) / statistics.median(native),
        "paired_bootstrap_95_percent_interval": [bootstrap[249], bootstrap[9749]],
        "bootstrap": "10000 paired resamples, seed 912, nearest-rank 2.5/97.5 percentiles; five pairs give limited precision",
    }


def summarize(records, workers, repeats, epochs):
    results = {}
    for worker in workers:
        for backend in ("mask", "both"):
            for epoch in range(epochs):
                pairs = []
                for repeat in range(repeats):
                    group = {
                        r["backend"]: r["report"] for r in records if r["workers"] == worker and r["repeat"] == repeat
                    }
                    pairs.append(
                        (group["reference"]["epochs"][epoch]["seconds"], group[backend]["epochs"][epoch]["seconds"])
                    )
                results[f"workers={worker}/{backend}/epoch={epoch}"] = ratio_summary(pairs)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--fresh-check", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", choices=(0, 2, 8), default=[0, 2, 8])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--overlap", choices=("yes", "no"), default="yes")
    args = parser.parse_args()
    assert args.repeats >= 5 and args.epochs >= 2
    assert len(args.workers) == len(set(args.workers))
    args.out = args.out.resolve()
    directory = args.out.with_suffix(".runs")
    assert not args.out.exists() and not directory.exists(), "retain prior results; choose a new series output"
    directory.mkdir(parents=True)
    script = Path(__file__).with_name("train_coco_gpu.py")
    script_sha = file_sha(script)
    records = []
    report = {
        "complete": False,
        "scope": "repeated full real-data GPU training, not accuracy",
        "script_sha256": file_sha(__file__),
        "trial_script_sha256": script_sha,
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "ordering": "rotate backend order across repetitions and worker groups; reverse alternating repetitions",
        "first_epoch": "reported separately; includes warmup and cannot be pooled with epoch 1",
        "startup": "original annotation cache is warm; native content cache starts cold per job; do not interpret total-job timing as an equivalent cache-hit benchmark",
        "records": records,
    }

    def save():
        args.out.write_text(json.dumps(report, indent=2) + "\n")

    save()
    identity = None
    try:
        for repeat in range(args.repeats):
            for index, worker in enumerate(args.workers):
                order = ["reference", "mask", "both"]
                rotate = (repeat + index) % len(order)
                order = order[rotate:] + order[:rotate]
                if repeat % 2:
                    order.reverse()
                for backend in order:
                    assert file_sha(script) == script_sha, "trial script changed during the series"
                    path = directory / f"r{repeat}-w{worker}-{backend}.json"
                    command = [
                        sys.executable,
                        str(script),
                        "--corpus",
                        str(args.corpus.resolve()),
                        "--fresh-check",
                        str(args.fresh_check.resolve()),
                        "--out",
                        str(path),
                        "--backend",
                        backend,
                        "--workers",
                        str(worker),
                        "--epochs",
                        str(args.epochs),
                        "--batch",
                        str(args.batch),
                        "--imgsz",
                        str(args.imgsz),
                        "--overlap",
                        args.overlap,
                    ]
                    print(f"Starting repeat={repeat} workers={worker} backend={backend}", flush=True)
                    with path.with_suffix(".log").open("w") as log:
                        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                    value = json.loads(path.read_text())
                    assert value["complete"] and value["script_sha256"] == script_sha
                    assert len(value["epochs"]) == args.epochs
                    profile = {
                        k: value[k]
                        for k in (
                            "upstream_files",
                            "packages",
                            "fixture",
                            "torch",
                            "numpy",
                            "opencv",
                            "gpu",
                            "cuda",
                            "initial_state_sha256",
                            "original_cache_sha256",
                            "fresh_check_sha256",
                        )
                    }
                    if identity is None:
                        identity = profile
                    assert profile == identity, "inputs, runtime, GPU, or initial weights changed"
                    # Keep raw traces in their hashed files. Retaining every loss
                    # tensor value/resource sample here would make the parent
                    # progressively consume more memory between paired trials.
                    compact = {k: v for k, v in value.items() if k not in ("resource_samples", "epochs")}
                    compact["epochs"] = [
                        {k: v for k, v in epoch.items() if k != "all_batch_losses"} for epoch in value["epochs"]
                    ]
                    records.append(
                        {
                            "repeat": repeat,
                            "workers": worker,
                            "backend": backend,
                            "command": command,
                            "path": str(path),
                            "sha256": file_sha(path),
                            "report": compact,
                        }
                    )
                    save()
                    print(f"Completed {len(records)}/{args.repeats * len(args.workers) * 3} trials", flush=True)
        report["summary"] = summarize(records, args.workers, args.repeats, args.epochs)
        report["complete"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
