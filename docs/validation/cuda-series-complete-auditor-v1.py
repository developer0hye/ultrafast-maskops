"""Independently audit frozen GPU series artifacts without importing Torch.

Partial snapshots require --allow-partial and never receive a timing summary.
Use --runs-dir to audit relocated raw reports by their validated trial names.
The audit does not rerun training or establish accuracy, unsampled RSS peaks,
or target parity beyond the separate full-target verification experiments.
"""

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from itertools import pairwise
from pathlib import Path

BACKENDS = ("reference", "mask", "both")
IDENTITY = (
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
    "python",
    "platform",
    "cpu_count",
    "ram_bytes",
    "mask_build",
    "timing",
    "memory",
    "callbacks",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def compact(raw):
    result = {k: v for k, v in raw.items() if k not in ("epochs", "resource_samples")}
    result["epochs"] = [{k: v for k, v in epoch.items() if k != "all_batch_losses"} for epoch in raw["epochs"]]
    return result


def expected_order(config):
    result = []
    for repeat in range(config["repeats"]):
        for position, workers in enumerate(config["workers"]):
            start = (repeat + position) % 3
            backends = list(BACKENDS[start:] + BACKENDS[:start])
            if repeat % 2:
                backends.reverse()
            result.extend((repeat, workers, backend) for backend in backends)
    return result


def paired_summary(pairs):
    # Independent implementation of the documented paired percentile bootstrap.
    reference = statistics.median(pair[0] for pair in pairs)
    candidate = statistics.median(pair[1] for pair in pairs)
    randomizer = random.Random(912)
    estimates = []
    for _ in range(10000):
        positions = [randomizer.randrange(len(pairs)) for _ in pairs]
        numerator = statistics.median(pairs[index][0] for index in positions)
        denominator = statistics.median(pairs[index][1] for index in positions)
        estimates.append(numerator / denominator)
    estimates.sort()
    return {
        "pairs": len(pairs),
        "reference_median_s": reference,
        "candidate_median_s": candidate,
        "reference_over_candidate_median_ratio": reference / candidate,
        "paired_bootstrap_95_percent_interval": [estimates[249], estimates[9749]],
    }


def audit(series_path, runs_dir, parent_script, trial_script, allow_partial):
    series_bytes = series_path.read_bytes()
    series = json.loads(series_bytes)
    require(series["script_sha256"] == sha(parent_script.read_bytes()), "parent script hash mismatch")
    require(series["trial_script_sha256"] == sha(trial_script.read_bytes()), "trial script hash mismatch")
    config = series["configuration"]
    require(config["repeats"] >= 5 and config["epochs"] >= 2, "insufficient planned repetition/epoch count")
    require(config["workers"] and len(config["workers"]) == len(set(config["workers"])), "duplicate/empty workers")
    require(set(config["workers"]) <= {0, 2, 8}, "unsupported worker configuration")
    require(config["batch"] == 4 and config["imgsz"] == 640, "different batch/image protocol")
    require(config["overlap"] in ("yes", "no"), "invalid overlap setting")
    planned = expected_order(config)
    records = series["records"]
    actual = [(r["repeat"], r["workers"], r["backend"]) for r in records]
    require(actual == planned[: len(actual)], "missing, duplicate, extra, or reordered records")
    require(records, "no completed trials to audit")
    complete = series.get("complete") is True
    require(allow_partial or complete, "series incomplete; use --allow-partial only for an interim audit")
    require(not complete or len(actual) == len(planned), "complete flag with missing trials")
    require(not complete or "error" not in series, "complete series also reports an error")
    identity, resolved = None, None
    raw_by_key, validated = {}, []
    for record in records:
        key = (record["repeat"], record["workers"], record["backend"])
        name = f"r{key[0]}-w{key[1]}-{key[2]}.json"
        require(Path(record["path"]).name == name, f"trial filename mismatch: {key}")
        path = runs_dir / name if runs_dir else Path(record["path"])
        data = path.read_bytes()
        require(sha(data) == record["sha256"], f"raw checksum mismatch: {name}")
        raw = json.loads(data)
        require(canonical(compact(raw)) == canonical(record["report"]), f"compact/raw disagreement: {name}")
        require(raw["complete"] is True and "error" not in raw, f"incomplete raw trial: {name}")
        require(raw["script_sha256"] == series["trial_script_sha256"], f"trial code drift: {name}")
        current_identity = {k: raw[k] for k in IDENTITY}
        if identity is None:
            identity = current_identity
        require(current_identity == identity, f"runtime/input/measurement identity drift: {name}")
        require(
            raw["fixture"]
            == {
                "files": 9952,
                "bytes": 830426850,
                "sha256": "8b079ad2d8472ff17b6cca9b0250e831472f12ef500f2b065125f6f39406c208",
            },
            f"different corpus: {name}",
        )
        require(raw["initial_state_sha256"] != raw["final_state_sha256"], f"no model update: {name}")
        # Pinned build_dataloader caps requested workers by CPU count on this
        # single-GPU host. At workers=8, validation requests 16 but receives 12.
        expected_val_workers = min(2 * key[1], raw["cpu_count"])
        require(
            raw["train_workers"] == key[1] and raw["val_workers"] == expected_val_workers,
            f"worker drift: {name}",
        )
        for k in ("batch", "imgsz", "epochs", "overlap", "corpus", "fresh_check"):
            require(raw["args"][k] == config[k], f"trial argument drift {k}: {name}")
        require(raw["args"]["backend"] == key[2] and raw["args"]["workers"] == key[1], f"backend drift: {name}")
        require(raw["args"]["out"] == record["path"], f"output argument mismatch: {name}")
        command = record["command"]
        require(len(command) == 20 and Path(command[1]).name == trial_script.name, f"unexpected command: {name}")
        command_args = dict(zip(command[2::2], command[3::2]))
        expected_args = {"--" + k.replace("_", "-"): str(v) for k, v in raw["args"].items()}
        require(command_args == expected_args, f"command/raw argument mismatch: {name}")
        current_config = {
            k: v for k, v in raw["resolved_config"].items() if k not in ("workers", "data", "project", "save_dir")
        }
        if resolved is None:
            resolved = current_config
        require(current_config == resolved, f"resolved training configuration drift: {name}")
        for k, value in {
            "model": "yolo11n-seg.yaml",
            "seed": 912,
            "amp": False,
            "optimizer": "SGD",
            "pretrained": False,
            "deterministic": True,
            "close_mosaic": 0,
            "mask_ratio": 4,
            "overlap_mask": config["overlap"] == "yes",
            "batch": config["batch"],
            "epochs": config["epochs"],
            "imgsz": config["imgsz"],
            "lr0": 0.001,
            "cache": False,
            "rect": False,
            "fraction": 1.0,
        }.items():
            require(current_config[k] == value, f"unexpected training setting {k}: {name}")
        constructors = raw["datasets"]
        require([d["mode"] for d in constructors] == ["train", "val"], f"missing dataset construction: {name}")
        for constructor in constructors:
            require(constructor["images"] == 5000 and positive(constructor["seconds"]), f"bad constructor: {name}")
            require(constructor["replaced_formats"] == int(key[2] != "reference"), f"replacement not active: {name}")
            expected_hit = False if key[2] == "both" else None
            require(constructor["native_cache_hit"] is expected_hit, f"cache temperature drift: {name}")
        require(len(raw["epochs"]) == config["epochs"], f"epoch count mismatch: {name}")
        losses = []
        for number, epoch in enumerate(raw["epochs"]):
            require(epoch["epoch"] == number, f"missing/duplicate epoch: {name}")
            require(epoch["images"] == 5000 and epoch["batches"] == 1250, f"partial epoch: {name}")
            require(positive(epoch["seconds"]), f"invalid time: {name}")
            require(
                math.isclose(epoch["images_per_second"], 5000 / epoch["seconds"], rel_tol=1e-12),
                f"bad throughput: {name}",
            )
            names = epoch["loss_names"]
            require(names == ["box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"], f"loss schema drift: {name}")
            vectors = epoch["all_batch_losses"]
            require(len(vectors) == 1250, f"loss trace truncated: {name}")
            for vector in vectors:
                require(len(vector) == len(names), f"loss width mismatch: {name}")
                require(all(type(v) in (int, float) and math.isfinite(v) for v in vector), f"nonfinite loss: {name}")
            require(sum(v[names.index("seg_loss")] for v in vectors) > 0, f"no segmentation loss: {name}")
            allocated, reserved = epoch["cuda_peak_allocated_bytes"], epoch["cuda_peak_reserved_bytes"]
            require(
                type(allocated) is int and type(reserved) is int and 0 < allocated <= reserved,
                f"bad CUDA peaks: {name}",
            )
            losses.append(sha(canonical({"names": names, "values": vectors})))
        require(
            positive(raw["whole_job_seconds"]) and raw["whole_job_seconds"] >= sum(e["seconds"] for e in raw["epochs"]),
            f"invalid whole-job time: {name}",
        )
        samples = raw["resource_samples"]
        require(len(samples) >= 2, f"missing RSS trace: {name}")
        for sample in samples:
            require(
                positive(sample["time"])
                and type(sample["family_rss_bytes"]) is int
                and sample["family_rss_bytes"] >= 0,
                f"bad RSS sample: {name}",
            )
            require(
                len(sample["load"]) == 3
                and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in sample["load"]),
                f"invalid load sample: {name}",
            )
        peak = max(s["family_rss_bytes"] for s in samples)
        require(peak > 0 and peak == raw["sampled_peak_family_rss_bytes"], f"RSS maximum mismatch: {name}")
        gaps = [b["time"] - a["time"] for a, b in pairwise(samples)]
        validated.append(
            {
                "repeat": key[0],
                "workers": key[1],
                "backend": key[2],
                "raw_sha256": record["sha256"],
                "epoch_loss_trace_sha256": losses,
                "final_state_sha256": raw["final_state_sha256"],
                "sampled_whole_job_family_rss_bytes": peak,
                "rss_sample_count": len(samples),
                "rss_largest_sampling_gap_s": max(gaps),
                "rss_wall_clock_went_backwards": min(gaps) < 0,
            }
        )
        raw_by_key[key] = raw
    groups, pending = [], []
    for repeat in range(config["repeats"]):
        for workers in config["workers"]:
            present = [b for b in BACKENDS if (repeat, workers, b) in raw_by_key]
            if len(present) != 3:
                pending.append({"repeat": repeat, "workers": workers, "present_backends": present})
                continue
            group = [raw_by_key[repeat, workers, b] for b in BACKENDS]
            for candidate in group[1:]:
                require(
                    candidate["final_state_sha256"] == group[0]["final_state_sha256"],
                    f"final model differs: r{repeat}/w{workers}",
                )
                for reference_epoch, candidate_epoch in zip(group[0]["epochs"], candidate["epochs"]):
                    require(
                        canonical(reference_epoch["all_batch_losses"])
                        == canonical(candidate_epoch["all_batch_losses"]),
                        f"loss trace differs: r{repeat}/w{workers}/epoch{reference_epoch['epoch']}",
                    )
            groups.append(
                {
                    "repeat": repeat,
                    "workers": workers,
                    "epochs": config["epochs"],
                    "all_losses_and_final_model_equal": True,
                }
            )
    checked_summary = None
    if complete:
        require(not pending, "complete series with missing backend groups")
        checked_summary = {}
        for workers in config["workers"]:
            for backend in BACKENDS[1:]:
                for epoch in range(config["epochs"]):
                    pairs = [
                        (
                            raw_by_key[repeat, workers, "reference"]["epochs"][epoch]["seconds"],
                            raw_by_key[repeat, workers, backend]["epochs"][epoch]["seconds"],
                        )
                        for repeat in range(config["repeats"])
                    ]
                    name = f"workers={workers}/{backend}/epoch={epoch}"
                    result = paired_summary(pairs)
                    for k, value in result.items():
                        require(series["summary"][name][k] == value, f"paired summary mismatch: {name}/{k}")
                    checked_summary[name] = result
        require(set(checked_summary) == set(series["summary"]), "extra/missing summary groups")
    return {
        "audit_passed": True,
        "series_complete": complete,
        "series_recorded_error": series.get("error"),
        "series_sha256": sha(series_bytes),
        "auditor_sha256": sha(Path(__file__).read_bytes()),
        "validated_trials": len(validated),
        "planned_trials": len(planned),
        "configuration": config,
        "trial_checks": validated,
        "complete_backend_groups": groups,
        "pending_backend_groups": pending,
        "independently_recomputed_epoch_summary": checked_summary,
        "scope": "raw/compact/command identity, complete epochs, exact loss traces and final models within worker/repetition groups, sampled RSS and CUDA allocator accounting; timing summaries only when complete",
        "limits": [
            "Sampled whole-job summed RSS includes initialization and validation, double-counts shared pages and misses between-sample spikes.",
            "CUDA allocator peaks are per epoch, not total device or whole-job GPU memory.",
            "Equal losses and final model hashes do not replace direct target parity or establish accuracy.",
            "Five paired repetitions on a shared host do not establish a general speedup.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--parent-script", type=Path, default=Path(__file__).with_name("repeat_coco_gpu.py"))
    parser.add_argument("--trial-script", type=Path, default=Path(__file__).with_name("train_coco_gpu.py"))
    args = parser.parse_args()
    require(not args.out.exists(), "retain existing audit reports")
    try:
        result = audit(args.series, args.runs_dir, args.parent_script, args.trial_script, args.allow_partial)
    except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
        result = {"audit_passed": False, "error": str(error), "auditor_sha256": sha(Path(__file__).read_bytes())}
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), file=sys.stderr)
        raise SystemExit(1) from error
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("audit_passed", "series_complete", "validated_trials", "planned_trials")}))


if __name__ == "__main__":
    main()
