"""Independently audit the complete 54-trial lifecycle artifact graph.

Standard library only. Never imports the coordinator, Torch, or pickle. Partial
series are rejected. Relocated artifacts retain and validate original path names.
Recorded loss/model equality is not held-out accuracy or uninterrupted/resumed
training equivalence. Checkpoint bytes and recorded restoration counts are checked,
not independently deserialized optimizer tensors.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

BACKENDS = ("reference", "mask", "both")
STAGES = (("fresh", 0), ("resume-at-boundary", 2), ("resume-after-boundary", 3))
FIXTURE = {
    "files": 9952,
    "bytes": 830426850,
    "sha256": "8b079ad2d8472ff17b6cca9b0250e831472f12ef500f2b065125f6f39406c208",
}
IDENTITY = (
    "harness_sources",
    "upstream_files",
    "packages",
    "fixture",
    "corpus_root",
    "original_cache_sha256",
    "fresh_check_sha256",
    "python",
    "platform",
    "torch",
    "numpy",
    "opencv",
    "gpu",
    "cuda",
    "cpu_count",
    "ram_bytes",
)


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def load(path):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=constant)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def exact(actual, expected, reason):
    need(type(actual) is type(expected) and actual == expected, reason)


def ids(values, count):
    need(
        type(values) is list
        and len(values) == count
        and len(set(values)) == count
        and all(type(p) is int and p > 0 for p in values),
        "worker PID count/type/uniqueness",
    )
    return values


def exited(pids, codes, alive, count):
    ids(pids, count)
    need(
        type(codes) is list and len(codes) == count and all(type(c) is int and c == 0 for c in codes),
        "worker exit code",
    )
    need(type(alive) is list and len(alive) == count and all(a is False for a in alive), "worker still alive")


def file_binding(path, recorded, expected_name):
    need(Path(recorded["path"]).name == expected_name, "artifact filename")
    exact(path.stat().st_size, recorded["bytes"], "checkpoint size")
    need(sha(path) == recorded["sha256"], "checkpoint checksum")


def trial(raw, spec, record, runs, harness, corpus, fresh_sha, cache_sha, reference):
    workers, overlap, start = spec["workers"], spec["overlap"], spec["start_epoch"]
    original = Path(record["path"])
    need(
        raw["complete"] is True and raw["lifecycle_requested"] is True and "error" not in raw, "incomplete/error trial"
    )
    need(raw["harness_sources"] == harness and raw["script_sha256"] == harness["train_coco_gpu.py"], "trial source")
    need(raw["fixture"] == FIXTURE and raw["corpus_root"] == corpus, "fixture identity")
    need(raw["fresh_check_sha256"] == fresh_sha and raw["original_cache_sha256"] == cache_sha, "cache/receipt identity")
    args = raw["args"]
    for key, expected in {
        "workers": workers,
        "epochs": 4,
        "batch": 4,
        "imgsz": 640,
        "close_mosaic": 2,
        "backend": spec["backend"],
        "overlap": overlap,
        "persistent_mask": True,
        "out": str(original),
    }.items():
        exact(args[key], expected, "trial argument " + key)
    for key, expected in {
        "epochs": 4,
        "batch": 4,
        "imgsz": 640,
        "workers": workers,
        "seed": 912,
        "deterministic": True,
        "amp": False,
        "optimizer": "SGD",
        "lr0": 0.001,
        "cache": False,
        "rect": False,
        "fraction": 1.0,
        "mask_ratio": 4,
        "overlap_mask": overlap == "yes",
        "close_mosaic": 2,
    }.items():
        exact(raw["resolved_config"][key], expected, "resolved setting " + key)
    expected_epochs = list(range(start, 4))
    need([e["epoch"] for e in raw["epochs"]] == expected_epochs, "epoch order/count")
    loss_hashes, compact_epochs = [], []
    for epoch in raw["epochs"]:
        need(type(epoch["epoch"]) is int, "epoch type")
        exact(epoch["images"], 5000, "epoch image count")
        exact(epoch["batches"], 1250, "epoch batch count")
        need(positive(epoch["seconds"]) and positive(epoch["images_per_second"]), "epoch timing")
        need(math.isclose(epoch["images_per_second"], 5000 / epoch["seconds"], rel_tol=1e-12), "throughput arithmetic")
        names, vectors = epoch["loss_names"], epoch["all_batch_losses"]
        need(
            names and all(type(n) is str for n in names) and len(set(names)) == len(names) and "seg_loss" in names,
            "loss names",
        )
        need(
            len(vectors) == 1250
            and all(
                len(v) == len(names) and all(type(x) in (int, float) and math.isfinite(x) for x in v) for v in vectors
            ),
            "loss trace shape/finite values",
        )
        need(sum(v[names.index("seg_loss")] for v in vectors) > 0, "foreground loss")
        a, r = epoch["cuda_peak_allocated_bytes"], epoch["cuda_peak_reserved_bytes"]
        need(type(a) is int and type(r) is int and 0 < a <= r, "CUDA peak arithmetic")
        loss_hashes.append(digest({"names": names, "values": vectors}))
        compact_epochs.append(
            {**{k: v for k, v in epoch.items() if k != "all_batch_losses"}, "loss_sha256": loss_hashes[-1]}
        )
    for key in ("initial_state_sha256", "final_state_sha256"):
        value = raw[key]
        need(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "model digest")
    need(raw["initial_state_sha256"] != raw["final_state_sha256"], "model did not update")
    need(
        positive(raw["whole_job_seconds"])
        and sum(e["seconds"] for e in raw["epochs"]) <= raw["whole_job_seconds"] * (1 + 1e-9),
        "job duration",
    )
    need(len(raw["datasets"]) == 2 and {d["mode"] for d in raw["datasets"]} == {"train", "val"}, "dataset construction")
    for dataset in raw["datasets"]:
        exact(dataset["images"], 5000, "dataset length")
        exact(dataset["replaced_formats"], int(spec["backend"] != "reference"), "dataset replacement")
    exact(raw["train_workers"], workers, "train worker count")
    need(type(raw["cpu_count"]) is int and raw["cpu_count"] >= 8, "host CPU count")
    exact(raw["val_workers"], min(2 * workers, raw["cpu_count"]), "actual validation worker count")
    shutdown = raw["worker_shutdown"]
    need(set(shutdown) == {"train", "val"}, "shutdown loader set")
    for name, count in (("train", workers), ("val", raw["val_workers"])):
        item = shutdown[name]
        exact(item["worker_count"], count, "shutdown worker count")
        exited(item["worker_pids"], item["worker_exitcodes"], item["workers_alive"], count)
    need(not set(shutdown["train"]["worker_pids"]) & set(shutdown["val"]["worker_pids"]), "shared train/val worker PID")
    rows = raw["lifecycle"]
    need(len(rows) == 1 + len(expected_epochs), "lifecycle observation count")
    wanted = "Format" if spec["backend"] == "reference" else "FastFormat"
    collator = (
        "ultralytics.data.dataset.YOLODataset.collate_fn"
        if wanted == "Format"
        else "ultrafast_maskops._shared_collate.shared_collate_fn"
    )
    settings = []
    for position, row in enumerate(rows):
        exact(row["start_epoch"], start, "lifecycle start epoch")
        exact(row["stage"], "prepared" if position == 0 else "first-batch", "lifecycle stage")
        closed = start > 2 if position == 0 else expected_epochs[position - 1] >= 2
        exact(row["rebuilt_from_initial"], closed, "transform rebuild")
        need(row["formatter"] == row["factory"] == wanted and row["collator"] == collator, "formatter/factory/collator")
        need(row["pin_memory"] is True, "loader pinning")
        exact(row["prefetch_factor"], 4 if workers else None, "prefetch factor")
        method = row["worker_start_method"]
        need(method in ("fork", "spawn", "forkserver") if workers else method is None, "worker start method")
        need(method == rows[0]["worker_start_method"], "changed worker context")
        pids = ids(row["worker_pids"], workers)
        if position:
            exact(row["epoch"], expected_epochs[position - 1], "lifecycle epoch")
            need(row["pinned_image"] is True, "image pinning")
        replacement = (position == 0 and start == 3) or (position > 0 and expected_epochs[position - 1] == 2)
        if replacement:
            exited(
                row["replaced_worker_pids"], row["replaced_worker_exitcodes"], row["replaced_workers_alive"], workers
            )
            need(not set(pids) & set(row["replaced_worker_pids"]), "reset reused workers")
            if position:
                need(row["replaced_worker_pids"] == rows[position - 1]["worker_pids"], "reset worker history")
        else:
            need(not any(k.startswith("replaced_worker") for k in row), "unexpected reset")
            if position:
                need(pids == rows[position - 1]["worker_pids"], "worker change outside reset")
        settings.append({k: row[k] for k in ("worker_start_method", "pin_memory", "prefetch_factor")})
    need(shutdown["train"]["worker_pids"] == rows[-1]["worker_pids"], "final train worker pool")
    if start == 0:
        need(
            raw["resume_input"] is None and args["resume_from"] is None and args["resume_receipt"] is None,
            "unexpected resume",
        )
        need(
            args["checkpoint_epochs"] == [1, 2] and [c["epoch"] for c in raw["checkpoints"]] == [1, 2],
            "retained checkpoint set",
        )
        for checkpoint in raw["checkpoints"]:
            need(type(checkpoint["epoch"]) is int, "checkpoint epoch type")
            name = f"resume-epoch-{checkpoint['epoch']}.pt"
            need(checkpoint["path"] == str(original.with_suffix(".run") / name), "checkpoint path escaped run")
            file_binding(runs / (original.stem + ".run") / name, checkpoint, name)
    else:
        need(
            reference is not None and not raw["checkpoints"] and args["checkpoint_epochs"] == [],
            "resume checkpoint set",
        )
        ref_raw, ref_record = reference
        checkpoint = next(c for c in ref_raw["checkpoints"] if c["epoch"] == start - 1)
        original_ref = str(Path(ref_record["path"]))
        need(args["resume_from"] == checkpoint["path"] and args["resume_receipt"] == original_ref, "resume arguments")
        resume = raw["resume_input"]
        need(resume is not None, "missing resume input")
        for key, expected in {
            "path": checkpoint["path"],
            "sha256": checkpoint["sha256"],
            "receipt_sha256": ref_record["sha256"],
            "expected_start_epoch": start,
        }.items():
            exact(resume[key], expected, "resume identity " + key)
        restored = resume["restored_state_expectation"]
        exact(restored["checkpoint_epoch"], start - 1, "restored epoch")
        need(
            type(restored["optimizer_states"]) is int
            and restored["optimizer_states"] > 0
            and positive(restored["ema_updates"]),
            "restored optimizer/EMA",
        )
    samples = raw["resource_samples"]
    need(
        samples
        and all(
            positive(s["time"]) and type(s["family_rss_bytes"]) is int and s["family_rss_bytes"] > 0 for s in samples
        ),
        "resource samples",
    )
    exact(raw["sampled_peak_family_rss_bytes"], max(s["family_rss_bytes"] for s in samples), "RSS peak arithmetic")
    compact = {k: v for k, v in raw.items() if k not in ("epochs", "resource_samples")}
    compact["epochs"] = compact_epochs
    need(canonical(record["report"]) == canonical(compact), "parent compaction disagrees with raw evidence")
    signature = {
        "initial_model": raw["initial_state_sha256"],
        "final_model": raw["final_state_sha256"],
        "epochs": expected_epochs,
        "loss_traces": loss_hashes,
        "resume_input": raw["resume_input"],
        "loader_settings": settings,
    }
    return signature


def audit(grid_path, runs, source, fresh_file, cache_file):
    grid = load(grid_path)
    need(
        grid["complete"] is True
        and grid.get("full_protocol_complete") is True
        and grid["full_protocol_requested"] is True
        and "error" not in grid,
        "complete full grid required",
    )
    exact(grid["planned_trials"], 54, "planned trial count")
    exact(grid["planned_epochs"], 126, "planned epoch count")
    records = grid["records"]
    need(len(records) == 54, "actual trial count")
    conditions = [(records[i]["spec"]["workers"], records[i]["spec"]["overlap"]) for i in range(0, 54, 9)]
    need(set(conditions) == {(w, o) for w in (0, 2, 8) for o in ("yes", "no")}, "worker/mask condition coverage")
    expected = []
    for workers, overlap in conditions:
        need(type(workers) is int, "worker condition type")
        for stage, start in STAGES:
            for backend in BACKENDS:
                expected.append(
                    {
                        "id": f"w{workers}-{overlap}-{stage}-{backend}",
                        "workers": workers,
                        "overlap": overlap,
                        "stage": stage,
                        "backend": backend,
                        "start_epoch": start,
                        "reference_id": f"w{workers}-{overlap}-fresh-reference",
                    }
                )
    need(canonical([r["spec"] for r in records]) == canonical(expected), "trial order/specification")
    harness = {name: sha(source / "bench" / name) for name in ("train_coco_gpu.py", "coco_loader.py")}
    need(
        grid["harness_sources"] == harness and grid["script_sha256"] == sha(source / "bench/lifecycle_coco_gpu.py"),
        "frozen harness identity",
    )
    fresh_sha, cache_sha = sha(fresh_file), sha(cache_file)
    fresh = load(fresh_file)
    need(
        grid["fresh_check_sha256"] == fresh_sha and fresh["fresh_cache_sha256"] == cache_sha, "fresh/cache file binding"
    )
    need(
        fresh["fresh_reference"]["images"] == 5000
        and fresh["matched_runs"] == 30
        and fresh["all_benchmark_outputs_match_fresh_reference"] is True,
        "fresh reference coverage",
    )
    original_runs = Path(records[0]["path"]).parent
    need(original_runs.is_absolute() and ".." not in original_runs.parts, "original run directory")
    command_prefix = records[0]["command"][:2]
    need(
        len(command_prefix) == 2
        and Path(command_prefix[0]).is_absolute()
        and Path(command_prefix[1]).is_absolute()
        and Path(command_prefix[1]).name == "train_coco_gpu.py",
        "command executable/script",
    )
    references, cohorts, summaries = {}, {}, []
    identity = None
    for record, spec in zip(records, expected, strict=True):
        name = spec["id"] + ".json"
        original = original_runs / name
        need(record["path"] == str(original), "raw path escaped run directory")
        exact(record["returncode"], 0, "child return code")
        need(record["validated"] is True, "unvalidated child")
        for k in ("started_at_ns", "finished_at_ns"):
            need(type(record[k]) is int and record[k] > 0, "child timestamp")
        need(record["finished_at_ns"] >= record["started_at_ns"], "child finish before start")
        command = command_prefix + [
            "--corpus",
            grid["corpus"],
            "--fresh-check",
            grid["fresh_check"],
            "--out",
            str(original),
            "--backend",
            spec["backend"],
            "--workers",
            str(spec["workers"]),
            "--epochs",
            "4",
            "--batch",
            "4",
            "--imgsz",
            "640",
            "--overlap",
            spec["overlap"],
            "--close-mosaic",
            "2",
            "--persistent-mask",
        ]
        if spec["start_epoch"]:
            ref = original_runs / (spec["reference_id"] + ".json")
            command += [
                "--resume-from",
                str(ref.with_suffix(".run") / f"resume-epoch-{spec['start_epoch'] - 1}.pt"),
                "--resume-receipt",
                str(ref),
            ]
        else:
            command += ["--checkpoint-epochs", "1", "2"]
        need(record["command"] == command, "child command differs from protocol")
        path = runs / name
        need(
            sha(path) == record["sha256"] and sha(path.with_suffix(".log")) == record["log_sha256"], "raw/log checksum"
        )
        raw = load(path)
        profile = {k: raw[k] for k in IDENTITY}
        identity = profile if identity is None else identity
        need(profile == identity, "runtime identity drift")
        signature = trial(
            raw, spec, record, runs, harness, grid["corpus"], fresh_sha, cache_sha, references.get(spec["reference_id"])
        )
        cohort = f"w{spec['workers']}-{spec['overlap']}-{spec['stage']}"
        if spec["backend"] == "reference":
            cohorts[cohort] = {"signature": signature, "backends": []}
        need(canonical(signature) == canonical(cohorts[cohort]["signature"]), "cohort loss/model/resume mismatch")
        cohorts[cohort]["backends"].append(spec["backend"])
        if spec["stage"] == "fresh" and spec["backend"] == "reference":
            references[spec["id"]] = (
                {"checkpoints": raw["checkpoints"]},
                {"path": record["path"], "sha256": record["sha256"]},
            )
        summaries.append(
            {
                "id": spec["id"],
                "raw_sha256": record["sha256"],
                "epochs": signature["epochs"],
                "loss_sha256": signature["loss_traces"],
                "final_model_sha256": signature["final_model"],
            }
        )
    need(canonical(grid["cohorts"]) == canonical(cohorts) and len(cohorts) == 18, "parent cohort summary")
    need(grid["runtime_identity"] == identity, "parent runtime summary")
    need(sum(len(s["epochs"]) for s in summaries) == 126, "actual total epochs")
    return {
        "audit_passed": True,
        "full_protocol_complete": True,
        "trials": 54,
        "epochs": 126,
        "cohorts": 18,
        "grid_sha256": sha(grid_path),
        "auditor_sha256": sha(__file__),
        "validated": summaries,
        "scope": "Independent complete-artifact/lifecycle/cohort consistency audit; no Torch/pickle execution, held-out accuracy or repeated speedup claim.",
        "limitations": [
            "Checkpoint bytes and recorded optimizer/EMA restoration counts, not independently decoded state tensors.",
            "Summed family RSS double-counts shared pages; samples do not prove unsampled peaks.",
            "Resume cohorts are compared internally, not against uninterrupted training.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fresh-check", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.out.open("x") as stream:
        try:
            result = audit(args.grid, args.runs, args.source, args.fresh_check, args.reference_cache)
        except (ValueError, KeyError, TypeError, OSError, StopIteration) as error:
            result = {"audit_passed": False, "error": str(error), "auditor_sha256": sha(__file__)}
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "validated"}))
    return 0 if result["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
