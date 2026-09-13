"""Run the declared real-data close-mosaic/resume cohorts in fresh processes.

Default: six worker/mask conditions, three cohorts and three backends, giving
54 trials and 126 epochs. This is parity/lifecycle qualification, not repeated
throughput or resumed-versus-uninterrupted equivalence. No framework import here.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BACKENDS = ("reference", "mask", "both")
STAGES = (("fresh", 0), ("resume-at-boundary", 2), ("resume-after-boundary", 3))
FIXTURE = {
    "files": 9952, "bytes": 830426850,
    "sha256": "8b079ad2d8472ff17b6cca9b0250e831472f12ef500f2b065125f6f39406c208",
}
PROFILE_FIELDS = (
    "harness_sources", "upstream_files", "packages", "fixture", "corpus_root",
    "original_cache_sha256", "fresh_check_sha256", "python", "platform",
    "torch", "numpy", "opencv", "gpu", "cuda", "cpu_count", "ram_bytes",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def finite(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def make_plan(workers, overlaps):
    require(workers and len(workers) == len(set(workers)), "worker conditions must be nonempty and unique")
    require(all(type(w) is int and w in (0, 2, 8) for w in workers), "unsupported worker condition")
    require(overlaps and len(overlaps) == len(set(overlaps)), "mask conditions must be nonempty and unique")
    require(all(o in ("yes", "no") for o in overlaps), "unsupported mask condition")
    return [
        {
            "id": f"w{w}-{o}-{stage}-{backend}", "workers": w, "overlap": o,
            "stage": stage, "backend": backend, "start_epoch": start,
            "reference_id": f"w{w}-{o}-fresh-reference",
        }
        for w in workers for o in overlaps for stage, start in STAGES for backend in BACKENDS
    ]


def trial_command(spec, python, script, corpus, fresh, directory):
    path = directory / (spec["id"] + ".json")
    command = [
        str(python), str(script), "--corpus", str(corpus), "--fresh-check", str(fresh),
        "--out", str(path), "--backend", spec["backend"], "--workers", str(spec["workers"]),
        "--epochs", "4", "--batch", "4", "--imgsz", "640", "--overlap", spec["overlap"],
        "--close-mosaic", "2", "--persistent-mask",
    ]
    if spec["start_epoch"]:
        reference = directory / (spec["reference_id"] + ".json")
        checkpoint = reference.with_suffix(".run") / f"resume-epoch-{spec['start_epoch'] - 1}.pt"
        command += ["--resume-from", str(checkpoint), "--resume-receipt", str(reference)]
    else:
        command += ["--checkpoint-epochs", "1", "2"]
    return command


def check_workers(pids, codes, alive, count):
    require(type(count) is int and count >= 0, "invalid worker count")
    require(len(pids) == count and len(set(pids)) == count, "missing or duplicate worker PIDs")
    require(all(type(p) is int and p > 0 for p in pids), "invalid worker PID")
    require(len(codes) == count and all(type(c) is int and c == 0 for c in codes), "unclean worker exit")
    require(len(alive) == count and all(value is False for value in alive), "worker still alive")


def validate_trial(value, spec, path, harness, corpus, fresh_sha, resume_input=None):
    require(value["complete"] is True and value["lifecycle_requested"] is True, "incomplete lifecycle trial")
    require(value["script_sha256"] == harness["train_coco_gpu.py"] and value["harness_sources"] == harness,
            "trial harness drift")
    require(value["corpus_root"] == str(corpus) and value["fresh_check_sha256"] == fresh_sha, "input identity drift")
    require(value["fixture"] == FIXTURE, "wrong real-data fixture")
    config = value["args"]
    for key, expected in (("workers", spec["workers"]), ("epochs", 4), ("batch", 4), ("imgsz", 640), ("close_mosaic", 2)):
        require(type(config[key]) is int and config[key] == expected, f"wrong {key}")
    require(config["backend"] == spec["backend"] and config["overlap"] == spec["overlap"], "wrong backend or mask mode")
    require(config["persistent_mask"] is True and config["out"] == str(path), "wrong opt-in or output")
    require(value["train_workers"] == spec["workers"], "wrong actual train workers")
    resolved = value["resolved_config"]
    for key, expected in {
        "epochs": 4, "batch": 4, "imgsz": 640, "workers": spec["workers"],
        "overlap_mask": spec["overlap"] == "yes", "close_mosaic": 2, "seed": 912,
        "deterministic": True, "amp": False, "optimizer": "SGD", "lr0": 0.001,
        "cache": False, "rect": False, "fraction": 1.0, "mask_ratio": 4,
    }.items():
        require(type(resolved[key]) is type(expected) and resolved[key] == expected, f"resolved {key} differs")
    expected_epochs = list(range(spec["start_epoch"], 4))
    require([e["epoch"] for e in value["epochs"]] == expected_epochs, "missing or reordered epochs")
    epochs, losses = [], []
    for epoch in value["epochs"]:
        require(type(epoch["epoch"]) is int and epoch["images"] == 5000 and epoch["batches"] == 1250,
                "incomplete full-image epoch")
        require(finite(epoch["seconds"], 1e-12) and finite(epoch["images_per_second"], 1e-12)
                and math.isclose(epoch["images_per_second"], 5000 / epoch["seconds"], rel_tol=1e-12),
                "invalid epoch timing")
        names, trace = epoch["loss_names"], epoch["all_batch_losses"]
        require(names and len(names) == len(set(names)) and "seg_loss" in names, "invalid loss names")
        require(len(trace) == 1250 and all(len(row) == len(names) and all(finite(x, -math.inf) for x in row) for row in trace),
                "incomplete or nonfinite loss trace")
        require(sum(row[names.index("seg_loss")] for row in trace) > 0, "no foreground segmentation loss")
        allocated, reserved = epoch["cuda_peak_allocated_bytes"], epoch["cuda_peak_reserved_bytes"]
        require(type(allocated) is int and type(reserved) is int and 0 < allocated <= reserved, "invalid CUDA peaks")
        loss_hash = digest({"names": names, "values": trace})
        losses.append(loss_hash)
        epochs.append({**{k: v for k, v in epoch.items() if k != "all_batch_losses"}, "loss_sha256": loss_hash})
    for key in ("initial_state_sha256", "final_state_sha256"):
        require(is_sha(value[key]), "invalid model digest")
    require(value["initial_state_sha256"] != value["final_state_sha256"], "model did not update")
    require(finite(value["whole_job_seconds"], 1e-12), "invalid job duration")
    require({d["mode"] for d in value["datasets"]} == {"train", "val"} and len(value["datasets"]) == 2,
            "missing dataset construction")
    require(all(d["images"] == 5000 and d["replaced_formats"] == int(spec["backend"] != "reference") for d in value["datasets"]),
            "wrong replacement or dataset size")
    for name, count in (("train", value["train_workers"]), ("val", value["val_workers"])):
        shutdown = value["worker_shutdown"][name]
        require(shutdown["worker_count"] == count, "shutdown count mismatch")
        check_workers(shutdown["worker_pids"], shutdown["worker_exitcodes"], shutdown["workers_alive"], count)

    lifecycle = value["lifecycle"]
    require(len(lifecycle) == len(expected_epochs) + 1, "missing lifecycle observations")
    require([r["stage"] for r in lifecycle] == ["prepared"] + ["first-batch"] * len(expected_epochs), "wrong lifecycle stages")
    wanted = "Format" if spec["backend"] == "reference" else "FastFormat"
    collator = "ultralytics.data.dataset.YOLODataset.collate_fn" if wanted == "Format" else "ultrafast_maskops._shared_collate.shared_collate_fn"
    for index, observation in enumerate(lifecycle):
        require(type(observation["start_epoch"]) is int and observation["start_epoch"] == spec["start_epoch"], "wrong start epoch")
        closed = spec["start_epoch"] > 2 if index == 0 else expected_epochs[index - 1] >= 2
        require(observation["formatter"] == observation["factory"] == wanted and observation["collator"] == collator,
                "formatter/factory/collator mismatch")
        require(observation["rebuilt_from_initial"] is closed and observation["pin_memory"] is True, "wrong rebuild or pinning")
        pids = observation["worker_pids"]
        require(len(pids) == len(set(pids)) == spec["workers"] and all(type(p) is int and p > 0 for p in pids), "invalid live worker IDs")
        require(observation["worker_start_method"] in ("fork", "spawn", "forkserver") if spec["workers"] else observation["worker_start_method"] is None,
                "missing actual worker context")
        require(observation["worker_start_method"] == lifecycle[0]["worker_start_method"], "worker context changed during trial")
        require(observation["prefetch_factor"] == (4 if spec["workers"] else None), "wrong training prefetch setting")
        if index:
            require(observation["epoch"] == expected_epochs[index - 1] and observation["pinned_image"] is True, "wrong first-batch observation")
        replacement = (index == 0 and spec["start_epoch"] > 2) or (index > 0 and expected_epochs[index - 1] == 2)
        if replacement:
            check_workers(observation["replaced_worker_pids"], observation["replaced_worker_exitcodes"],
                          observation["replaced_workers_alive"], spec["workers"])
            require(not set(pids) & set(observation["replaced_worker_pids"]), "worker PIDs reused at reset")
            if index:
                require(observation["replaced_worker_pids"] == lifecycle[index - 1]["worker_pids"], "replacement history mismatch")
        else:
            require(not any(k.startswith("replaced_worker") for k in observation), "unexpected worker replacement")
            if index:
                require(pids == lifecycle[index - 1]["worker_pids"], "workers changed outside a reset")
    require(value["worker_shutdown"]["train"]["worker_pids"] == lifecycle[-1]["worker_pids"], "shutdown observed another worker pool")
    require(not set(value["worker_shutdown"]["train"]["worker_pids"]) & set(value["worker_shutdown"]["val"]["worker_pids"]),
            "train and validation share worker identities")

    if spec["start_epoch"]:
        require(value["resume_input"] is not None and resume_input is not None, "missing resume identity")
        require(config["resume_from"] == resume_input["path"]
                and config["resume_receipt"] == str(path.parent / (spec["reference_id"] + ".json")),
                "wrong resume arguments")
        for key, expected in resume_input.items():
            require(value["resume_input"][key] == expected, f"resume {key} mismatch")
        restored = value["resume_input"]["restored_state_expectation"]
        require(restored["checkpoint_epoch"] == spec["start_epoch"] - 1 and restored["optimizer_states"] > 0 and restored["ema_updates"] > 0,
                "missing checkpoint optimizer/EMA state")
        require(config["checkpoint_epochs"] == [] and value["checkpoints"] == [], "resumed trial requested new checkpoint cohort")
    else:
        require(value["resume_input"] is None and config["checkpoint_epochs"] == [1, 2], "wrong fresh checkpoint protocol")
        require(config["resume_from"] is None and config["resume_receipt"] is None, "fresh trial resumed unexpectedly")
        require([c["epoch"] for c in value["checkpoints"]] == [1, 2], "missing retained checkpoints")
        for checkpoint in value["checkpoints"]:
            checkpoint_path = path.with_suffix(".run") / f"resume-epoch-{checkpoint['epoch']}.pt"
            require(checkpoint["path"] == str(checkpoint_path), "checkpoint escaped trial directory")
            require(checkpoint_path.stat().st_size == checkpoint["bytes"] and file_sha(checkpoint_path) == checkpoint["sha256"],
                    "retained checkpoint changed")
    samples = value["resource_samples"]
    require(samples and all(finite(s["time"], 1) and type(s["family_rss_bytes"]) is int and s["family_rss_bytes"] > 0 for s in samples),
            "missing resource samples")
    require(value["sampled_peak_family_rss_bytes"] == max(s["family_rss_bytes"] for s in samples), "wrong family RSS peak")
    compact = {k: v for k, v in value.items() if k not in ("epochs", "resource_samples")}
    compact["epochs"] = epochs
    return compact, {
        "initial_model": value["initial_state_sha256"], "final_model": value["final_state_sha256"],
        "epochs": expected_epochs, "loss_traces": losses,
        "resume_input": value["resume_input"],
        "loader_settings": [
            {k: r[k] for k in ("worker_start_method", "pin_memory", "prefetch_factor")} for r in lifecycle
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--fresh-check", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(0, 2, 8), nargs="+", default=[0, 2, 8])
    parser.add_argument("--overlaps", choices=("yes", "no"), nargs="+", default=["yes", "no"])
    args = parser.parse_args()
    plan = make_plan(args.workers, args.overlaps)
    out, corpus, fresh = args.out.resolve(), args.corpus.resolve(), args.fresh_check.resolve()
    directory = out.with_suffix(".runs")
    temporary = out.with_suffix(".tmp")
    require(not out.exists() and not directory.exists() and not temporary.exists(), "retain previous results; choose a new output")
    script = Path(__file__).with_name("train_coco_gpu.py").resolve()
    harness = {name: file_sha(script.with_name(name)) for name in ("train_coco_gpu.py", "coco_loader.py")}
    fresh_sha = file_sha(fresh)
    directory.mkdir(parents=True)
    report = {
        "complete": False, "controller_pid": os.getpid(), "script_sha256": file_sha(__file__),
        "harness_sources": harness, "corpus": str(corpus), "fresh_check": str(fresh), "fresh_check_sha256": fresh_sha,
        "scope": "real-data lifecycle/cohort parity; no repeated speedup or resumed/uninterrupted equivalence claim",
        "full_protocol_requested": set(args.workers) == {0, 2, 8} and set(args.overlaps) == {"yes", "no"},
        "planned_trials": len(plan), "planned_epochs": sum(4 - s["start_epoch"] for s in plan),
        "plan": plan, "records": [], "cohorts": {},
    }

    def save():
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, out)

    identity, references = None, {}
    save()
    try:
        for spec in plan:
            require(harness == {n: file_sha(script.with_name(n)) for n in harness}, "harness changed during series")
            require(file_sha(fresh) == fresh_sha, "fresh receipt changed during series")
            path = directory / (spec["id"] + ".json")
            command = trial_command(spec, sys.executable, script, corpus, fresh, directory)
            resume_input = None
            if spec["start_epoch"]:
                reference = references[spec["reference_id"]]
                reference_path = directory / (spec["reference_id"] + ".json")
                require(file_sha(reference_path) == reference["sha256"], "reference receipt changed")
                checkpoint = next(c for c in reference["checkpoints"] if c["epoch"] == spec["start_epoch"] - 1)
                require(file_sha(checkpoint["path"]) == checkpoint["sha256"], "reference checkpoint changed")
                resume_input = {"path": checkpoint["path"], "sha256": checkpoint["sha256"],
                                "receipt_sha256": reference["sha256"], "expected_start_epoch": spec["start_epoch"]}
            record = {"spec": spec, "path": str(path), "command": command, "started_at_ns": time.time_ns()}
            report["records"].append(record)
            save()
            print("Starting", spec["id"], flush=True)
            log = path.with_suffix(".log")
            with log.open("x") as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                record["pid"] = process.pid
                save()
                try:
                    code = process.wait()
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                    record.update(returncode=process.returncode, finished_at_ns=time.time_ns(), log_sha256=file_sha(log))
                    if path.exists():
                        record["sha256"] = file_sha(path)
                    save()
            require(code == 0, f"trial failed: {spec['id']}; preserve original report and log")
            value = json.loads(path.read_text())
            compact, signature = validate_trial(value, spec, path, harness, corpus, fresh_sha, resume_input)
            profile = {key: value[key] for key in PROFILE_FIELDS}
            identity = profile if identity is None else identity
            require(profile == identity, "runtime, corpus, or cache identity changed across trials")
            cohort = f"w{spec['workers']}-{spec['overlap']}-{spec['stage']}"
            if spec["backend"] == "reference":
                report["cohorts"][cohort] = {"signature": signature, "backends": []}
            require(report["cohorts"][cohort]["signature"] == signature, "loss/model mismatch inside cohort")
            report["cohorts"][cohort]["backends"].append(spec["backend"])
            record.update(validated=True, report=compact)
            if spec["stage"] == "fresh" and spec["backend"] == "reference":
                references[spec["id"]] = {"sha256": record["sha256"], "checkpoints": value["checkpoints"]}
            if resume_input:
                require(file_sha(resume_input["path"]) == resume_input["sha256"], "resume input modified by trial")
                require(file_sha(reference_path) == resume_input["receipt_sha256"], "resume receipt modified by trial")
            save()
            print("Validated", len(report["records"]), "/", len(plan), spec["id"], flush=True)
            del value, compact
        require(all(c["backends"] == list(BACKENDS) for c in report["cohorts"].values()), "incomplete backend cohorts")
        report.update(complete=True, runtime_identity=identity,
                      full_protocol_complete=report["full_protocol_requested"], finished_at_ns=time.time_ns())
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
