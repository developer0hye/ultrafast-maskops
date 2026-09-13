"""Synthetic protocol checks, not GPU training or measured benchmark evidence."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("lifecycle_coordinator", Path(__file__).parents[1] / "bench/lifecycle_coco_gpu.py")
coordinator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coordinator)


def synthetic_trial(tmp_path, stage="fresh", backend="reference", workers=2):
    spec = next(r for r in coordinator.make_plan([workers], ["yes"]) if r["stage"] == stage and r["backend"] == backend)
    path = tmp_path / (spec["id"] + ".json")
    run = path.with_suffix(".run")
    run.mkdir()
    harness = {"train_coco_gpu.py": "a" * 64, "coco_loader.py": "b" * 64}
    corpus = tmp_path / "synthetic-corpus"
    names = ["box_loss", "seg_loss", "cls_loss", "dfl_loss"]
    epochs = [
        {"epoch": epoch, "seconds": 100.0, "images": 5000, "batches": 1250, "images_per_second": 50.0,
         "cuda_peak_allocated_bytes": 100, "cuda_peak_reserved_bytes": 200, "loss_names": names,
         "all_batch_losses": [[1.0, 2.0, 3.0, 4.0] for _ in range(1250)]}
        for epoch in range(spec["start_epoch"], 4)
    ]
    wanted = "Format" if backend == "reference" else "FastFormat"
    collator = "ultralytics.data.dataset.YOLODataset.collate_fn" if backend == "reference" else "ultrafast_maskops._shared_collate.shared_collate_fn"
    lifecycle = []
    for epoch in [None] + list(range(spec["start_epoch"], 4)):
        replacement = (epoch is None and spec["start_epoch"] > 2) or epoch == 2
        closed = spec["start_epoch"] > 2 if epoch is None else epoch >= 2
        row = {
            "stage": "prepared" if epoch is None else "first-batch", "epoch": epoch,
            "start_epoch": spec["start_epoch"], "formatter": wanted, "factory": wanted, "collator": collator,
            "rebuilt_from_initial": closed, "pin_memory": True, "prefetch_factor": 4 if workers else None,
            "worker_start_method": "fork" if workers else None,
            "worker_pids": list(range(2000 if closed else 1000, (2000 if closed else 1000) + workers)),
        }
        if epoch is not None:
            row["pinned_image"] = True
        if replacement:
            row.update(replaced_worker_pids=list(range(1000, 1000 + workers)), replaced_worker_exitcodes=[0] * workers,
                       replaced_workers_alive=[False] * workers)
        lifecycle.append(row)
    checkpoints = []
    for epoch in (() if spec["start_epoch"] else (1, 2)):
        checkpoint = run / f"resume-epoch-{epoch}.pt"
        checkpoint.write_bytes(f"synthetic checkpoint {epoch}; not a Torch model".encode())
        checkpoints.append({"epoch": epoch, "path": str(checkpoint), "bytes": checkpoint.stat().st_size,
                            "sha256": coordinator.file_sha(checkpoint)})
    resume = None if not spec["start_epoch"] else {
        "path": str(tmp_path / (spec["reference_id"] + ".run") / f"resume-epoch-{spec['start_epoch'] - 1}.pt"),
        "sha256": "c" * 64, "receipt_sha256": "d" * 64,
        "expected_start_epoch": spec["start_epoch"],
    }
    value = {
        "complete": True, "lifecycle_requested": True, "script_sha256": harness["train_coco_gpu.py"],
        "harness_sources": harness, "corpus_root": str(corpus), "fresh_check_sha256": "e" * 64,
        "fixture": copy.deepcopy(coordinator.FIXTURE),
        "args": {"workers": workers, "epochs": 4, "batch": 4, "imgsz": 640, "close_mosaic": 2,
                 "backend": backend, "overlap": "yes", "persistent_mask": True, "out": str(path),
                 "checkpoint_epochs": [] if spec["start_epoch"] else [1, 2],
                 "resume_from": None if resume is None else resume["path"],
                 "resume_receipt": None if resume is None else str(tmp_path / (spec["reference_id"] + ".json"))},
        "resolved_config": {"epochs": 4, "batch": 4, "imgsz": 640, "workers": workers, "overlap_mask": True,
                            "close_mosaic": 2, "seed": 912, "deterministic": True, "amp": False,
                            "optimizer": "SGD", "lr0": 0.001, "cache": False, "rect": False,
                            "fraction": 1.0, "mask_ratio": 4},
        "train_workers": workers, "val_workers": workers, "epochs": epochs,
        "initial_state_sha256": "1" * 64, "final_state_sha256": "2" * 64, "whole_job_seconds": 500.0,
        "datasets": [{"mode": mode, "images": 5000, "replaced_formats": int(backend != "reference")} for mode in ("train", "val")],
        "worker_shutdown": {name: {"worker_count": workers, "worker_pids": list(range(2000 if name == "train" else 3000,
                                                                                   (2000 if name == "train" else 3000) + workers)),
                                   "worker_exitcodes": [0] * workers, "workers_alive": [False] * workers}
                            for name in ("train", "val")},
        "lifecycle": lifecycle, "checkpoints": checkpoints,
        "resume_input": None if resume is None else {
            **resume, "restored_state_expectation": {"checkpoint_epoch": spec["start_epoch"] - 1,
                                                    "optimizer_states": 7, "ema_updates": 12}},
        "resource_samples": [{"time": 1234.0, "family_rss_bytes": 1000}], "sampled_peak_family_rss_bytes": 1000,
    }
    return value, spec, path, harness, corpus, "e" * 64, resume


def test_full_plan_has_54_trials_126_epochs_and_shared_reference_checkpoints(tmp_path):
    plan = coordinator.make_plan([0, 2, 8], ["yes", "no"])
    assert len(plan) == len({row["id"] for row in plan}) == 54
    assert sum(4 - row["start_epoch"] for row in plan) == 126
    for workers in (0, 2, 8):
        for overlap in ("yes", "no"):
            condition = [row for row in plan if row["workers"] == workers and row["overlap"] == overlap]
            assert [row["start_epoch"] for row in condition] == [0, 0, 0, 2, 2, 2, 3, 3, 3]
            for stage, start in (("fresh", 0), ("resume-at-boundary", 2), ("resume-after-boundary", 3)):
                rows = [row for row in condition if row["stage"] == stage]
                assert [row["backend"] for row in rows] == ["reference", "mask", "both"]
                commands = [coordinator.trial_command(row, "python", Path("train.py"), Path("corpus"), Path("fresh.json"), tmp_path) for row in rows]
                if start:
                    inputs = [cmd[cmd.index("--resume-from") + 1] for cmd in commands]
                    assert len(set(inputs)) == 1
                    assert inputs[0] == str(tmp_path / f"w{workers}-{overlap}-fresh-reference.run" / f"resume-epoch-{start - 1}.pt")
                    assert all("--checkpoint-epochs" not in cmd for cmd in commands)
                else:
                    assert all(cmd[-3:] == ["--checkpoint-epochs", "1", "2"] for cmd in commands)


@pytest.mark.parametrize("workers,overlaps", [([], ["yes"]), ([2, 2], ["yes"]), ([True], ["yes"]), ([3], ["yes"]),
                                            ([2], []), ([2], ["no", "no"]), ([2], ["unknown"])])
def test_invalid_plan_rejected(workers, overlaps):
    with pytest.raises(ValueError):
        coordinator.make_plan(workers, overlaps)


@pytest.mark.parametrize("stage", ["fresh", "resume-at-boundary", "resume-after-boundary"])
@pytest.mark.parametrize("backend", ["reference", "mask", "both"])
@pytest.mark.parametrize("workers", [0, 2, 8])
def test_synthetic_trial_validation_and_compaction(tmp_path, stage, backend, workers):
    data = synthetic_trial(tmp_path, stage, backend, workers)
    compact, signature = coordinator.validate_trial(*data)
    assert "resource_samples" not in compact
    assert all("all_batch_losses" not in row and "loss_sha256" in row for row in compact["epochs"])
    assert len(signature["loss_traces"]) == 4 - data[1]["start_epoch"]
    assert "resource_samples" in data[0]  # Validation does not mutate the raw evidence.


@pytest.mark.parametrize("damage", ["incomplete", "wrong_fixture", "wrong_resolved_config", "missing_epoch", "nan_loss",
                                    "short_trace", "no_foreground", "wrong_collator", "not_pinned", "lost_factory",
                                    "old_worker_abort", "old_worker_alive", "reused_pid", "final_worker_abort",
                                    "missing_val_worker", "bad_rss", "changed_checkpoint", "bool_exit", "wrong_start",
                                    "unexpected_worker_change", "replacement_history", "wrong_shutdown_pool"])
def test_corrupt_lifecycle_report_is_rejected(tmp_path, damage):
    data = synthetic_trial(tmp_path)
    value = data[0]
    transition = value["lifecycle"][3]  # Epoch 2, after the real closure boundary.
    if damage == "incomplete":
        value["complete"] = False
    elif damage == "wrong_fixture":
        value["fixture"]["files"] -= 1
    elif damage == "wrong_resolved_config":
        value["resolved_config"]["amp"] = True
    elif damage == "missing_epoch":
        value["epochs"].pop()
    elif damage == "nan_loss":
        value["epochs"][0]["all_batch_losses"][0][0] = float("nan")
    elif damage == "short_trace":
        value["epochs"][0]["all_batch_losses"].pop()
    elif damage == "no_foreground":
        for row in value["epochs"][0]["all_batch_losses"]:
            row[1] = 0.0
    elif damage == "wrong_collator":
        transition["collator"] = "unknown"
    elif damage == "not_pinned":
        transition["pinned_image"] = False
    elif damage == "lost_factory":
        transition["factory"] = "UnknownFormat"
    elif damage == "old_worker_abort":
        transition["replaced_worker_exitcodes"][0] = -6
    elif damage == "old_worker_alive":
        transition["replaced_workers_alive"][0] = True
    elif damage == "reused_pid":
        transition["replaced_worker_pids"][0] = transition["worker_pids"][0]
    elif damage == "final_worker_abort":
        value["worker_shutdown"]["train"]["worker_exitcodes"][0] = -6
    elif damage == "missing_val_worker":
        value["worker_shutdown"]["val"]["worker_pids"].pop()
    elif damage == "bad_rss":
        value["sampled_peak_family_rss_bytes"] += 1
    elif damage == "changed_checkpoint":
        Path(value["checkpoints"][0]["path"]).write_bytes(b"changed")
    elif damage == "bool_exit":
        transition["replaced_worker_exitcodes"][0] = False
    elif damage == "wrong_start":
        transition["start_epoch"] = 1
    elif damage == "unexpected_worker_change":
        value["lifecycle"][1]["worker_pids"][0] += 100
    elif damage == "replacement_history":
        transition["replaced_worker_pids"][0] += 100
    elif damage == "wrong_shutdown_pool":
        value["worker_shutdown"]["train"]["worker_pids"][0] += 100
    with pytest.raises(ValueError):
        coordinator.validate_trial(*data)


@pytest.mark.parametrize("stage", ["resume-at-boundary", "resume-after-boundary"])
def test_resume_must_bind_same_reference_and_check_setup_reset(tmp_path, stage):
    data = synthetic_trial(tmp_path, stage)
    value = data[0]
    original = value["resume_input"]["sha256"]
    value["resume_input"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="resume sha256"):
        coordinator.validate_trial(*data)
    value["resume_input"]["sha256"] = original
    transition = value["lifecycle"][0 if stage == "resume-after-boundary" else 1]
    transition["replaced_worker_exitcodes"][0] = -6
    with pytest.raises(ValueError, match="unclean worker exit"):
        coordinator.validate_trial(*data)


def test_cohort_signature_detects_loss_model_and_loader_changes(tmp_path):
    data = synthetic_trial(tmp_path)
    _, original = coordinator.validate_trial(*data)
    data[0]["epochs"][0]["all_batch_losses"][10][0] += 0.01
    _, changed_loss = coordinator.validate_trial(*data)
    assert original != changed_loss
    data[0]["final_state_sha256"] = "3" * 64
    _, changed_model = coordinator.validate_trial(*data)
    assert changed_loss != changed_model
    for observation in data[0]["lifecycle"]:
        observation["worker_start_method"] = "spawn"
    _, changed_context = coordinator.validate_trial(*data)
    assert changed_model != changed_context


@pytest.mark.parametrize("failure", [None, "exit", "parity"])
def test_coordinator_retains_process_failure_and_compares_complete_cohorts(tmp_path, monkeypatch, failure):
    """A fake child writes marked synthetic records; no GPU job is launched."""
    own = tmp_path / "lifecycle_coco_gpu.py"
    own.write_text("synthetic controller identity; not executable training code\n")
    for name in ("train_coco_gpu.py", "coco_loader.py"):
        (tmp_path / name).write_text("synthetic harness identity: " + name + "\n")
    fresh = tmp_path / "fresh.json"
    fresh.write_text("synthetic fresh-cache identity; not a real cache proof\n")
    corpus = tmp_path / "synthetic-corpus"
    output = tmp_path / "series.json"
    harness = {name: coordinator.file_sha(tmp_path / name) for name in ("train_coco_gpu.py", "coco_loader.py")}
    plan = coordinator.make_plan([0], ["yes"])
    calls = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            spec = plan[len(calls)]
            calls.append(command)
            path = Path(command[command.index("--out") + 1])
            value, *_ = synthetic_trial(path.parent, spec["stage"], spec["backend"], 0)
            profile = {key: "synthetic" for key in coordinator.PROFILE_FIELDS}
            value = {**profile, **value, "harness_sources": harness,
                     "script_sha256": harness["train_coco_gpu.py"], "corpus_root": str(corpus),
                     "fresh_check_sha256": coordinator.file_sha(fresh)}
            if spec["start_epoch"]:
                checkpoint = Path(command[command.index("--resume-from") + 1])
                receipt = Path(command[command.index("--resume-receipt") + 1])
                value["resume_input"].update(path=str(checkpoint), sha256=coordinator.file_sha(checkpoint),
                                             receipt_sha256=coordinator.file_sha(receipt))
            if failure == "parity" and len(calls) == 2:
                value["final_state_sha256"] = "f" * 64
            self.code = 1 if failure == "exit" and len(calls) == 3 else 0
            path.write_text(json.dumps(value) + "\n")
            self.pid, self.returncode = 30000 + len(calls), None

        def wait(self, **kwargs):
            self.returncode = self.code
            return self.code

        def poll(self):
            return self.returncode

    monkeypatch.setattr(coordinator, "__file__", str(own))
    monkeypatch.setattr(coordinator.subprocess, "Popen", FakeProcess)
    # Even an external test interruption must not signal the invented PID.
    monkeypatch.setattr(coordinator.os, "killpg", lambda *_: None, raising=False)
    monkeypatch.setattr(coordinator.sys, "argv", [str(own), "--corpus", str(corpus), "--fresh-check", str(fresh),
                                                 "--out", str(output), "--workers", "0", "--overlaps", "yes"])
    if failure:
        with pytest.raises(ValueError, match="trial failed" if failure == "exit" else "loss/model mismatch"):
            coordinator.main()
    else:
        coordinator.main()
    report = json.loads(output.read_text())
    assert report["full_protocol_requested"] is False
    assert report["planned_trials"] == 9 and report["planned_epochs"] == 21
    assert not output.with_suffix(".tmp").exists()
    if failure:
        assert report["complete"] is False and "error" in report
        assert len(calls) == len(report["records"]) == (3 if failure == "exit" else 2)
        failed = report["records"][-1]
        assert "validated" not in failed
        assert Path(failed["path"]).is_file() and coordinator.file_sha(failed["path"]) == failed["sha256"]
        assert failed["returncode"] == (1 if failure == "exit" else 0)
    else:
        assert report["complete"] is True and report["full_protocol_complete"] is False
        assert len(calls) == len(report["records"]) == 9
        assert all(r["validated"] for r in report["records"])
        assert len(report["cohorts"]) == 3
        assert all(c["backends"] == ["reference", "mask", "both"] for c in report["cohorts"].values())
