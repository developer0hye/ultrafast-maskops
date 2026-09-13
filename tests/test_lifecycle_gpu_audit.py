"""Synthetic complete-artifact graphs and corruption tests; never training evidence."""

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


auditor = module("independent_lifecycle_audit", ROOT / "bench/audit_lifecycle_gpu.py")
fixtures = module("lifecycle_test_fixture", ROOT / "tests/test_lifecycle_coordinator.py")
coordinator = fixtures.coordinator


def save_record(record, raw):
    """Keep file hash and parent compaction coherent so semantic checks must work."""
    path = Path(record["path"])
    path.write_text(json.dumps(raw) + "\n")
    record["sha256"] = coordinator.file_sha(path)
    compact = {k: copy.deepcopy(v) for k, v in raw.items() if k not in ("resource_samples", "epochs")}
    compact["epochs"] = []
    for epoch in raw["epochs"]:
        compact["epochs"].append(
            {
                **{k: v for k, v in epoch.items() if k != "all_batch_losses"},
                "loss_sha256": coordinator.digest({"names": epoch["loss_names"], "values": epoch["all_batch_losses"]}),
            }
        )
    record["report"] = compact


def graph(tmp_path):
    source, runs = tmp_path / "source", tmp_path / "runs"
    (source / "bench").mkdir(parents=True)
    runs.mkdir()
    for name in ("train_coco_gpu.py", "coco_loader.py", "lifecycle_coco_gpu.py"):
        (source / "bench" / name).write_text("SYNTHETIC source identity, not executable benchmark code: " + name)
    harness = {name: coordinator.file_sha(source / "bench" / name) for name in ("train_coco_gpu.py", "coco_loader.py")}
    cache = tmp_path / "cache.bin"
    cache.write_bytes(b"SYNTHETIC cache, not COCO labels")
    fresh = tmp_path / "fresh.json"
    fresh.write_text(
        json.dumps(
            {
                "fresh_cache_sha256": coordinator.file_sha(cache),
                "fresh_reference": {"images": 5000},
                "matched_runs": 30,
                "all_benchmark_outputs_match_fresh_reference": True,
            }
        )
    )
    corpus = tmp_path / "synthetic-corpus"
    grid = {
        "complete": True,
        "full_protocol_complete": True,
        "full_protocol_requested": True,
        "planned_trials": 54,
        "planned_epochs": 126,
        "script_sha256": coordinator.file_sha(source / "bench/lifecycle_coco_gpu.py"),
        "harness_sources": harness,
        "corpus": str(corpus),
        "fresh_check": str(fresh),
        "fresh_check_sha256": coordinator.file_sha(fresh),
        "records": [],
        "cohorts": {},
    }
    plan = coordinator.make_plan([2, 0, 8], ["yes", "no"])
    references = {}
    for index, spec in enumerate(plan):
        constructor = tmp_path / "construct" / str(index)
        constructor.mkdir(parents=True)
        raw, _, old_path, *_ = fixtures.synthetic_trial(constructor, spec["stage"], spec["backend"], spec["workers"])
        path = runs / (spec["id"] + ".json")
        shutil.move(str(old_path.with_suffix(".run")), path.with_suffix(".run"))
        raw["args"].update(out=str(path), overlap=spec["overlap"])
        raw["resolved_config"]["overlap_mask"] = spec["overlap"] == "yes"
        raw.update(
            harness_sources=harness,
            script_sha256=harness["train_coco_gpu.py"],
            corpus_root=str(corpus),
            fresh_check_sha256=coordinator.file_sha(fresh),
            original_cache_sha256=coordinator.file_sha(cache),
            cpu_count=12,
            ram_bytes=32 * 1024**3,
            upstream_files={"synthetic": "f" * 64},
            packages={"synthetic": "e" * 64},
            python="synthetic",
            platform="synthetic",
            torch="synthetic",
            numpy="synthetic",
            opencv="synthetic",
            gpu="synthetic",
            cuda="synthetic",
        )
        val = min(spec["workers"] * 2, 12)
        raw["val_workers"] = val
        raw["worker_shutdown"]["val"] = {
            "worker_count": val,
            "worker_pids": list(range(5000, 5000 + val)),
            "worker_exitcodes": [0] * val,
            "workers_alive": [False] * val,
        }
        resume = None
        for checkpoint in raw["checkpoints"]:
            checkpoint["path"] = str(path.with_suffix(".run") / f"resume-epoch-{checkpoint['epoch']}.pt")
        if spec["start_epoch"]:
            reference = references[spec["reference_id"]]
            checkpoint = next(c for c in reference["checkpoints"] if c["epoch"] == spec["start_epoch"] - 1)
            ref_path = runs / (spec["reference_id"] + ".json")
            raw["args"].update(resume_from=checkpoint["path"], resume_receipt=str(ref_path))
            raw["resume_input"].update(
                path=checkpoint["path"], sha256=checkpoint["sha256"], receipt_sha256=coordinator.file_sha(ref_path)
            )
            resume = {k: v for k, v in raw["resume_input"].items() if k != "restored_state_expectation"}
        record = {
            "spec": spec,
            "path": str(path),
            "returncode": 0,
            "validated": True,
            "started_at_ns": 1000 + index * 2,
            "finished_at_ns": 1001 + index * 2,
            "command": coordinator.trial_command(
                spec, str(source / "synthetic-python"), source / "bench/train_coco_gpu.py", corpus, fresh, runs
            ),
        }
        path.with_suffix(".log").write_text("SYNTHETIC child log, no GPU job\n")
        record["log_sha256"] = coordinator.file_sha(path.with_suffix(".log"))
        compact, signature = coordinator.validate_trial(
            raw, spec, path, harness, corpus, coordinator.file_sha(fresh), resume
        )
        save_record(record, raw)
        assert record["report"] == compact
        grid["records"].append(record)
        key = f"w{spec['workers']}-{spec['overlap']}-{spec['stage']}"
        if spec["backend"] == "reference":
            grid["cohorts"][key] = {"signature": signature, "backends": []}
        grid["cohorts"][key]["backends"].append(spec["backend"])
        if spec["stage"] == "fresh" and spec["backend"] == "reference":
            references[spec["id"]] = raw
    grid["runtime_identity"] = {k: raw[k] for k in coordinator.PROFILE_FIELDS}
    grid_path = tmp_path / "grid.json"
    grid_path.write_text(json.dumps(grid) + "\n")
    return grid, grid_path, runs, source, fresh, cache


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    return graph(tmp_path_factory.mktemp("synthetic-lifecycle-audit"))


def check(data):
    _, grid, runs, source, fresh, cache = data
    return auditor.audit(grid, runs, source, fresh, cache)


def test_complete_graph_checks_54_trials_126_epochs_without_framework_import(baseline):
    result = check(baseline)
    assert result["audit_passed"] and result["trials"] == 54 and result["epochs"] == 126 and result["cohorts"] == 18
    assert len(result["validated"]) == 54
    assert "coordinator" not in auditor.__dict__ and "torch" not in auditor.__dict__


def test_relocated_artifacts_preserve_original_path_bindings(baseline, tmp_path):
    _, grid, runs, source, fresh, cache = baseline
    relocated = tmp_path / "relocated-runs"
    shutil.copytree(runs, relocated)
    assert auditor.audit(grid, relocated, source, fresh, cache)["audit_passed"]


@pytest.mark.parametrize(
    "damage",
    [
        "incomplete",
        "missing_last",
        "reordered",
        "duplicate_condition",
        "wrong_planned_epochs",
        "subset_as_full",
        "failed_returncode",
        "bool_returncode",
        "unvalidated",
        "wrong_command",
        "escaped_path",
        "raw_checksum",
        "log_checksum",
        "parent_compaction",
        "parent_cohort",
        "runtime_summary",
        "changed_source",
        "changed_cache",
        "changed_fresh",
        "missing_epoch",
        "short_loss",
        "loss_mismatch",
        "no_foreground",
        "bad_images",
        "bad_throughput",
        "bad_cuda",
        "bad_rss",
        "wrong_factory",
        "not_pinned",
        "reset_abort",
        "reset_pid_reuse",
        "unexpected_reset",
        "wrong_final_pool",
        "wrong_val_pool_size",
        "checkpoint_bytes",
        "checkpoint_size",
        "wrong_resume_receipt",
        "resume_setup_abort",
        "restored_optimizer_zero",
        "bool_start_epoch",
    ],
)
def test_corruption_rejected_even_with_coherent_checksums(baseline, damage):
    original, grid_path, runs, source, fresh, cache = baseline
    grid = copy.deepcopy(original)
    changed = {}

    def replace(path, payload):
        changed.setdefault(path, path.read_bytes())
        path.write_bytes(payload)

    record = grid["records"][1]  # fresh native, workers=2
    raw = json.loads(Path(record["path"]).read_text())
    raw_changed = False
    try:
        if damage == "incomplete":
            grid["complete"] = False
        elif damage == "missing_last":
            grid["records"].pop()
        elif damage == "reordered":
            grid["records"][1:3] = reversed(grid["records"][1:3])
        elif damage == "duplicate_condition":
            grid["records"][-9:] = copy.deepcopy(grid["records"][:9])
        elif damage == "wrong_planned_epochs":
            grid["planned_epochs"] = 125
        elif damage == "subset_as_full":
            grid["records"] = grid["records"][:9]
            grid["planned_trials"] = 9
            grid["planned_epochs"] = 21
        elif damage == "failed_returncode":
            record["returncode"] = 1
        elif damage == "bool_returncode":
            record["returncode"] = False
        elif damage == "unvalidated":
            record["validated"] = False
        elif damage == "wrong_command":
            record["command"][record["command"].index("--batch") + 1] = "2"
        elif damage == "escaped_path":
            record["path"] = str(runs / "../escaped.json")
        elif damage == "raw_checksum":
            record["sha256"] = "0" * 64
        elif damage == "log_checksum":
            replace(Path(record["path"]).with_suffix(".log"), b"changed log")
        elif damage == "parent_compaction":
            record["report"]["epochs"][0]["seconds"] = 99
        elif damage == "parent_cohort":
            grid["cohorts"]["w2-yes-fresh"]["signature"]["final_model"] = "a" * 64
        elif damage == "runtime_summary":
            grid["runtime_identity"]["gpu"] = "different"
        elif damage == "changed_source":
            replace(source / "bench/train_coco_gpu.py", b"changed source")
        elif damage == "changed_cache":
            replace(cache, b"changed cache")
        elif damage == "changed_fresh":
            replace(fresh, b"{}")
        elif damage == "checkpoint_bytes":
            replace(Path(raw["checkpoints"][0]["path"]), b"changed checkpoint")
        elif damage == "bool_start_epoch":
            record["spec"]["start_epoch"] = False
        else:
            raw_changed = True
            if damage in ("wrong_resume_receipt", "resume_setup_abort", "restored_optimizer_zero"):
                record = grid["records"][7]
                raw = json.loads(Path(record["path"]).read_text())
            if damage == "missing_epoch":
                raw["epochs"].pop()
            elif damage == "short_loss":
                raw["epochs"][0]["all_batch_losses"].pop()
            elif damage == "loss_mismatch":
                raw["epochs"][0]["all_batch_losses"][17][0] += 0.01
            elif damage == "no_foreground":
                for vector in raw["epochs"][0]["all_batch_losses"]:
                    vector[1] = 0
            elif damage == "bad_images":
                raw["epochs"][0]["images"] = 4999
            elif damage == "bad_throughput":
                raw["epochs"][0]["images_per_second"] += 1
            elif damage == "bad_cuda":
                raw["epochs"][0]["cuda_peak_reserved_bytes"] = 1
            elif damage == "bad_rss":
                raw["sampled_peak_family_rss_bytes"] += 1
            elif damage == "wrong_factory":
                raw["lifecycle"][3]["factory"] = "Format"
            elif damage == "not_pinned":
                raw["lifecycle"][1]["pinned_image"] = False
            elif damage == "reset_abort":
                raw["lifecycle"][3]["replaced_worker_exitcodes"][0] = -6
            elif damage == "reset_pid_reuse":
                raw["lifecycle"][3]["replaced_worker_pids"][0] = raw["lifecycle"][3]["worker_pids"][0]
            elif damage == "unexpected_reset":
                raw["lifecycle"][1]["replaced_worker_pids"] = []
            elif damage == "wrong_final_pool":
                raw["worker_shutdown"]["train"]["worker_pids"][0] += 10
            elif damage == "wrong_val_pool_size":
                raw["val_workers"] = 2
                for key in ("worker_pids", "worker_exitcodes", "workers_alive"):
                    raw["worker_shutdown"]["val"][key] = raw["worker_shutdown"]["val"][key][:2]
                raw["worker_shutdown"]["val"]["worker_count"] = 2
            elif damage == "checkpoint_size":
                raw["checkpoints"][0]["bytes"] += 1
            elif damage == "wrong_resume_receipt":
                raw["resume_input"]["receipt_sha256"] = "f" * 64
            elif damage == "resume_setup_abort":
                raw["lifecycle"][0]["replaced_worker_exitcodes"][0] = -6
            elif damage == "restored_optimizer_zero":
                raw["resume_input"]["restored_state_expectation"]["optimizer_states"] = 0
            else:
                raise AssertionError(damage)
        if raw_changed:
            path = Path(record["path"])
            changed.setdefault(path, path.read_bytes())
            save_record(record, raw)
        replace(grid_path, (json.dumps(grid) + "\n").encode())
        with pytest.raises((ValueError, KeyError, TypeError, OSError)):
            check(baseline)
    finally:
        for path, payload in changed.items():
            path.write_bytes(payload)


@pytest.mark.parametrize("payload", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_noncanonical_json_rejected(tmp_path, payload):
    path = tmp_path / "invalid.json"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        auditor.load(path)


def test_cli_preserves_failure_and_refuses_to_overwrite_existing_receipt(baseline, tmp_path):
    grid, _, runs, source, fresh, cache = baseline
    partial = tmp_path / "partial.json"
    value = copy.deepcopy(grid)
    value["complete"] = False
    partial.write_text(json.dumps(value))
    output = tmp_path / "audit.json"
    command = [
        sys.executable,
        str(ROOT / "bench/audit_lifecycle_gpu.py"),
        "--grid",
        str(partial),
        "--runs",
        str(runs),
        "--source",
        str(source),
        "--fresh-check",
        str(fresh),
        "--reference-cache",
        str(cache),
        "--out",
        str(output),
    ]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 1
    saved = output.read_bytes()
    assert json.loads(saved)["audit_passed"] is False
    assert "complete full grid required" in json.loads(saved)["error"]
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0 and output.read_bytes() == saved
