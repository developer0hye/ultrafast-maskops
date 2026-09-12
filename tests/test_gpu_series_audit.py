"""Synthetic artifact corruption checks; these files are never benchmark data."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "bench/audit_gpu_series.py"
SPEC = importlib.util.spec_from_file_location("audit_gpu_series", SCRIPT)
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


def save_record(record, raw):
    path = Path(record["path"])
    data = (json.dumps(raw, indent=2) + "\n").encode()
    path.write_bytes(data)
    record["sha256"] = audit_module.sha(data)
    report = copy.deepcopy(raw)
    report.pop("resource_samples")
    for epoch in report["epochs"]:
        epoch.pop("all_batch_losses")
    record["report"] = report


def fixture(tmp_path, workers=0):
    parent, trial = tmp_path / "repeat.py", tmp_path / "train_coco_gpu.py"
    parent.write_text("synthetic parent identity, not executable benchmark code\n")
    trial.write_text("synthetic trial identity, not executable benchmark code\n")
    config = {
        "workers": [workers],
        "repeats": 5,
        "epochs": 2,
        "batch": 4,
        "imgsz": 640,
        "overlap": "yes",
        "corpus": "/synthetic/corpus",
        "fresh_check": "/synthetic/fresh.json",
    }
    settings = {
        "model": "yolo11n-seg.yaml",
        "seed": 912,
        "amp": False,
        "optimizer": "SGD",
        "pretrained": False,
        "deterministic": True,
        "close_mosaic": 0,
        "mask_ratio": 4,
        "overlap_mask": True,
        "batch": 4,
        "epochs": 2,
        "imgsz": 640,
        "lr0": 0.001,
        "cache": False,
        "rect": False,
        "fraction": 1.0,
    }
    records, summary = [], {}
    for repeat in range(5):
        order = ["reference", "mask", "both"]
        start = repeat % 3
        order = order[start:] + order[:start]
        if repeat % 2:
            order.reverse()
        for backend in order:
            path = tmp_path / f"r{repeat}-w{workers}-{backend}.json"
            args = {k: v for k, v in config.items() if k not in ("workers", "repeats")}
            args.update(workers=workers, backend=backend, out=str(path))
            command = ["synthetic-python", str(trial)]
            for name, value in args.items():
                command.extend(["--" + name.replace("_", "-"), str(value)])
            divisor = {"reference": 1, "mask": 2, "both": 4}[backend]
            epochs = []
            for epoch in range(2):
                seconds = (100 + repeat * 2) * (epoch + 1) / divisor
                epochs.append(
                    {
                        "epoch": epoch,
                        "seconds": seconds,
                        "images": 5000,
                        "batches": 1250,
                        "images_per_second": 5000 / seconds,
                        "cuda_peak_allocated_bytes": 100,
                        "cuda_peak_reserved_bytes": 200,
                        "loss_names": ["box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"],
                        "all_batch_losses": [[1.0, 1.0, 1.0, 1.0, 0.0] for _ in range(1250)],
                    }
                )
            raw = {
                "complete": True,
                "args": args,
                "script_sha256": audit_module.sha(trial.read_bytes()),
                "upstream_files": {},
                "packages": {},
                "torch": "synthetic",
                "numpy": "synthetic",
                "opencv": "synthetic",
                "gpu": "synthetic",
                "cuda": "synthetic",
                "python": "synthetic",
                "platform": "synthetic",
                "cpu_count": 12,
                "ram_bytes": 1000,
                "mask_build": {},
                "timing": "synthetic",
                "memory": "synthetic",
                "callbacks": "synthetic",
                "fixture": {
                    "files": 9952,
                    "bytes": 830426850,
                    "sha256": "8b079ad2d8472ff17b6cca9b0250e831472f12ef500f2b065125f6f39406c208",
                },
                "original_cache_sha256": "a" * 64,
                "fresh_check_sha256": "b" * 64,
                "initial_state_sha256": "c" * 64,
                "final_state_sha256": "d" * 64,
                "train_workers": workers,
                "val_workers": min(2 * workers, 12),
                "resolved_config": settings.copy(),
                "datasets": [
                    {
                        "mode": mode,
                        "images": 5000,
                        "seconds": 1.0,
                        "replaced_formats": int(backend != "reference"),
                        "native_cache_hit": False if backend == "both" else None,
                    }
                    for mode in ("train", "val")
                ],
                "epochs": epochs,
                "whole_job_seconds": sum(e["seconds"] for e in epochs) + 10,
                "resource_samples": [
                    {"time": 1.0, "family_rss_bytes": 100, "load": [0.0] * 3},
                    {"time": 1.2, "family_rss_bytes": 200, "load": [0.0] * 3},
                ],
                "sampled_peak_family_rss_bytes": 200,
            }
            record = {"repeat": repeat, "workers": workers, "backend": backend, "path": str(path), "command": command}
            save_record(record, raw)
            records.append(record)
    for backend, ratio in (("mask", 2), ("both", 4)):
        for epoch in range(2):
            reference = 104.0 * (epoch + 1)
            summary[f"workers={workers}/{backend}/epoch={epoch}"] = {
                "pairs": 5,
                "reference_median_s": reference,
                "candidate_median_s": reference / ratio,
                "reference_over_candidate_median_ratio": float(ratio),
                "paired_bootstrap_95_percent_interval": [float(ratio), float(ratio)],
            }
    series = {
        "complete": True,
        "script_sha256": audit_module.sha(parent.read_bytes()),
        "trial_script_sha256": audit_module.sha(trial.read_bytes()),
        "configuration": config,
        "records": records,
        "summary": summary,
    }
    path = tmp_path / "series.json"

    def run(allow_partial=False):
        path.write_text(json.dumps(series))
        return audit_module.audit(path, tmp_path, parent, trial, allow_partial)

    return series, run


@pytest.mark.parametrize("workers", [0, 8])
def test_full_audit_and_cpu_capped_validation(tmp_path, workers):
    series, run = fixture(tmp_path, workers)
    result = run()
    assert result["audit_passed"] and result["series_complete"]
    assert len(result["complete_backend_groups"]) == 5
    assert result["independently_recomputed_epoch_summary"] == series["summary"]


def test_partial_cannot_become_a_final_summary(tmp_path):
    series, run = fixture(tmp_path)
    series["complete"] = False
    series["records"] = series["records"][:8]
    series.pop("summary")
    with pytest.raises(ValueError, match="series incomplete"):
        run()
    result = run(allow_partial=True)
    assert result["validated_trials"] == 8 and not result["series_complete"]
    assert result["independently_recomputed_epoch_summary"] is None
    assert len(result["complete_backend_groups"]) == 2


@pytest.mark.parametrize(
    "fault, message",
    [
        ("checksum", "raw checksum mismatch"),
        ("loss", "loss trace differs"),
        ("truncated_loss", "loss trace truncated"),
        ("peak", "RSS maximum mismatch"),
        ("model", "final model differs"),
        ("compact", "compact/raw disagreement"),
    ],
)
def test_changed_raw_artifacts_are_rejected(tmp_path, fault, message):
    series, run = fixture(tmp_path)
    record = next(r for r in series["records"] if r["backend"] == "both")
    path = Path(record["path"])
    raw = json.loads(path.read_text())
    if fault == "checksum":
        path.write_text(path.read_text() + " ")
    elif fault == "compact":
        record["report"]["whole_job_seconds"] += 1
    else:
        if fault == "loss":
            raw["epochs"][0]["all_batch_losses"][0][1] += 0.01
        elif fault == "truncated_loss":
            raw["epochs"][0]["all_batch_losses"].pop()
        elif fault == "peak":
            raw["sampled_peak_family_rss_bytes"] += 1
        elif fault == "model":
            raw["final_state_sha256"] = "e" * 64
        # Update both checksums and compact metadata: relational checks must
        # detect the corruption even when the archive hash alone still agrees.
        save_record(record, raw)
    with pytest.raises(ValueError, match=message):
        run()


@pytest.mark.parametrize(
    "fault, message",
    [
        ("missing", "complete flag with missing trials"),
        ("order", "reordered records"),
        ("summary", "paired summary mismatch"),
    ],
)
def test_false_completion_and_statistics_are_rejected(tmp_path, fault, message):
    series, run = fixture(tmp_path)
    if fault == "missing":
        series["records"].pop()
    elif fault == "order":
        series["records"][0], series["records"][1] = series["records"][1], series["records"][0]
    else:
        next(iter(series["summary"].values()))["reference_over_candidate_median_ratio"] = 99.0
    with pytest.raises(ValueError, match=message):
        run()
