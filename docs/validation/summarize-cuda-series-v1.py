"""Summarize an independently audited complete GPU series, using only stdlib."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def distribution(values):
    return {"values": values, "median": statistics.median(values), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    series_bytes, audit_bytes = args.series.read_bytes(), args.audit.read_bytes()
    series, audit = json.loads(series_bytes), json.loads(audit_bytes)
    assert audit["audit_passed"] and audit["series_complete"] and series["complete"]
    assert audit["series_sha256"] == sha(series_bytes)
    assert audit["validated_trials"] == audit["planned_trials"] == len(series["records"]) == 45
    assert len(audit["complete_backend_groups"]) == 15 and not audit["pending_backend_groups"]
    rows = {}
    for record in series["records"]:
        data = (args.runs_dir / Path(record["path"]).name).read_bytes()
        assert sha(data) == record["sha256"]
        rows[record["repeat"], record["workers"], record["backend"]] = json.loads(data)
    memory = {}
    for workers in series["configuration"]["workers"]:
        for backend in ("reference", "mask", "both"):
            group = [rows[repeat, workers, backend] for repeat in range(5)]
            memory[f"workers={workers}/{backend}"] = {
                "whole_job_sampled_summed_family_rss_bytes": distribution(
                    [row["sampled_peak_family_rss_bytes"] for row in group]
                ),
                "epochs": {
                    str(epoch): {
                        name: distribution([row["epochs"][epoch][name] for row in group])
                        for name in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")
                    }
                    for epoch in range(2)
                },
            }
    summary = {
        "series_sha256": sha(series_bytes),
        "audit_sha256": sha(audit_bytes),
        "summary_script_sha256": sha(Path(__file__).read_bytes()),
        "configuration": series["configuration"],
        "epoch_timing": audit["independently_recomputed_epoch_summary"],
        "memory": memory,
        "largest_rss_sampling_gap_s": max(row["rss_largest_sampling_gap_s"] for row in audit["trial_checks"]),
        "scope": [
            "Five paired repetitions; first and second epoch remain separate.",
            "Epoch timing excludes initialization, validation and checkpoint saving.",
            "RSS is sampled summed family RSS over the whole job, not per-epoch working allocation; shared pages are counted repeatedly and between-sample spikes are missed.",
            "CUDA allocator peaks are per epoch, not total device usage.",
            "Memory distributions retain all five values and observed ranges, without claiming statistical confidence from the range.",
            "Shared-host overlap training on the frozen original Linux wheels; not an accuracy study or evidence for later masks-only/resize-ROI candidates.",
        ],
    }
    with args.out.open("x") as output:
        output.write(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
