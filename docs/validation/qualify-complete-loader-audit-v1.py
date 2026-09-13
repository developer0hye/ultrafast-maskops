"""Exercise full loader-audit rejection paths against retained completed records.

Run only after both timing matrices and their fresh-reference follow-up finish.
Only temporary copies of evidence are modified; no dataset or cache is touched.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path


def encoded(value):
    return json.dumps(value, indent=2) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--auditor", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    followup = json.loads((args.root / "mask-unit-scale-linux-loader-followup-v1.json").read_text())
    assert followup["complete"] is True, "fresh-reference follow-up is not finished"
    spec = importlib.util.spec_from_file_location("loader_auditor", args.auditor)
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    parent_fields = {"round", "parity", "sampled_family_peak_rss_bytes", "memory_sample_count",
                     "max_sampled_processes", "memory_sample_max_gap_s"}
    damages = ("missing_run", "order", "raw_parent", "incomplete_images", "wrong_overlap", "wrong_source",
               "output_parity", "bad_throughput", "wrong_topology", "missing_verification", "missing_parity",
               "memory_peak", "memory_count", "memory_order", "worker_phase", "extra_artifact",
               "fresh_binding", "fresh_output", "fresh_phase", "fresh_count", "cache_hash_missing", "aggregate")
    results, controls = [], {}
    for mode in ("yes", "no"):
        report_path = args.root / f"mask-unit-scale-linux-loader-overlap-{mode}-v1.json"
        runs_path = report_path.with_suffix(".runs")
        fresh_path = args.root / f"mask-unit-scale-linux-loader-fresh-{mode}-v1.json"
        original = json.loads(report_path.read_text())
        controls[mode] = auditor.audit(report_path, args.identity, runs_path, mode == "yes", fresh_path=fresh_path)
        assert controls[mode]["complete_requested_samples"] and controls[mode]["completed_workers"] == 30
        for damage in damages:
            with tempfile.TemporaryDirectory(prefix="mask-loader-audit-") as directory:
                tmp = Path(directory)
                report, fresh = tmp / "report.json", tmp / "fresh.json"
                runs, fresh_runs = tmp / "report.runs", tmp / "fresh.runs"
                shutil.copytree(runs_path, runs)
                shutil.copytree(fresh_path.with_suffix(".runs"), fresh_runs)
                value = copy.deepcopy(original)
                fresh_value = json.loads(fresh_path.read_text())
                row = value["results"][0]
                stem = f"{row['workers']}-{row['round']}-{row['backend']}"
                raw = runs / (stem + ".json")
                memory = runs / (stem + ".memory.json")
                consistent = False
                if damage == "missing_run":
                    value["results"].pop()
                elif damage == "order":
                    value["results"][0], value["results"][1] = value["results"][1], value["results"][0]
                elif damage == "raw_parent":
                    row["epoch_s"] += 1
                elif damage == "incomplete_images":
                    row["images"] -= 1
                    consistent = True
                elif damage == "wrong_overlap":
                    row["overlap"] = mode != "yes"
                    consistent = True
                elif damage == "wrong_source":
                    row["extension_sha256"] = "0" * 64
                    consistent = True
                elif damage == "output_parity":
                    row["output_sha256"] = "0" * 64
                    consistent = True
                elif damage == "bad_throughput":
                    row["images_per_s"] += 1
                    consistent = True
                elif damage == "wrong_topology":
                    row["topology"]["instances"] -= 1
                    consistent = True
                elif damage == "missing_verification":
                    row["verification_batches"] -= 1
                    consistent = True
                elif damage == "missing_parity":
                    row["parity"] = False
                elif damage == "memory_peak":
                    row["sampled_family_peak_rss_bytes"] += 1
                elif damage == "memory_count":
                    row["memory_sample_count"] += 1
                elif damage == "memory_order":
                    samples = json.loads(memory.read_text())
                    samples[1][0] = samples[0][0]
                    memory.write_text(encoded(samples))
                elif damage == "worker_phase":
                    (runs / (stem + ".phase")).write_text("timing")
                elif damage == "extra_artifact":
                    (runs / "extra.json").write_text("{}")
                elif damage == "fresh_binding":
                    fresh_value["benchmark_sha256"] = "0" * 64
                elif damage == "fresh_output":
                    fresh_value["fresh_reference"]["output_sha256"] = "0" * 64
                    (fresh_runs / "fresh-reference.json").write_text(encoded(fresh_value["fresh_reference"]))
                elif damage == "fresh_phase":
                    (fresh_runs / "fresh-reference.phase").write_text("timing")
                elif damage == "fresh_count":
                    fresh_value["matched_runs"] = 29
                elif damage == "cache_hash_missing":
                    fresh_value["fresh_cache_sha256"] = ""
                elif damage == "aggregate":
                    value["summary"]["0"]["epoch_s"]["native"]["median"] += 1
                if consistent:
                    raw.write_text(encoded({k: v for k, v in row.items() if k not in parent_fields}))
                # Rebind the fresh receipt to intentionally modified parent bytes,
                # so tests exercise semantic checks rather than only outer hashes.
                report.write_text(encoded(value))
                if damage != "fresh_binding":
                    fresh_value["benchmark_sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
                fresh.write_text(encoded(fresh_value))
                try:
                    auditor.audit(report, args.identity, runs, mode == "yes", fresh_path=fresh)
                except ValueError as error:
                    results.append({"mode": mode, "damage": damage, "rejected": True, "reason": str(error)})
                else:
                    raise AssertionError(f"accepted damaged {mode} evidence: {damage}")
    args.out.write_text(encoded({"complete_controls_passed": list(controls), "rejected": len(results), "cases": results,
                                "auditor_sha256": hashlib.sha256(args.auditor.read_bytes()).hexdigest(),
                                "qualification_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                "input_report_sha256": {k: v["report_sha256"] for k, v in controls.items()}}))
    print(f"Both complete controls passed; {len(results)} damaged records rejected.")


if __name__ == "__main__":
    main()
