"""Audit the retained two-host COCO baseline without importing either backend.

This checks evidence consistency, not runtime correctness or new performance.
Run from any directory with NumPy installed. No fixture or native build needed.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench/results"
BASELINE = "7c508a6"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(name):
    return json.loads((RESULTS / name).read_text())


def source_bytes(name, expected):
    local = ROOT / name
    if local.exists() and sha(local.read_bytes()) == expected:
        return local.read_bytes()
    snapshot = RESULTS / "coco-baseline-server-source" / name
    if snapshot.exists() and sha(snapshot.read_bytes()) == expected:
        return snapshot.read_bytes()
    return subprocess.check_output(["git", "show", f"{BASELINE}:{name}"], cwd=ROOT)


def main():
    artifacts = {}
    for host in ("m2", "server"):
        name = f"coco-loader-{host}.json"
        report = read(name)
        report_sha = sha((RESULTS / name).read_bytes())
        runs = report["results"]
        assert len(runs) == 30 and set(report["summary"]) == {"0", "2", "8"}
        for path, expected in report["source_sha256"].items():
            assert sha(source_bytes(path, expected)) == expected, (host, path)
        outputs = {r["output_sha256"] for r in runs}
        assert len(outputs) == 1
        for r in runs:
            assert r["parity"] and r["images"] == r["verification_images"] == 5000
            assert r["verification_batches"] == 625 and len(r["batch_arrivals_s"]) == 625
            assert r["source_sha256"] == report["source_sha256"]
            assert r["extension_sha256"] == report["extension_sha256"]
        for workers, metrics in report["summary"].items():
            selected = [r for r in runs if r["workers"] == int(workers)]
            assert len(selected) == 10
            for backend in ("reference", "native"):
                assert sorted(r["round"] for r in selected if r["backend"] == backend) == list(range(5))
            for metric, summary in metrics.items():
                values = {
                    b: np.array([r[metric] for r in selected if r["backend"] == b]) for b in ("reference", "native")
                }
                for b, v in values.items():
                    assert summary[b] == {"median": float(np.median(v)), "p95": float(np.quantile(v, 0.95))}
                indices = np.random.default_rng(912).integers(0, 5, (10000, 5))
                ratios = np.median(values["reference"][indices], axis=1) / np.median(values["native"][indices], axis=1)
                assert summary["reference_over_native_ci95"] == np.quantile(ratios, [0.025, 0.975]).tolist()
        fresh = read(f"coco-fresh-{host}.json")
        assert fresh["benchmark_sha256"] == report_sha
        assert fresh["matched_runs"] == 30 and fresh["all_benchmark_outputs_match_fresh_reference"]
        assert fresh["fresh_reference"]["output_sha256"] in outputs
        assert fresh["fresh_reference"]["source_sha256"] == report["source_sha256"]
        assert fresh["fresh_reference"]["extension_sha256"] == report["extension_sha256"]
        for backend in ("reference", "native"):
            stage = read(f"coco-stages-{backend}-{host}.json")
            sample = RESULTS / f"coco-stages-{backend}-{host}-samples.json"
            assert stage["benchmark_sha256"] == report_sha and stage["raw_samples_sha256"] == sha(sample.read_bytes())
            assert stage["parity"] and stage["images"] == 5000 and stage["output_sha256"] in outputs
            assert stage["source_sha256"] == report["source_sha256"]
            assert stage["extension_sha256"] == report["extension_sha256"]
            raw = json.loads(sample.read_text())
            assert raw.keys() == stage["stats"].keys() == stage["expected_call_counts"].keys()
            for k, v in raw.items():
                assert len(v) == stage["expected_call_counts"][k] == stage["stats"][k]["calls"]
                assert sum(v) / 1e9 == stage["stats"][k]["inclusive_s"]
                assert stage["stats"][k]["median_ns"] == (float(np.median(v)) if v else None)
                assert stage["stats"][k]["p95_ns"] == (float(np.quantile(v, 0.95)) if v else None)
        audit = read(f"coco-cprofile-audit-{host}.json")
        for profile in audit["profiles"]:
            assert sha((ROOT / profile["artifact"]).read_bytes()) == profile["sha256"]
            assert profile["valid_for_component_timing"] == all(c["valid"] for c in profile["counts"].values())
        artifacts[host] = {"measured_runs": len(runs), "benchmark_sha256": report_sha, "output_sha256": outputs.pop()}
    print(json.dumps({"scope": "retained baseline evidence consistency audit", "verified": artifacts}, indent=2))


if __name__ == "__main__":
    main()
