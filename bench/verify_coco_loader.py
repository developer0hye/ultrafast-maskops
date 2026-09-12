"""Rebuild the trusted fixture's label cache and verify complete benchmark outputs.

Run only after the timing job has finished: this explicitly replaces the generated
Ultralytics label cache. It never deletes image/annotation input files. The fresh
reference pass validates that an older cache did not make both backends agree on
stale labels. Its timings are not added to the repeated performance experiment.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from coco_loader import COCO_FINGERPRINT, fingerprint, sha, validate_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    validate_profile()
    root = args.corpus.resolve()
    report = json.loads(args.benchmark.read_text())
    assert "summary" in report, "wait for the benchmark to finish"
    assert fingerprint(root) == COCO_FINGERPRINT
    config = report["results"][0]
    for r in report["results"]:
        assert all(r[k] == config[k] for k in ["batch_size", "imgsz", "mask_ratio", "overlap", "images"])
    work = args.out.with_suffix(".runs")
    work.mkdir(parents=True, exist_ok=False)
    cache_path = root / "labels/val2017.cache"
    old_hash = sha(cache_path.read_bytes()) if cache_path.exists() else None
    cache_path.unlink(missing_ok=True)
    result = work / "fresh-reference.json"
    command = [
        sys.executable,
        str(Path(__file__).with_name("coco_loader.py")),
        "--corpus",
        str(root),
        "--backend",
        "reference",
        "--worker-count",
        "0",
        "--result",
        str(result),
        "--marker",
        str(work / "fresh-reference.phase"),
        "--batch",
        str(config["batch_size"]),
        "--imgsz",
        str(config["imgsz"]),
        "--mask-ratio",
        str(config["mask_ratio"]),
        "--overlap",
        "yes" if config["overlap"] else "no",
        "--limit",
        str(config["images"]),
    ]
    with (work / "fresh-reference.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    value = json.loads(result.read_text())
    assert value["source_sha256"] == report["source_sha256"]
    assert all(r["output_sha256"] == value["output_sha256"] for r in report["results"])
    assert cache_path.is_file() and fingerprint(root) == COCO_FINGERPRINT
    output = {
        "scope": "fresh original YOLODataset cache rebuild and full output verification, not a performance sample",
        "benchmark_sha256": sha(args.benchmark.read_bytes()),
        "script_sha256": sha(Path(__file__).read_bytes()),
        "previous_cache_sha256": old_hash,
        "fresh_cache_sha256": sha(cache_path.read_bytes()),
        "fresh_reference": value,
        "all_benchmark_outputs_match_fresh_reference": True,
        "matched_runs": len(report["results"]),
    }
    with args.out.open("x") as stream:
        json.dump(output, stream, indent=2)
    print(f"Fresh reference matches all {len(report['results'])} benchmark outputs", flush=True)


if __name__ == "__main__":
    main()
