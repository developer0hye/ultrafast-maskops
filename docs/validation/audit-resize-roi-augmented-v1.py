"""Audit a sealed real-data augmentation archive without importing either backend.

Usage: python audit-resize-roi-augmented-v1.py ARCHIVE > RECEIPT
Checks retained evidence consistency, not a new runtime execution or performance.
"""

import hashlib
import json
import sys
import tarfile
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


archive_path = Path(sys.argv[1])
with tarfile.open(archive_path) as archive:
    members = archive.getmembers()
    assert all(member.isfile() for member in members)
    assert len({member.name for member in members}) == len(members)
    files = {member.name: archive.extractfile(member).read() for member in members}
manifest = json.loads(files["manifest.json"])
assert set(manifest) == set(files) - {"manifest.json"}
assert all(sha(files[name]) == digest for name, digest in manifest.items())
report = json.loads(files["report.json"])
baseline = json.loads(files["baseline.json"])
fresh = json.loads(files["fresh.json"])
installed = json.loads(files["installed-receipt.json"])
assert report["complete"] is True and len(report["results"]) == 9
assert report["scope"] == "full augmented real COCO target parity"
assert report["candidate"] == "mask" and report["dataset_artifact"] is None
assert type(report["overlap"]) is bool
assert report["batch_size"] == 8 and report["imgsz"] == 640 and report["mask_ratio"] == 4
assert report["upstream_commit"] == "795a556942a12fe0124cf767888194a1d0b83e2e"
assert report["upstream_commit"] == baseline["upstream_commit"]
assert report["benchmark_role"] == "reference cache provenance only; not candidate performance evidence"
assert report["benchmark_sha256"] == sha(files["baseline.json"]) == fresh["benchmark_sha256"]
assert report["fresh_check_sha256"] == sha(files["fresh.json"])
assert report["cache_sha256"] == fresh["fresh_cache_sha256"]
assert fresh["all_benchmark_outputs_match_fresh_reference"] is True
assert fresh["matched_runs"] == len(baseline["results"])
assert "summary" in baseline and len(baseline["results"]) >= 10
assert all(row["images"] == 5000 for row in baseline["results"])
assert all(row["output_sha256"] == fresh["fresh_reference"]["output_sha256"] for row in baseline["results"])
for field in ("source_sha256", "extension_sha256"):
    assert fresh["fresh_reference"][field] == baseline[field]
assert report["extension_sha256"] == installed["extension_sha256"]
assert report["extension_sha256"] == "9b86c9dca9e4a694c57e14ba4d31ef8b303a47404f45849e181af5fad2d78325"
assert report["source_sha256"]["src/bindings.cpp"] == "c63940b67b291123d6d5d895858cb981afaa17bd1aea94856cd9f40dcebb1813"
for name, digest in report["source_sha256"].items():
    assert sha(files["source/" + name]) == digest, name
for key, name in (
    ("bindings_sha256", "src/bindings.cpp"),
    ("cmake_sha256", "CMakeLists.txt"),
    ("template_sha256", "src/build_profile.h.in"),
):
    assert installed["native_build"][key] == report["source_sha256"][name], name
assert sha(files["source/bench/verify_augmented_coco.py"]) == report["script_sha256"]
assert report["stress_overrides"] == {
    "mosaic": 1.0,
    "mixup": 1.0,
    "copy_paste": 1.0,
    "copy_paste_mode": "flip",
    "cutmix": 0.0,
    "degrees": 15.0,
    "translate": 0.15,
    "scale": 0.5,
    "shear": 5.0,
    "perspective": 0.0005,
    "flipud": 0.5,
    "fliplr": 0.5,
    "bgr": 0.5,
}
summaries, expected_raw = [], set()
for workers in (0, 2, 8):
    rows = []
    for trial, backend in enumerate(("reference", "reference", "native")):
        name = f"runs/w{workers}-{trial}-{backend}.json"
        expected_raw.add(name)
        raw = json.loads(files[name])
        matches = [r for r in report["results"] if r["workers"] == workers and r["trial"] == trial]
        assert len(matches) == 1 and matches[0]["matches_reference"] is True
        assert raw == {k: v for k, v in matches[0].items() if k not in ("trial", "matches_reference")}
        assert raw["backend"] == backend and raw["candidate"] == "mask"
        assert raw["dataset_artifact"] is None and raw["dataset_class"] == "YOLODataset"
        assert raw["annotation_cache_hit"] is None
        assert raw["workers"] == workers and raw["seed"] == 912
        assert raw["images"] == 5000 and len(raw["batches"]) == 625
        assert [b["batch"] for b in raw["batches"]] == list(range(625))
        assert all(b["images"] == 8 for b in raw["batches"])
        assert sum(b["instances"] for b in raw["batches"]) == raw["augmented_instances"]
        assert raw["hyp"]["overlap_mask"] is report["overlap"] and raw["hyp"]["mask_ratio"] == 4
        for key, value in report["stress_overrides"].items():
            assert raw["hyp"][key] == value, key
        assert {"img", "cls", "bboxes", "batch_idx", "masks", "sem_masks", "im_file"} <= set(raw["fields"])
        assert raw["mask_dtypes"] == ["torch.uint8"]
        for key in ("source_sha256", "extension_sha256", "script_sha256", "cache_sha256", "packages"):
            assert raw[key] == report[key], key
        rows.append(raw)
    for raw in rows[1:]:
        for key in (
            "batches",
            "output_sha256",
            "hyp",
            "fields",
            "mask_dtypes",
            "augmented_instances",
            "objects_per_image",
        ):
            assert raw[key] == rows[0][key], key
    summaries.append(
        {
            "workers": workers,
            "runs": 3,
            "images_per_run": 5000,
            "batches_per_run": 625,
            "output_sha256": rows[0]["output_sha256"],
            "augmented_instances": rows[0]["augmented_instances"],
            "objects_per_image": rows[0]["objects_per_image"],
            "mask_dtypes": rows[0]["mask_dtypes"],
        }
    )
assert {name for name in files if name.startswith("runs/") and name.endswith(".json")} == expected_raw
print(
    json.dumps(
        {
            "scope": "complete Linux ROI augmented mask-only output evidence; hashing inside iteration, no speed or memory claim",
            "complete": True,
            "overlap": report["overlap"],
            "groups": summaries,
            "archive_sha256": sha(archive_path.read_bytes()),
            "archive_files": len(files),
            "auditor_sha256": sha(Path(__file__).read_bytes()),
            "report_sha256": sha(files["report.json"]),
            "extension_sha256": report["extension_sha256"],
            "source_sha256": report["source_sha256"],
            "limitations": [
                "No corpus rerun by this auditor; source worker code records pre/post corpus and cache fingerprint checks.",
                "One fixed augmentation seed per worker count; streams may differ between worker counts.",
                "Only mask preparation replaced; original dataset scanner retained.",
                "Does not establish performance, sanitizer coverage, other resolutions or cross-platform parity.",
            ],
        },
        indent=2,
    )
)
