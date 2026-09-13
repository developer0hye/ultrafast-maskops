"""Seal the sixty-process packet-loader comparison after independent qualification."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path("/home/yonghye/ultrafast-vision-build")
PREFIX = "mask-shared-loader-linux-full-v1"
SOURCE_PREFIX = "mask-shared-loader-linux-v1"
SOURCE = ROOT / (SOURCE_PREFIX + "-source")
INSTALLED = ROOT / "mask-shared-linux-v2-clean/lib/python3.12/site-packages"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    archive_path = ROOT / (PREFIX + "-evidence.tar.gz")
    receipt_path = ROOT / (PREFIX + "-preservation.json")
    assert not archive_path.exists() and not receipt_path.exists()
    state_path = ROOT / (PREFIX + "-state.json")
    state = json.loads(state_path.read_text())
    assert state["complete"] and state["timing_and_fresh_passed"]
    assert [c["label"] for c in state["commands"]] == ["overlap-yes", "overlap-no", "fresh-yes", "fresh-no"]
    identity_path = ROOT / (SOURCE_PREFIX + "-audit-identity.json")
    identity = json.loads(identity_path.read_text())
    negative_path = ROOT / (PREFIX + "-audit-negative.json")
    negative = json.loads(negative_path.read_text())
    assert negative["complete_controls_passed"] == ["yes", "no"] and negative["rejected"] == 92
    assert len(negative["cases"]) == 92
    assert len({(r["mode"], r["damage"]) for r in negative["cases"]}) == 92
    assert all(r["rejected"] for r in negative["cases"])
    assert negative["synthetic_bootstrap_controls"] == 3 and negative["real_bootstrap_metric_controls"] == 30
    assert negative["controller_state_sha256"] == sha(state_path.read_bytes())
    files = {
        "state.json": state_path.read_bytes(),
        "identity.json": identity_path.read_bytes(),
        "negative.json": negative_path.read_bytes(),
        "sealer.py": Path(__file__).read_bytes(),
    }
    for record in state["commands"]:
        assert record["returncode"] == 0
        data = Path(record["log"]).read_bytes()
        assert sha(data) == record["log_sha256"]
        files["controller-logs/" + record["label"] + ".log"] = data
    for mode in ("yes", "no"):
        report_path = ROOT / f"{PREFIX}-overlap-{mode}.json"
        fresh_path = ROOT / f"{PREFIX}-fresh-{mode}.json"
        audit_path = ROOT / f"{PREFIX}-{mode}-audit.json"
        audit = json.loads(audit_path.read_text())
        assert sha(json.dumps(audit, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()) == negative[
            "control_audit_canonical_sha256"
        ][mode]
        assert audit["complete_requested_samples"] and audit["completed_workers"] == 30
        assert audit["fresh_verified"] and audit["rounds"] == 5
        assert audit["scope"] == "five-pair performance protocol"
        assert sha(report_path.read_bytes()) == audit["report_sha256"] == negative["input_report_sha256"][mode]
        assert sha(report_path.read_bytes()) == state["completed_reports"][mode]
        assert sha(fresh_path.read_bytes()) == audit["fresh_reference_sha256"] == state["fresh_reports"][mode]
        assert sha(identity_path.read_bytes()) == audit["identity_sha256"]
        assert audit["auditor_sha256"] == negative["auditor_sha256"]
        assert all(
            len(metric["raw_pairs"]) == 5 and len(metric["paired_bootstrap_ci95"]) == 2
            for group in audit["summary"].values()
            for metric in group.values()
        )
        for name, path in (("report.json", report_path), ("fresh.json", fresh_path), ("audit.json", audit_path)):
            files[mode + "/" + name] = path.read_bytes()
        for directory, label, hashes in (
            (report_path.with_suffix(".runs"), "runs", audit["raw_sha256"]),
            (fresh_path.with_suffix(".runs"), "fresh.runs", audit["fresh_raw_sha256"]),
        ):
            assert {p.name for p in directory.iterdir()} == set(hashes)
            for name, digest in hashes.items():
                data = (directory / name).read_bytes()
                assert sha(data) == digest
                files[f"{mode}/{label}/{name}"] = data
    for name, digest in identity["source_sha256"].items():
        data = (SOURCE / name).read_bytes()
        assert sha(data) == digest
        files["sources/" + name] = data
        if name.startswith("python/"):
            assert (INSTALLED / name.removeprefix("python/")).read_bytes() == data
    for name, digest in identity["upstream_files"].items():
        data = (INSTALLED / "ultralytics" / name).read_bytes()
        assert sha(data) == digest
        files["upstream/" + name] = data
    for name in (
        "audit-shared-loader-v1.py",
        "qualify-shared-loader-full-audit-v1.py",
        "run-shared-loader-linux-full-v1.py",
        SOURCE_PREFIX + "-source.tar",
        SOURCE_PREFIX + "-source.json",
        PREFIX + "-launch-identity.json",
        PREFIX + "-audit-negative.log",
        PREFIX + "-audit-lint.json",
    ):
        files["validation/" + name] = (ROOT / name).read_bytes()
    assert sha(files["validation/audit-shared-loader-v1.py"]) == negative["auditor_sha256"]
    assert sha(files["validation/qualify-shared-loader-full-audit-v1.py"]) == negative["qualification_source_sha256"]
    assert sha(files["validation/run-shared-loader-linux-full-v1.py"]) == state["script_sha256"]
    assert sha(files["validation/" + SOURCE_PREFIX + "-source.tar"]) == state["source_archive_sha256"]
    files["sources/bench/verify_coco_loader.py"] = (SOURCE / "bench/verify_coco_loader.py").read_bytes()
    assert (
        sha(files["sources/bench/verify_coco_loader.py"])
        == "cbc01f8460c4fe1c2c8754b89d4fca02da4b736e59d94292518a5894bdb3fe91"
    )
    # Keep the prior failed-wheel evidence separate; embed only this exact
    # successful wheel qualification, which also contains the wheel and source.
    files["runtime-evidence.tar.gz"] = (ROOT / "mask-shared-linux-v2-qualification-evidence.tar.gz").read_bytes()
    assert sha(files["runtime-evidence.tar.gz"]) == "ecd5ebadb94f4c3dfcdeaa616f57c6914737a4b0bca0e80abe9ed2a33cc6701b"
    runtime_manifest = json.loads((ROOT / "mask-shared-linux-v2-qualification-preservation.json").read_text())
    wheel_name = "mask-shared-linux-v2-dist/ultrafast_maskops-0.1.0a1-cp312-cp312-linux_x86_64.whl"
    assert runtime_manifest["files"][wheel_name]["sha256"] == identity["wheel_sha256"]
    manifest = {name: {"bytes": len(data), "sha256": sha(data)} for name, data in files.items()}
    files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, data in sorted(files.items()):
            entry = tarfile.TarInfo(name)
            entry.size, entry.mtime = len(data), 0
            archive.addfile(entry, io.BytesIO(data))
    with tarfile.open(archive_path) as archive:
        assert {m.name: archive.extractfile(m).read() for m in archive.getmembers()} == files
    receipt = {
        "preservation_complete": True,
        "timing_and_fresh_passed": True,
        "independent_audit_passed": True,
        "production_qualified": False,
        "archive_sha256": sha(archive_path.read_bytes()),
        "archive_bytes": archive_path.stat().st_size,
        "file_count": len(files),
        "all_members_read_back": True,
        "manifest": manifest,
        "scope": "Sixty complete 5000-image loader processes: both mask modes, workers 0/2/8, five counterbalanced pairs per condition; two fresh-cache reference verifications, two independent audits, 92 damaged-evidence rejections, 30 real metric and three synthetic bootstrap oracle controls. CPU loader evidence only; no GPU training or production qualification claim.",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "manifest"}))


if __name__ == "__main__":
    main()
