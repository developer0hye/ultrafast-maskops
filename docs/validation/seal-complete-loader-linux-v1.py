"""Seal all 60 loader runs, fresh-reference evidence and qualified full audits."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path("/home/yonghye/ultrafast-vision-build")
SOURCE = Path("/home/yonghye/ultrafast-maskops-unit-scale-v1")
INSTALLED = ROOT / "mask-unit-scale-linux-clean-v1/lib/python3.12/site-packages"
OUT = ROOT / "mask-unit-scale-linux-loader-complete-evidence-v1.tar.gz"


def sha(data):
    return hashlib.sha256(data).hexdigest()


assert not OUT.exists()
identity_path = ROOT / "mask-unit-scale-linux-loader-identity-v1.json"
identity = json.loads(identity_path.read_text())
followup_path = ROOT / "mask-unit-scale-linux-loader-followup-v1.json"
followup = json.loads(followup_path.read_text())
assert followup["complete"] and len(followup["steps"]) == 4
negative_path = ROOT / "mask-unit-scale-linux-loader-audit-negative-v1.json"
negative = json.loads(negative_path.read_text())
assert negative["complete_controls_passed"] == ["yes", "no"] and negative["rejected"] == 44
files = {"identity.json": identity_path.read_bytes(), "followup.json": followup_path.read_bytes(),
         "negative.json": negative_path.read_bytes(), "sealer.py": Path(__file__).read_bytes()}
for mode in ("yes", "no"):
    report = ROOT / f"mask-unit-scale-linux-loader-overlap-{mode}-v1.json"
    fresh = ROOT / f"mask-unit-scale-linux-loader-fresh-{mode}-v1.json"
    audit_path = ROOT / f"mask-unit-scale-linux-loader-complete-{mode}-audit-v1.json"
    audit = json.loads(audit_path.read_text())
    assert audit["completed_workers"] == 30 and audit["complete_requested_samples"]
    assert sha(report.read_bytes()) == audit["report_sha256"] == negative["input_report_sha256"][mode]
    assert sha(identity_path.read_bytes()) == audit["identity_sha256"]
    assert sha(fresh.read_bytes()) == audit["fresh_reference_sha256"]
    assert audit["auditor_sha256"] == negative["auditor_sha256"]
    for name, path in (("report.json", report), ("report.log", report.with_suffix(".log")),
                       ("fresh.json", fresh), ("audit.json", audit_path)):
        files[mode + "/" + name] = path.read_bytes()
    for directory, label, expected in ((report.with_suffix(".runs"), "runs", audit["raw_sha256"]),
                                       (fresh.with_suffix(".runs"), "fresh.runs", audit["fresh_raw_sha256"])):
        assert {p.name for p in directory.iterdir()} == set(expected)
        for name, value in expected.items():
            data = (directory / name).read_bytes()
            assert sha(data) == value
            files[f"{mode}/{label}/{name}"] = data
    for index in (0, 1):
        log = ROOT / f"mask-unit-scale-linux-loader-followup-{mode}-{index}-v1.log"
        files[f"{mode}/followup-{index}.log"] = log.read_bytes()
for name, expected in identity["source_sha256"].items():
    data = (SOURCE / name).read_bytes()
    assert sha(data) == expected
    files["sources/" + name] = data
for name, expected in identity["upstream_files"].items():
    data = (INSTALLED / "ultralytics" / name).read_bytes()
    assert sha(data) == expected
    files["upstream/" + name] = data
for name in ("audit-unit-scale-loader-v1.py", "finish-unit-scale-linux-loader-v1.py",
             "run-unit-scale-linux-loader-v1.sh", "qualify-complete-loader-audit-v1.py",
             "mask-unit-scale-linux-loader-audit-negative-v1.log", "mask-unit-scale-linux-passed-v1.json",
             "mask-unit-scale-linux-tests-v1.xml", "mask-unit-scale-linux-tests-v1.log",
             "mask-unit-scale-linux-descriptor-v1.json"):
    files["validation/" + name] = (ROOT / name).read_bytes()
assert sha(files["validation/audit-unit-scale-loader-v1.py"]) == negative["auditor_sha256"]
assert sha(files["validation/qualify-complete-loader-audit-v1.py"]) == negative["qualification_source_sha256"]
files["sources/bench/verify_coco_loader.py"] = (SOURCE / "bench/verify_coco_loader.py").read_bytes()
assert sha(files["sources/bench/verify_coco_loader.py"]) == "b17bbcbb736f867c8f066242304bd22e1f76cce95ace63b75f8adaec830e613d"
wheel = ROOT / "mask-unit-scale-linux-dist-v1/ultrafast_maskops-0.1.0a1-cp312-cp312-linux_x86_64.whl"
files["wheel.whl"] = wheel.read_bytes()
assert sha(files["wheel.whl"]) == identity["wheel_sha256"]
manifest = {name: sha(data) for name, data in files.items()}
files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
with tarfile.open(OUT, "w:gz") as archive:
    for name, data in sorted(files.items()):
        entry = tarfile.TarInfo(name)
        entry.size, entry.mtime = len(data), 0
        archive.addfile(entry, io.BytesIO(data))
with tarfile.open(OUT) as archive:
    assert {m.name: archive.extractfile(m).read() for m in archive.getmembers()} == files
receipt = {"archive_sha256": sha(OUT.read_bytes()), "files": len(files), "bytes": OUT.stat().st_size,
           "all_members_read_back": True, "manifest": manifest,
           "scope": "Complete 60-process real-image DataLoader comparison; full raw memory/arrival samples, two rebuilt-cache reference verifications, independently audited statistics and 44 rejection cases. Input image/label corpus remains external."}
with (ROOT / "mask-unit-scale-linux-loader-complete-preservation-v1.json").open("x") as stream:
    stream.write(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k != "manifest"}))
