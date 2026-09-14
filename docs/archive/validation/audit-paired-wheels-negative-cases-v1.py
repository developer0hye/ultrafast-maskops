"""Exercise forged-but-rehashed reports against the paired-wheel auditor.

Usage: python SCRIPT AUDITOR ORIGINAL_ARCHIVE NEW_OUTPUT_DIRECTORY
All mutations are separate archives; original evidence is never modified.
"""

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

auditor, original, out = map(Path, sys.argv[1:])
out.mkdir(exist_ok=False)
with tarfile.open(original) as archive:
    files = {m.name: archive.extractfile(m).read() for m in archive.getmembers()}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def run(name, path, expect_success):
    result = subprocess.run([sys.executable, str(auditor), str(path)], capture_output=True, check=False)
    (out / (name + ".stdout")).write_bytes(result.stdout)
    (out / (name + ".stderr")).write_bytes(result.stderr)
    assert (result.returncode == 0) is expect_success, (name, result.stderr.decode())
    if expect_success:
        assert json.loads(result.stdout)["measured_processes"] == 270
    else:
        assert b"AssertionError" in result.stderr, result.stderr
    return {"name": name, "returncode": result.returncode, "archive_sha256": sha(path.read_bytes())}


results = [run("valid-control", original, True)]
for mode in (
    "missing-ci-endpoint",
    "forged-ci",
    "reordered-pair",
    "missing-worker",
    "changed-output",
    "changed-samples",
    "orphan-worker",
    "wrong-empty-layout",
):
    changed = dict(files)
    report = json.loads(files["series.json"])

    def update_worker(entry, transform, changed=changed):
        name = "runs/" + Path(entry["path"]).name
        raw = json.loads(changed[name])
        transform(raw)
        changed[name] = encoded(raw)
        entry["sha256"] = sha(changed[name])
        entry["report"] = {k: v for k, v in raw.items() if k != "descriptor"}

    if mode == "missing-ci-endpoint":
        report["summary"][0]["median_ns"]["paired_bootstrap_ci95"].pop()
    elif mode == "forged-ci":
        report["summary"][0]["median_ns"]["paired_bootstrap_ci95"][1] *= 2
    elif mode == "reordered-pair":
        report["records"][0], report["records"][1] = report["records"][1], report["records"][0]
    elif mode == "missing-worker":
        report["records"].pop()
    elif mode == "changed-output":

        def alter(raw):
            raw["output_hashes"][0]["sha256"] = "0" * 64

        update_worker(report["records"][0], alter)
    elif mode == "changed-samples":

        def alter(raw):
            raw["samples_ns"] = [n * 2 for n in raw["samples_ns"]]
            raw["median_ns"] *= 2

        update_worker(report["records"][0], alter)
    elif mode == "orphan-worker":
        changed["runs/extra.json"] = files["runs/r0-c0-overlap-baseline.json"]
    elif mode == "wrong-empty-layout":
        for entry in report["oracles"].values():
            entry["report"]["expected"]["0"]["masks"][0]["dtype"] = "uint8"
            name = "runs/" + Path(entry["path"]).name
            changed[name] = encoded(entry["report"])
            entry["sha256"] = sha(changed[name])
        for entry in report["records"]:
            if entry["case"] == 0 and entry["operation"] == "masks":

                def alter(raw):
                    raw["output_hashes"][0]["dtype"] = "uint8"

                update_worker(entry, alter)
    changed["series.json"] = encoded(report)
    changed["manifest.json"] = encoded({k: sha(v) for k, v in changed.items() if k != "manifest.json"})
    target = out / (mode + ".tar.gz")
    with tarfile.open(target, "x:gz") as archive:
        for name, data in sorted(changed.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    results.append(run(mode, target, False))
receipt = {
    "scope": "one valid complete archive accepted; eight forged archives rejected after all modified checksums were recomputed",
    "auditor_sha256": sha(auditor.read_bytes()),
    "driver_sha256": sha(Path(__file__).read_bytes()),
    "original_archive_sha256": sha(original.read_bytes()),
    "cases": results,
}
(out / "receipt.json").write_bytes(encoded(receipt))
print(json.dumps({"valid_controls": 1, "rejected_forged_archives": 8}))
