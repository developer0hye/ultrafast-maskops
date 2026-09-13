"""Accept the valid allocation archive; reject separately rehashed forgeries.

Usage: python SCRIPT AUDITOR ORIGINAL_ARCHIVE NEW_OUTPUT_DIRECTORY
"""

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2) + "\n").encode()


auditor, original, out = map(Path, sys.argv[1:])
out.mkdir(exist_ok=False)
with tarfile.open(original) as archive:
    files = {m.name: archive.extractfile(m).read() for m in archive.getmembers()}
results = []
for mode in ("valid-control", "wrong-peak", "missing-worker", "missing-trace", "swapped-traces", "wrong-output"):
    target = original
    if mode != "valid-control":
        changed = dict(files)
        report = json.loads(files["results.json"])
        if mode == "wrong-peak":
            report["results"][0]["peak_tracked_allocation_bytes"] += 1
        elif mode == "missing-worker":
            report["results"].pop()
        elif mode == "missing-trace":
            del changed["traces/0-4-reference.bin"]
        elif mode == "swapped-traces":
            a, b = "traces/0-4-reference.bin", "traces/0-4-wrapper.bin"
            changed[a], changed[b] = changed[b], changed[a]
        elif mode == "wrong-output":
            report["results"][0]["output_hashes"][0] = "0" * 64 + ":uint8:(160, 160)"
        changed["results.json"] = encoded(report)
        changed["manifest.json"] = encoded({n: sha(v) for n, v in changed.items() if n != "manifest.json"})
        target = out / (mode + ".tar.gz")
        with tarfile.open(target, "x:gz") as archive:
            for name, data in sorted(changed.items()):
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    result = subprocess.run([sys.executable, str(auditor), str(target)], capture_output=True, check=False)
    (out / (mode + ".stdout")).write_bytes(result.stdout)
    (out / (mode + ".stderr")).write_bytes(result.stderr)
    assert (result.returncode == 0) == (mode == "valid-control"), (mode, result.stderr.decode())
    if mode == "valid-control":
        assert json.loads(result.stdout)["complete"] is True
    else:
        assert b"AssertionError" in result.stderr
    results.append({"name": mode, "returncode": result.returncode, "archive_sha256": sha(target.read_bytes())})
receipt = {
    "auditor_sha256": sha(auditor.read_bytes()),
    "driver_sha256": sha(Path(__file__).read_bytes()),
    "original_archive_sha256": sha(original.read_bytes()),
    "cases": results,
}
(out / "receipt.json").write_bytes(encoded(receipt))
print(json.dumps({"valid_controls": 1, "rejected_forged_archives": 5}))
