"""Seal the completed Linux 75-trace allocation series and frozen identity."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

root = Path("/home/yonghye/ultrafast-vision-build")
source = Path("/home/yonghye/ultrafast-maskops-unit-scale-v1")
runs = root / "mask-unit-scale-linux-allocations-v1"
files = {f"traces/{p.name}": p.read_bytes() for p in runs.glob("*.bin")}
assert len(files) == 75
files.update(
    {
        "results.json": (runs / "results.json").read_bytes(),
        "run.log": (root / "mask-unit-scale-linux-allocations-v1.log").read_bytes(),
        "descriptor-before.json": (root / "mask-unit-scale-linux-descriptor-v1.json").read_bytes(),
        "descriptor-after.json": (root / "mask-unit-scale-linux-descriptor-after-allocations-v1.json").read_bytes(),
        "oracle.json": (root / "mask-unit-vs-resize-roi-linux-v1.runs/oracle-candidate.json").read_bytes(),
        "wheel.whl": (
            root / "mask-unit-scale-linux-dist-v1/ultrafast_maskops-0.1.0a1-cp312-cp312-linux_x86_64.whl"
        ).read_bytes(),
        "validation/passed.json": (root / "mask-unit-scale-linux-passed-v1.json").read_bytes(),
    }
)
for name in ["bench/allocations.py", "bench/feasibility.py", "tests/reference.py", "src/bindings.cpp"]:
    files["sources/" + Path(name).name] = (source / name).read_bytes()
sha = lambda data: hashlib.sha256(data).hexdigest()
files["manifest.json"] = (json.dumps({n: sha(v) for n, v in files.items()}, indent=2) + "\n").encode()
out = root / "mask-unit-scale-linux-allocations-evidence-v1.tar.gz"
assert not out.exists()
with tarfile.open(out, "w:gz") as archive:
    for name, data in sorted(files.items()):
        member = tarfile.TarInfo(name)
        member.size, member.mtime = len(data), 0
        archive.addfile(member, io.BytesIO(data))
with tarfile.open(out) as archive:
    actual = {m.name: archive.extractfile(m).read() for m in archive.getmembers()}
assert actual == files
print(
    json.dumps({"archive": str(out), "sha256": sha(out.read_bytes()), "files": len(files), "bytes": out.stat().st_size})
)
