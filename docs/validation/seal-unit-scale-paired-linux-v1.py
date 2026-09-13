"""Seal complete raw measurements, tested binaries and validation artifacts."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

R = Path("/home/yonghye/ultrafast-vision-build")
S = Path("/home/yonghye/ultrafast-maskops-unit-scale-v1")


def sha(data):
    return hashlib.sha256(data).hexdigest()


for base in ("resize-roi", "masks-only"):
    path = R / f"mask-unit-vs-{base}-linux-v1.json"
    report = json.loads(path.read_bytes())
    assert report["complete"] is True and len(report["records"]) == 270
    files = {"series.json": path.read_bytes(), "run.log": path.with_suffix(".log").read_bytes()}
    for p in sorted(path.with_suffix(".runs").iterdir()):
        assert p.is_file()
        files["runs/" + p.name] = p.read_bytes()
    for src, name in [
        ("bench/compare_wheels.py", "harness.py"),
        ("bench/feasibility.py", "feasibility.py"),
        ("tests/reference.py", "reference.py"),
    ]:
        files["sources/" + name] = (S / src).read_bytes()
    for label, version in [("baseline", base), ("candidate", "unit-scale")]:
        source = Path("/home/yonghye") / f"ultrafast-maskops-{version}-v1"
        for name in [
            "src/bindings.cpp",
            "CMakeLists.txt",
            "src/build_profile.h.in",
            "python/ultrafast_maskops/__init__.py",
        ]:
            files[f"sources/{label}/{name}"] = (source / name).read_bytes()
        prefix = f"mask-{version}-linux"
        files[f"wheels/{label}.whl"] = (
            R / f"{prefix}-dist-v1/ultrafast_maskops-0.1.0a1-cp312-cp312-linux_x86_64.whl"
        ).read_bytes()
        for suffix in ["tests-v1.xml", "tests-v1.log", "freeze-v1.txt", "build-v1.log"]:
            files[f"validation/{label}-{suffix}"] = (R / f"{prefix}-{suffix}").read_bytes()
        files[f"validation/{label}-wheel-validation.json"] = (
            R / f"{prefix}-dist-v1/wheel-validation.json"
        ).read_bytes()
    files["validation/candidate-passed-v1.json"] = (R / "mask-unit-scale-linux-passed-v1.json").read_bytes()
    for p in sorted((S / "tests").glob("*.py")):
        files["sources/candidate/tests/" + p.name] = p.read_bytes()
    files["manifest.json"] = (json.dumps({name: sha(data) for name, data in files.items()}, indent=2) + "\n").encode()
    out = R / f"mask-unit-vs-{base}-linux-evidence-v1.tar.gz"
    assert not out.exists()
    with tarfile.open(out, "w:gz") as tar:
        for name, data in sorted(files.items()):
            item = tarfile.TarInfo(name)
            item.size = len(data)
            item.mtime = 0
            tar.addfile(item, io.BytesIO(data))
    with tarfile.open(out) as tar:
        actual = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    assert actual == files
    print(
        json.dumps(
            {"archive": str(out), "sha256": sha(out.read_bytes()), "bytes": out.stat().st_size, "files": len(files)}
        ),
        flush=True,
    )
