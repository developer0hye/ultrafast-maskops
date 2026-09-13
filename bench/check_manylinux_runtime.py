"""Check the repaired wheel's tags, installed ELF and base-glibc runtime."""

import hashlib
import importlib
import json
import os
import subprocess
from pathlib import Path
from zipfile import ZipFile


def main():
    root = Path(__file__).resolve().parents[1]
    packages = [p for p in ("ultrafast_maskops", "ultrafast_yolo_dataset") if (root / "python" / p).is_dir()]
    assert len(packages) == 1
    native = importlib.import_module(packages[0] + "._native")
    extension = Path(native.__file__)
    assert os.confstr("CS_GNU_LIBC_VERSION") == "glibc 2.28"
    assert not os.environ.get("LD_LIBRARY_PATH"), "runtime must not use GCC toolset library paths"
    wheel = next((root / "dist").glob("*.whl"))
    original = next((root / "manylinux-evidence/original-dist").glob("*.whl"))
    with ZipFile(wheel) as repaired, ZipFile(original) as before:
        metadata = next(n for n in repaired.namelist() if n.endswith(".dist-info/WHEEL"))
        tags = [line[5:] for line in repaired.read(metadata).decode().splitlines() if line.startswith("Tag: ")]
        assert any(t.endswith("-manylinux_2_28_x86_64") for t in tags), tags
        added = sorted(set(repaired.namelist()) - set(before.namelist()))
        assert all(".dist-info/" in n for n in added), "new bundled libraries require notice review: " + repr(added)
        native_members = [n for n in repaired.namelist() if n.endswith(".so")]
        assert len(native_members) == 1
        assert repaired.read(native_members[0]) == extension.read_bytes()
    linkage = subprocess.check_output(["ldd", str(extension)], text=True)
    assert "not found" not in linkage and "/opt/rh/" not in linkage, linkage
    versions = subprocess.check_output(["readelf", "--version-info", str(extension)], text=True)
    (root / "dist/elf-linkage.txt").write_text(linkage)
    (root / "dist/elf-symbol-versions.txt").write_text(versions)
    report = {
        "passed": True,
        "glibc": os.confstr("CS_GNU_LIBC_VERSION"),
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "original_wheel_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "native_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "tags": tags,
        "added_members": added,
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "scope": "Repaired wheel imports against glibc 2.28 without GCC toolset LD_LIBRARY_PATH. No newly bundled runtime files. See auditwheel logs, fresh-installed tests and ELF linkage; not universal distro/kernel qualification.",
    }
    (root / "dist/manylinux-runtime.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
