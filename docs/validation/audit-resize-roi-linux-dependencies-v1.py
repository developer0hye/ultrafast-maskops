"""Read the completed Linux build graph; do not compile or import the extension."""

import gzip
import hashlib
import json
import subprocess
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


root = Path("/home/yonghye/ultrafast-vision-build")
source = Path("/home/yonghye/ultrafast-maskops-resize-roi-v1")
build = root / "mask-resize-roi-linux-build-v1"
manifest_bytes = (source / "docs/validation/native-notice-collection-v1.json").read_bytes()
inventory = json.loads(manifest_bytes)["source_files"]
bundled_manifest = json.loads((source / "licenses/manifest.json").read_text())
assert bundled_manifest["provenance_sha256"] == sha(manifest_bytes)
for name, digest in bundled_manifest["files_sha256"].items():
    assert sha((source / "licenses" / name).read_bytes()) == digest
graph = subprocess.check_output(
    [str(root / "venv312-cuda/bin/ninja"), "-C", str(build), "-t", "deps"]
)
selected, omitted = {}, set()
for line in graph.decode().splitlines():
    if not line.startswith("    "):
        continue
    path = line.strip()
    for marker, prefix in (
        ("opencv-src/", "opencv-4.13.0/"),
        ("kleidicv-0.7.0/", "kleidicv-0.7.0/"),
        ("site-packages/pybind11/include/", "pybind11-3.0.1/include/"),
    ):
        if marker in path:
            key = prefix + path.split(marker, 1)[1]
            if key not in selected:
                data = Path(path).read_bytes()
                selected[key] = {"sha256": sha(data), "bytes": len(data)}
            break
    else:
        omitted.add(path)
assert selected, "empty dependency graph cannot prove coverage"
missing = sorted(set(selected) - set(inventory))
changed = sorted(key for key in selected if key in inventory and selected[key] != inventory[key])
report = {
    "scope": "Completed ROI Linux build: selected OpenCV and pybind11 source/header bytes covered by bundled notice inventory. Excludes toolchain, system, Python and generated headers; not a complete redistribution audit.",
    "build_directory": str(build),
    "graph_sha256": sha(graph),
    "cmake_cache_sha256": sha((build / "CMakeCache.txt").read_bytes()),
    "inventory_sha256": sha(manifest_bytes),
    "compiled_source_sha256": sha((source / "src/bindings.cpp").read_bytes()),
    "build_extension_sha256": sha((build / "_native.cpython-312-x86_64-linux-gnu.so").read_bytes()),
    "selected_count": len(selected),
    "selected_files": dict(sorted(selected.items())),
    "omitted_files": sorted(omitted),
    "missing_files": missing,
    "changed_files": changed,
    "all_selected_files_covered": not missing and not changed,
}
out = root / "mask-resize-roi-linux-dependency-audit-v1.json"
with out.open("x") as stream:
    stream.write(json.dumps(report, indent=2) + "\n")
with (root / "mask-resize-roi-linux-compiled-deps-v1.txt.gz").open("xb") as stream:
    stream.write(gzip.compress(graph, mtime=0))
print(json.dumps({key: report[key] for key in ("selected_count", "missing_files", "changed_files", "all_selected_files_covered")}))
assert not missing and not changed
