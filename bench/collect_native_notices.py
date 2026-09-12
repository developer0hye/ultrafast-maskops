"""Collect notices from the observed OpenCV/pybind11 compiler dependency graphs.

Conservatively includes notices from all compiled archive objects and headers,
even if the final linker discards some. This is not a legal-completeness proof.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*(?:\n[ \t]*//[^\n]*)*", re.DOTALL)
NOTICE = re.compile(r"copyright|license|redistribution|permission", re.IGNORECASE)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deps", type=Path, nargs="+", required=True)
    parser.add_argument("--opencv", type=Path, required=True)
    parser.add_argument("--kleidicv", type=Path, required=True)
    parser.add_argument("--pybind11", type=Path, required=True)
    parser.add_argument("--pybind11-license", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), "retain previous inventories"
    args.out.mkdir(parents=True)
    mappings = [
        ("opencv-src/", "opencv-4.13.0", args.opencv),
        ("kleidicv-0.7.0/", "kleidicv-0.7.0", args.kleidicv),
        ("site-packages/pybind11/include/", "pybind11-3.0.1/include", args.pybind11 / "include"),
    ]
    files, graphs, notices = {}, [], {}
    for graph in args.deps:
        included, omitted = [], set()
        for line in graph.read_text().splitlines():
            if not line.startswith("    "):
                continue
            source = line.strip()
            for marker, prefix, root in mappings:
                if marker in source:
                    relative = source.split(marker, 1)[1]
                    name = f"{prefix}/{relative}"
                    path = root / relative
                    content = path.read_bytes()
                    entry = {"sha256": sha(content), "bytes": len(content)}
                    assert name not in files or files[name] == entry
                    files[name] = entry
                    included.append(name)
                    for match in COMMENT.finditer(content.decode("utf-8", errors="replace")):
                        block = match.group()
                        if NOTICE.search(block):
                            digest = sha(block.encode())
                            entry = notices.setdefault(digest, {"text": block, "sources": set()})
                            entry["sources"].add(name)
                    break
            else:
                omitted.add(source)
        graphs.append(
            {
                "file": graph.name,
                "sha256": sha(graph.read_bytes()),
                "selected_files": sorted(set(included)),
                "omitted_files": sorted(omitted),
            }
        )
    texts = {
        "OpenCV-Apache-2.0.txt": args.opencv / "LICENSE",
        "OpenCV-Legacy-BSD.txt": args.opencv / "doc/LICENSE_BSD.txt",
        "zlib-LICENSE.txt": args.opencv / "3rdparty/zlib/LICENSE",
        "DLPack-LICENSE.txt": args.opencv / "3rdparty/dlpack/LICENSE",
        "SoftFloat-COPYING.txt": args.opencv / "modules/core/3rdparty/SoftFloat/COPYING.txt",
        "KleidiCV-Apache-2.0.txt": args.kleidicv / "LICENSES/Apache-2.0.txt",
        "pybind11-LICENSE.txt": args.pybind11_license,
    }
    licenses = {}
    for name, path in texts.items():
        data = path.read_bytes()
        (args.out / name).write_bytes(data)
        licenses[name] = {"source": str(path), "sha256": sha(data)}
    output = [
        (
            "Collected source/header notices from the recorded M2 and Linux compiler graphs.\n"
            "Conservative collection, not a claim that every listed file contributes linked code.\n"
        )
    ]
    for digest, entry in sorted(notices.items()):
        output.extend(
            [
                "\n" + "=" * 72,
                f"Notice SHA-256: {digest}",
                "Observed source files:\n" + "\n".join(sorted(entry["sources"])),
                "",
                entry["text"],
            ]
        )
    source_notices = "\n".join(output) + "\n"
    (args.out / "COMPILED-SOURCE-NOTICES.txt").write_text(source_notices)
    manifest = {
        "scope": "observed M2/Linux OpenCV, KleidiCV and pybind11 source/header notice collection; excludes toolchain/system/Python headers and generated build files; Windows and full redistribution review remain open",
        "collector_sha256": sha(Path(__file__).read_bytes()),
        "graphs": graphs,
        "source_files": dict(sorted(files.items())),
        "unique_notice_blocks": len(notices),
        "license_files": licenses,
        "source_notices_sha256": sha(source_notices.encode()),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"files": len(files), "notice_blocks": len(notices), "license_files": len(texts)}, indent=2))


if __name__ == "__main__":
    main()
