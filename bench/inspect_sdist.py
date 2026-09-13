"""Check source payload identity and unpack a project-source build outside Git."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import traceback
from pathlib import Path, PurePosixPath


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inspect(checkout, archive, destination, report):
    checkout, archive, destination = (p.resolve() for p in (checkout, archive, destination))
    assert not destination.exists() and not destination.is_relative_to(checkout)
    required = {"CMakeLists.txt", "pyproject.toml", "LICENSE", "README.md"}
    for folder, pattern in (
        ("src", "*"),
        ("licenses", "*"),
        ("python", "*"),
        ("tests", "*.py"),
        ("bench", "*.py"),
        ("docs/validation", "*.py"),
    ):
        required.update(
            p.relative_to(checkout).as_posix()
            for p in (checkout / folder).rglob(pattern)
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")
        )
    with tarfile.open(archive) as source:
        members = source.getmembers()
        names = [m.name for m in members]
        assert len(names) == len(set(names)), "duplicate source member"
        prefixes = {PurePosixPath(n).parts[0] for n in names}
        assert len(prefixes) == 1
        prefix = prefixes.pop()
        assert re.fullmatch(r"ultrafast_maskops-[0-9a-z.+]+", prefix)
        for member in members:
            path = PurePosixPath(member.name)
            assert not path.is_absolute() and ".." not in path.parts and (member.isfile() or member.isdir())
        inventory = {m.name[len(prefix) + 1 :]: m for m in members if m.isfile()}
        assert required <= inventory.keys(), sorted(required - inventory.keys())
        excluded = [n for n in inventory if n.startswith("bench/results/") and (".tar.gz" in n or n.endswith(".zip"))]
        assert not excluded, excluded
        manifest = {}
        for name in sorted(required):
            data = source.extractfile(inventory[name]).read()
            assert data == (checkout / name).read_bytes(), "source payload differs: " + name
            manifest[name] = {"bytes": len(data), "sha256": sha(data)}
        destination.mkdir()
        source.extractall(destination, filter="data")
    extracted = destination / prefix
    assert not any(p.name == ".git" for p in extracted.rglob(".git"))
    (extracted / "dist").mkdir()
    shutil.copyfile(archive, extracted / "dist" / archive.name)
    result = {
        "passed": True,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip(),
        "sdist_bytes": archive.stat().st_size,
        "sdist_sha256": sha(archive.read_bytes()),
        "regular_files": len(inventory),
        "source_root": str(extracted),
        "required_files": manifest,
        "scope": "All runtime/build/license/test/harness source bytes match the checkout; benchmark archives excluded. Rebuild/install/tests must pass separately.",
    }
    with report.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write(f"source={extracted}\n")
    print(f"Verified {len(manifest)} required source files; sdist {archive.stat().st_size} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "archive", "destination", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        inspect(args.checkout, args.archive, args.destination, args.report)
    except BaseException:
        if not args.report.exists():
            with args.report.open("x") as stream:
                json.dump({"passed": False, "error": traceback.format_exc()}, stream, indent=2)
                stream.write("\n")
        raise
