"""Inspect a retained GitHub wheel artifact on a runner, without executing it."""

import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZipFile


def api(endpoint):
    return json.loads(subprocess.check_output(["gh", "api", endpoint], text=True))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def inspect(repository, artifact_id, expected_run, out):
    assert repository == "developer0hye/ultrafast-maskops"
    assert artifact_id > 0 and expected_run > 0
    meta = api(f"repos/{repository}/actions/artifacts/{artifact_id}")
    assert meta["expired"] is False and meta["workflow_run"]["id"] == expected_run
    run = api(f"repos/{repository}/actions/runs/{expected_run}")
    assert meta["workflow_run"]["head_sha"] == run["head_sha"]
    out.mkdir()
    (out / "artifact-api.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out / "run-api.json").write_text(json.dumps(run, indent=2) + "\n")
    with tempfile.TemporaryDirectory() as folder:
        archive = Path(folder) / "artifact.zip"
        with archive.open("wb") as stream:
            subprocess.run(
                ["gh", "api", f"repos/{repository}/actions/artifacts/{artifact_id}/zip"], stdout=stream, check=True
            )
        assert "sha256:" + sha(archive) == meta["digest"]
        with ZipFile(archive) as zipped:
            inventory = [
                {"name": n.filename, "bytes": n.file_size, "compressed_bytes": n.compress_size}
                for n in zipped.infolist()
            ]
            sdists = [n for n in zipped.namelist() if n.endswith(".tar.gz")]
            wheels = [n for n in zipped.namelist() if n.endswith(".whl")]
            assert len(sdists) == len(wheels) == 1
            sdist = Path(folder) / "source.tar.gz"
            with zipped.open(sdists[0]) as source, sdist.open("wb") as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
            with tarfile.open(sdist) as source:
                members = [m for m in source if m.isfile()]
                prefixes = {PurePosixPath(m.name).parts[0] for m in members}
                assert len(prefixes) == 1
                prefix = prefixes.pop()
                pyproject = source.extractfile(prefix + "/pyproject.toml").read()
                remote = api(f"repos/{repository}/contents/pyproject.toml?ref={run['head_sha']}")
                assert base64.b64decode(remote["content"]) == pyproject
                (out / "source-pyproject.toml").write_bytes(pyproject)
                entries = [{"path": m.name[len(prefix) + 1 :], "bytes": m.size} for m in members]
                benchmark_archives = [
                    e
                    for e in entries
                    if e["path"].startswith("bench/results/") and (".tar.gz" in e["path"] or e["path"].endswith(".zip"))
                ]
            tests = {}
            for name in zipped.namelist():
                if name.endswith(".xml"):
                    data = zipped.read(name)
                    cases = ET.fromstring(data).findall(".//testcase")
                    tests[name] = dict(
                        cases=len(cases),
                        failures=sum(c.find("failure") is not None for c in cases),
                        errors=sum(c.find("error") is not None for c in cases),
                        skipped=[
                            dict(case=c.attrib, reason=c.find("skipped").attrib)
                            for c in cases
                            if c.find("skipped") is not None
                        ],
                    )
                    (out / Path(name).name).write_bytes(data)
                elif name.endswith(("wheel-validation.json", "standalone-cache-smoke.json", "final-environment.txt")):
                    (out / Path(name).name).write_bytes(zipped.read(name))
            result = dict(
                passed=True,
                artifact_id=artifact_id,
                run_id=expected_run,
                source_commit=run["head_sha"],
                artifact_zip_sha256=sha(archive),
                artifact_bytes=archive.stat().st_size,
                zip_inventory=inventory,
                sdist_sha256=sha(sdist),
                sdist_compressed_bytes=sdist.stat().st_size,
                sdist_regular_files=len(entries),
                sdist_uncompressed_bytes=sum(e["bytes"] for e in entries),
                largest_source_members=sorted(entries, key=lambda e: e["bytes"], reverse=True)[:30],
                benchmark_archives=benchmark_archives,
                benchmark_archive_bytes=sum(e["bytes"] for e in benchmark_archives),
                tests=tests,
                scope="Artifact ZIP digest and source pyproject match the recorded run. Contents were inspected, not executed. Original archive remains in its initial GitHub artifact and must be preserved before expiry; this small report is not a complete archive backup.",
            )
    (out / "inspection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("artifact_bytes", "sdist_compressed_bytes", "benchmark_archive_bytes", "tests")},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--artifact-id", type=int, required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inspect(args.repository, args.artifact_id, args.run_id, args.out)
