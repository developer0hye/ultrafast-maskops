"""Retain the exact failed Windows wheel and its original artifact receipts."""

import json
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile

from inspect_hosted_artifact import api, sha


def main():
    repository = "developer0hye/ultrafast-maskops"
    artifact_id = 10314196359
    run_id = 34744592119
    source = "06a26d106ba8dd896061f987a6561521de68e6e1"
    out = Path("reset-evidence/wheel-input")
    out.mkdir(parents=True)
    meta = api(f"repos/{repository}/actions/artifacts/{artifact_id}")
    assert not meta["expired"] and meta["workflow_run"]["id"] == run_id
    assert meta["workflow_run"]["head_sha"] == source
    assert meta["digest"] == "sha256:f48a32b5623243698d364100f10f31804f72261121453f8e277063422fb0cf8f"
    (out / "original-artifact-api.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with tempfile.TemporaryDirectory() as folder:
        archive = Path(folder) / "artifact.zip"
        with archive.open("wb") as stream:
            subprocess.run(
                ["gh", "api", f"repos/{repository}/actions/artifacts/{artifact_id}/zip"], stdout=stream, check=True
            )
        assert "sha256:" + sha(archive) == meta["digest"]
        with ZipFile(archive) as zipped:
            wheel_name = "ultrafast_maskops-0.1.0a1-cp313-cp313-win_amd64.whl"
            assert [n for n in zipped.namelist() if n.endswith(".whl")] == [wheel_name]
            for name in (
                wheel_name,
                "core-tests.xml",
                "integration-tests.xml",
                "wheel-validation.json",
                "final-environment.txt",
            ):
                (out / name).write_bytes(zipped.read(name))
        binding = {
            "artifact_id": artifact_id,
            "run_id": run_id,
            "source_commit": source,
            "artifact_sha256": sha(archive),
            "wheel_sha256": sha(out / wheel_name),
        }
    original = json.loads((out / "wheel-validation.json").read_text())
    assert original["wheel_sha256"] == binding["wheel_sha256"]
    (out / "binding.json").write_text(json.dumps(binding, indent=2), encoding="utf-8")
    print(json.dumps(binding))


if __name__ == "__main__":
    main()
