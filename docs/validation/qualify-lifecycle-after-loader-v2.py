"""Wait for the exact full-loader controller, then qualify staged protocol code.

Never restarts the measurement. No GPU training is launched. The stage is an
editable formatting/test copy; original benchmark sources/runtimes stay frozen.
"""

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path("/home/yonghye/ultrafast-vision-build")
PREFIX = "mask-lifecycle-coordinator-linux-v2"
STAGE = ROOT / (PREFIX + "-stage")
OUTPUT = ROOT / (PREFIX + "-qualification.json")
FILES = ("bench/lifecycle_coco_gpu.py", "tests/test_lifecycle_coordinator.py", "pyproject.toml", ".github/workflows/wheels.yml")
STATE = {"complete": False, "passed": False, "controller_pid": os.getpid(), "commands": []}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save():
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(STATE, indent=2) + "\n")
    os.replace(temp, OUTPUT)


def run(label, command):
    log = ROOT / (PREFIX + "-" + label + ".log")
    record = {"label": label, "command": command, "cwd": str(STAGE), "started_at_ns": time.time_ns()}
    STATE["commands"].append(record)
    save()
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               YOLO_OFFLINE="true", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    with log.open("x") as stream:
        result = subprocess.run(command, cwd=STAGE, env=env, stdout=stream, stderr=subprocess.STDOUT)
    record.update(returncode=result.returncode, finished_at_ns=time.time_ns(), log_sha256=sha(log))
    save()
    print(label, "returncode", result.returncode, flush=True)
    if result.returncode:
        print(log.read_text()[-12000:], flush=True)
        raise RuntimeError(f"{label} failed; preserve the first attempt")


def main():
    assert not OUTPUT.exists(), "retain previous qualification"
    identity_path = ROOT / "mask-shared-loader-linux-full-v1-launch-identity.json"
    identity = json.loads(identity_path.read_text())
    assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == identity["host_boot_id"]
    hashes = {name: sha(STAGE / name) for name in FILES}
    STATE.update(phase="waiting-for-measurement", source_before_sha256=hashes,
                 script_sha256=sha(Path(__file__)), dependency_identity_sha256=sha(identity_path),
                 dependency_pid=identity["controller_pid"], dependency_start_ticks=identity["controller_start_ticks"],
                 started_at_ns=time.time_ns(), gpu_training_executed=False)
    save()
    print("Waiting for exact measurement PID", identity["controller_pid"], flush=True)
    while True:
        assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == identity["host_boot_id"]
        try:
            fields = Path(f"/proc/{identity['controller_pid']}/stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            break
        if fields[19] != identity["controller_start_ticks"] or fields[0] in ("Z", "X"):
            break
        time.sleep(15)
    dependency_path = ROOT / "mask-shared-loader-linux-full-v1-state.json"
    dependency = json.loads(dependency_path.read_text())
    assert dependency["complete"] and dependency["timing_and_fresh_passed"], "measurement/fresh dependency did not pass"
    assert [c["label"] for c in dependency["commands"]] == ["overlap-yes", "overlap-no", "fresh-yes", "fresh-no"]
    assert all(c["returncode"] == 0 for c in dependency["commands"])
    assert {name: sha(STAGE / name) for name in FILES} == hashes, "stage changed during the measurement wait"
    STATE.update(phase="qualifying", dependency_state_sha256=sha(dependency_path), qualification_started_at_ns=time.time_ns())
    save()
    python_files = list(FILES[:2])
    trees = {name: ast.dump(ast.parse((STAGE / name).read_text()), include_attributes=False) for name in python_files}
    ruff = ["/home/yonghye/.local/bin/uvx", "ruff==0.15.6"]
    run("format", ruff + ["format", *python_files])
    assert all(ast.dump(ast.parse((STAGE / name).read_text()), include_attributes=False) == tree for name, tree in trees.items())
    STATE["formatting_preserved_ast"] = True
    STATE["source_after_sha256"] = {name: sha(STAGE / name) for name in FILES}
    save()
    run("lint", ruff + ["check", *python_files])
    run("format-check", ruff + ["format", "--check", *python_files])
    run("workflow", [str(ROOT / "mask-persistent-linux-v1-tools/actionlint"), "-oneline", FILES[3]])
    xml = ROOT / (PREFIX + "-tests.xml")
    run("tests", [str(ROOT / "mask-shared-linux-v2-clean/bin/python"), "-m", "pytest", "-q", FILES[1], "--junitxml=" + str(xml)])
    cases = ET.parse(xml).findall(".//testcase")
    assert len(cases) == 63 and all(c.find(tag) is None for c in cases for tag in ("failure", "error", "skipped"))
    assert {name: sha(STAGE / name) for name in FILES} == STATE["source_after_sha256"]
    STATE.update(complete=True, passed=True, phase="qualified-synthetic-protocol", finished_at_ns=time.time_ns(),
                 test_cases=len(cases), tests_xml_sha256=sha(xml),
                 scope="Synthetic coordinator plan, validation, fake child-process execution and failure retention; not GPU training or full runtime qualification.")
    save()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        STATE.update(complete=True, passed=False, error=traceback.format_exc(), finished_at_ns=time.time_ns())
        save()
        raise
