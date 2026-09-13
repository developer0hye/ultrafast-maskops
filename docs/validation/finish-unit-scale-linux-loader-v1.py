"""Wait for the exact existing timing parent, then verify fresh caches and audit.

No timing input/cache is mutated while the watched process remains live.
Failed/incomplete timing reports stop the follow-up; they are never restarted.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/yonghye/ultrafast-vision-build")
SOURCE = Path("/home/yonghye/ultrafast-maskops-unit-scale-v1")
PYTHON = ROOT / "mask-unit-scale-linux-clean-v1/bin/python"
PID = 1482166


def stamp():
    try:
        # /proc/stat field 22; split after the final ')' in the comm field.
        return (Path("/proc") / str(PID) / "stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


started = stamp()
require(started is not None, "timing parent is not live; inspect reports manually")
command = (Path("/proc") / str(PID) / "cmdline").read_bytes().split(b"\0")
require(
    command[:2] == [b"bash", str(ROOT / "run-unit-scale-linux-loader-v1.sh").encode()], "unexpected watched process"
)
state_path = ROOT / "mask-unit-scale-linux-loader-followup-v1.json"
require(not state_path.exists(), "retain prior follow-up state")
identity = json.loads((ROOT / "mask-unit-scale-linux-loader-identity-v1.json").read_text())
require(
    all(sha(SOURCE / name) == value for name, value in identity["source_sha256"].items()), "source drift before wait"
)
verify = SOURCE / "bench/verify_coco_loader.py"
require(sha(verify) == "b17bbcbb736f867c8f066242304bd22e1f76cce95ace63b75f8adaec830e613d", "verification harness drift")
auditor = ROOT / "audit-unit-scale-loader-v1.py"
auditor_sha = sha(auditor)
state = {
    "complete": False,
    "waiting_for_pid": PID,
    "parent_start_ticks": started,
    "script_sha256": sha(Path(__file__)),
    "auditor_sha256": auditor_sha,
    "steps": [],
}


def save():
    state_path.write_text(json.dumps(state, indent=2) + "\n")


save()
print(f"Waiting for timing parent {PID}, start ticks {started}", flush=True)
try:
    while stamp() == started:
        time.sleep(10)
    require(
        all(sha(SOURCE / name) == value for name, value in identity["source_sha256"].items()),
        "source drift after timing",
    )
    require(sha(auditor) == auditor_sha, "auditor changed during wait")
    reports = {}
    for mode in ("yes", "no"):
        path = ROOT / f"mask-unit-scale-linux-loader-overlap-{mode}-v1.json"
        report = json.loads(path.read_text())
        require(
            len(report["results"]) == 30 and set(report["summary"]) == {"0", "2", "8"},
            "timing incomplete; no cache mutation permitted",
        )
        require(
            report["source_sha256"] == identity["source_sha256"]
            and report["extension_sha256"] == identity["extension_sha256"],
            "measured identity mismatch",
        )
        require(
            all(
                row["overlap"] is (mode == "yes")
                and row["images"] == row["verification_images"] == 5000
                and row["parity"] is True
                for row in report["results"]
            ),
            "timing verification incomplete",
        )
        require(
            all(
                (path.with_suffix(".runs") / f"{row['workers']}-{row['round']}-{row['backend']}.phase").read_bytes()
                == b"done"
                for row in report["results"]
            ),
            "unfinished timed worker",
        )
        reports[mode] = path
    environment = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", YOLO_OFFLINE="true"
    )
    for mode, report in reports.items():
        fresh = ROOT / f"mask-unit-scale-linux-loader-fresh-{mode}-v1.json"
        audit = ROOT / f"mask-unit-scale-linux-loader-complete-{mode}-audit-v1.json"
        require(not fresh.exists() and not audit.exists(), "retain existing verification results")
        commands = [
            [
                str(PYTHON),
                str(verify),
                "--corpus",
                str(ROOT / "coco-val2017-yolo/segment"),
                "--benchmark",
                str(report),
                "--out",
                str(fresh),
            ],
            [
                sys.executable,
                str(auditor),
                "--report",
                str(report),
                "--identity",
                str(ROOT / "mask-unit-scale-linux-loader-identity-v1.json"),
                "--runs-dir",
                str(report.with_suffix(".runs")),
                "--overlap",
                mode,
                "--fresh",
                str(fresh),
                "--out",
                str(audit),
            ],
        ]
        for index, cmd in enumerate(commands):
            log = ROOT / f"mask-unit-scale-linux-loader-followup-{mode}-{index}-v1.log"
            print(f"Running follow-up overlap={mode} step={index}", flush=True)
            with log.open("x") as stream:
                subprocess.run(cmd, cwd=SOURCE, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
            state["steps"].append({"command": cmd, "log": str(log), "log_sha256": sha(log)})
            save()
    state["complete"] = True
    save()
    print("Both fresh references and complete audits finished", flush=True)
except BaseException as error:
    state["error"] = f"{type(error).__name__}: {error}"
    save()
    raise
