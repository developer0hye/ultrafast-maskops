"""Linux-only diagnostic supervisor: require process exit in addition to body completion.

Adds only a faulthandler signal handler to the child. Does not alter worker,
queue, sharing, or interpreter teardown. A timeout is retained as failure.
"""

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_snapshot(pid):
    root = Path(f"/proc/{pid}")
    try:
        raw = (root / "stat").read_text()
        fields = raw.rsplit(")", 1)[1].split()
        result = {
            "pid": pid,
            "stat": raw,
            "start_ticks": fields[19],
            "state": fields[0],
            "command": (root / "cmdline").read_bytes().replace(b"\0", b" ").decode(),
            "wchan": (root / "wchan").read_text(),
            "fds": {},
        }
        for fd in (root / "fd").iterdir():
            try:
                result["fds"][fd.name] = {"target": os.readlink(fd), "info": (root / "fdinfo" / fd.name).read_text()}
            except (FileNotFoundError, PermissionError) as exc:
                result["fds"][fd.name] = {"observation_error": repr(exc)}
        return result
    except FileNotFoundError:
        return {"pid": pid, "missing": True}
    except PermissionError as exc:
        return {"pid": pid, "observation_error": repr(exc)}


def related_snapshots(pid, log):
    result = [process_snapshot(pid)]
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == pid:
            continue
        try:
            cmd = (entry / "cmdline").read_bytes()
            if b"torch_shm_manager" not in cmd and b"multiprocessing.resource_tracker" not in cmd:
                continue
            if os.readlink(entry / "fd/2") == str(log):
                result.append(process_snapshot(int(entry.name)))
        except (FileNotFoundError, PermissionError):
            continue
    return result


def probe_report_names(report):
    library = ctypes.CDLL(None, use_errno=True)
    library.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
    library.shm_open.restype = ctypes.c_int
    names = set()
    for path in report.with_suffix(".traces").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            item = json.loads(line)
            assert item["handle"].startswith(f"/torch_{item['pid']}_") and item["handle"].count("/") == 1
            names.add(item["handle"])
    present = []
    for name in sorted(names):
        descriptor = library.shm_open(name.encode(), os.O_RDONLY, 0)
        if descriptor >= 0:
            os.close(descriptor)
            present.append(name)
        else:
            assert ctypes.get_errno() == errno.ENOENT
    return {"observed": len(names), "present": present, "at_ns": time.time_ns()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--backend", choices=("reference", "native"), required=True)
    parser.add_argument("--reset-point", choices=("setup", "first-batch", "full-epoch"), required=True)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = args.out.resolve()
    log = report.with_suffix(".log")
    receipt_path = report.with_suffix(".process.json")
    assert not report.exists() and not receipt_path.exists()
    # run_name must be __main__ so spawned workers can resolve Fixture normally.
    entry = "import faulthandler,runpy,signal,sys; faulthandler.register(signal.SIGUSR1,all_threads=True); p=sys.argv.pop(1); runpy.run_path(p,run_name='__main__')"
    command = [
        sys.executable,
        "-c",
        entry,
        str(args.script.resolve()),
        "--backend",
        args.backend,
        "--reset-point",
        args.reset_point,
        "--cycles",
        str(args.cycles),
        "--out",
        str(report),
    ]
    state = {
        "complete": False,
        "passed": False,
        "command": command,
        "started_at_ns": time.time_ns(),
        "supervisor_sha256": digest(__file__),
        "diagnostic_sha256": digest(args.script),
        "body_timeout_s": 240,
        "exit_grace_s": 20,
        "timeout": False,
    }

    def save():
        receipt_path.write_text(json.dumps(state, indent=2) + "\n")

    with log.open("x") as output:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        state["pid"] = child.pid
        save()
        start = time.monotonic()
        body_finished = None
        while child.poll() is None:
            if body_finished is None and report.exists():
                try:
                    if json.loads(report.read_text()).get("complete"):
                        body_finished = time.monotonic()
                        state["body_completion_observed_at_ns"] = time.time_ns()
                        save()
                except json.JSONDecodeError:
                    pass
            reason = None
            if body_finished is not None and time.monotonic() - body_finished > 20:
                reason = "process did not exit within 20s after diagnostic body completion"
            elif time.monotonic() - start > 240:
                reason = "diagnostic body exceeded 240s"
            if reason:
                state.update(timeout=True, timeout_reason=reason, before_termination=related_snapshots(child.pid, log))
                save()
                if child.poll() is None:
                    child.send_signal(signal.SIGUSR1)
                    time.sleep(1)
                if child.poll() is None:
                    child.terminate()
                    state["termination_signal"] = "SIGTERM"
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    state["termination_signal"] = "SIGKILL"
                    child.wait(timeout=10)
                break
            time.sleep(0.1)
        state["returncode"] = child.wait()
    time.sleep(3)
    if state.get("before_termination"):
        state["after_termination"] = [process_snapshot(p["pid"]) for p in state["before_termination"]]
    if report.exists():
        data = json.loads(report.read_text())
        state["report_sha256"] = digest(report)
        state["body_protocol_passed"] = data.get("protocol_passed") is True
        state["handles_after_process_exit"] = probe_report_names(report)
    state.update(complete=True, finished_at_ns=time.time_ns(), log_sha256=digest(log))
    state["passed"] = state["returncode"] == 0 and not state["timeout"] and state.get("body_protocol_passed", False)
    save()
    print(
        json.dumps({k: state.get(k) for k in ("complete", "passed", "pid", "returncode", "timeout", "timeout_reason")})
    )
    return 0 if state["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
