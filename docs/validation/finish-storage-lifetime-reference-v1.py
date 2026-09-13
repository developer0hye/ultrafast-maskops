"""Preserve an already completed diagnostic whose interpreter remains alive."""
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path("/home/yonghye/ultrafast-vision-build")
PREFIX = ROOT / "mask-storage-lifetime-linux-v1-reference-first-batch"
REPORT = PREFIX.with_suffix(".json")
OUT = Path(str(PREFIX) + "-exit-observation.json")
assert not OUT.exists()
data = json.loads(REPORT.read_text())
assert data["pid"] == 1647975 and data["complete"] and data["protocol_passed"]

def snapshot(pid):
    root = Path(f"/proc/{pid}")
    try:
        raw_stat = (root / "stat").read_text()
        state = raw_stat.rsplit(")", 1)[1].split()
        fds = {}
        for fd in (root / "fd").iterdir():
            try:
                fds[fd.name] = {"target": os.readlink(fd), "info": (root / "fdinfo" / fd.name).read_text()}
            except FileNotFoundError:
                pass
        return {"pid": pid, "stat": raw_stat, "start_ticks": state[19], "state": state[0], "ppid": int(state[1]),
                "command": (root / "cmdline").read_bytes().replace(b"\0", b" ").decode(),
                "wchan": (root / "wchan").read_text(), "fds": fds}
    except FileNotFoundError:
        return {"pid": pid, "missing": True}

parent = snapshot(data["pid"])
assert str(ROOT / "mask-storage-lifetime-linux-v1.py") in parent["command"]
assert "--backend reference --reset-point first-batch" in parent["command"]
assert parent["state"] != "Z"
related = [parent]
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit() or int(entry.name) == data["pid"]:
        continue
    try:
        command = (entry / "cmdline").read_bytes()
        if b"torch_shm_manager" not in command and b"multiprocessing.resource_tracker" not in command:
            continue
        if os.readlink(entry / "fd/2") != str(PREFIX.with_suffix(".log")):
            continue
        related.append(snapshot(int(entry.name)))
    except (FileNotFoundError, PermissionError):
        continue

libc = ctypes.CDLL(None, use_errno=True)
libc.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
libc.shm_open.restype = ctypes.c_int
names = {r["handle"] for r in data["handles_after_close_plus_3s"]["present"]}
def probe():
    present = []
    for name in sorted(names):
        fd = libc.shm_open(name.encode(), os.O_RDONLY, 0)
        if fd >= 0:
            os.close(fd)
            present.append(name)
        else:
            assert ctypes.get_errno() == errno.ENOENT
    return present

receipt = {"report_sha256": hashlib.sha256(REPORT.read_bytes()).hexdigest(),
           "report_completed_at_ns": data["finished_at_ns"], "observed_at_ns": time.time_ns(),
           "before": related, "present_before": probe(), "natural_exit_passed": False,
           "reason": "Diagnostic body completed; original parent still live waiting after completion. SIGTERM only the verified diagnostic parent; no manager/tracker signals or shm unlink.",
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
OUT.write_text(json.dumps(receipt, indent=2) + "\n")
current = snapshot(data["pid"])
assert current["start_ticks"] == parent["start_ticks"] and current["command"] == parent["command"]
os.kill(data["pid"], signal.SIGTERM)
receipt["signal_sent"] = {"pid": data["pid"], "signal": "SIGTERM", "at_ns": time.time_ns()}
for _ in range(150):
    after = [snapshot(item["pid"]) for item in related]
    if all(item.get("missing") or item.get("state") == "Z" for item in after):
        break
    time.sleep(0.1)
receipt.update(after=after, present_after=probe(), finished_at_ns=time.time_ns())
receipt["all_related_terminal"] = all(item.get("missing") or item.get("state") == "Z" for item in after)
OUT.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"receipt": str(OUT), "related_count": len(related), "present_before": len(receipt["present_before"]), "present_after": len(receipt["present_after"]), "all_related_terminal": receipt["all_related_terminal"]}))
