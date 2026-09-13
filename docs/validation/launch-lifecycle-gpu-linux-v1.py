"""Launch the full frozen GPU lifecycle grid only from a qualified combined runtime."""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

ROOT = Path("/home/yonghye/ultrafast-vision-build")
PREFIX = "mask-lifecycle-gpu-linux-v1"
SOURCE = ROOT / (PREFIX + "-source")
PYTHON = ROOT / "dataset-notices-linux-v2-clean/bin/python"
CORPUS = ROOT / "coco-val2017-yolo/segment"
STATE_PATH = ROOT / (PREFIX + "-launch.json")
OUT = ROOT / (PREFIX + "-grid.json")
ENV = os.environ.copy()
for key in ("PYTHONPATH", "PYTHONHOME"):
    ENV.pop(key, None)
ENV.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", YOLO_OFFLINE="true")
STATE = {
    "complete": False,
    "passed": False,
    "launcher_pid": os.getpid(),
    "started_at_ns": time.time_ns(),
    "commands": [],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save():
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(STATE, indent=2) + "\n")
    os.replace(tmp, STATE_PATH)


def run(label, command):
    log = ROOT / (PREFIX + "-" + label + ".log")
    record = {
        "label": label,
        "command": list(map(str, command)),
        "cwd": str(SOURCE),
        "started_at_ns": time.time_ns(),
        "log": str(log),
    }
    STATE["commands"].append(record)
    save()
    with log.open("x") as stream:
        child = subprocess.Popen(
            record["command"], cwd=SOURCE, env=ENV, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        record["pid"] = child.pid
        save()
        try:
            code = child.wait()
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
            record.update(returncode=child.returncode, finished_at_ns=time.time_ns(), log_sha256=sha(log))
            save()
    assert code == 0, f"{label} failed; preserve {log}"
    return log.read_text()


def main():
    assert not OUT.exists() and not OUT.with_suffix(".runs").exists()
    qualification = ROOT / "dataset-notices-linux-v2-qualification.json"
    q = json.loads(qualification.read_text())
    assert q["complete"] and q["passed"] and q["standalone_passed"] and q["notice_files_checked"] == 137
    assert q["normalized_dependency_pins_match"] and q["installed_sources_unchanged"]
    for name in ("dataset-tests", "mask-tests"):
        assert q[name]["collected"] == q[name]["cases"] > 100
        assert not q[name]["failed"] and not q[name]["skipped"]
    assert all(c["returncode"] == 0 for c in q["commands"])
    source_path = ROOT / (PREFIX + "-source.json")
    source = json.loads(source_path.read_text())
    assert sha(ROOT / (PREFIX + "-source.tar")) == source["archive_sha256"]
    for name, expected in source["files"].items():
        assert sha(SOURCE / name) == expected, name
    benchmark = ROOT / "mask-shared-loader-linux-full-v1-overlap-no.json"
    assert sha(benchmark) == source["loader_reference_report_sha256"]
    benchmark_data = json.loads(benchmark.read_text())
    for name, expected in benchmark_data["source_sha256"].items():
        assert sha(SOURCE / name) == expected, name
    old_fresh = ROOT / "mask-shared-loader-linux-full-v1-fresh-no.json"
    fresh_bytes = old_fresh.read_bytes()
    fresh_data = json.loads(fresh_bytes)
    cache = CORPUS / "labels/val2017.cache"
    cache_bytes = cache.read_bytes()
    assert hashlib.sha256(cache_bytes).hexdigest() == fresh_data["fresh_cache_sha256"]
    assert fresh_data["benchmark_sha256"] == sha(benchmark)
    assert fresh_data["matched_runs"] == 30 and fresh_data["all_benchmark_outputs_match_fresh_reference"]
    assert fresh_data["fresh_reference"]["images"] == 5000
    fresh = ROOT / (PREFIX + "-fresh-reference.json")
    with fresh.open("xb") as stream:
        stream.write(fresh_bytes)
    preserved_cache = ROOT / (PREFIX + "-original-reference.cache")
    with preserved_cache.open("xb") as stream:
        stream.write(cache_bytes)
    del cache_bytes
    STATE.update(
        qualification_sha256=sha(qualification),
        source_manifest_sha256=sha(source_path),
        source_commit=source["source_commit"],
        controller_sha256=sha(__file__),
        fresh_reference_source=str(old_fresh),
        fresh_reference_sha256=sha(fresh),
        original_cache_sha256=sha(preserved_cache),
        original_cache_copy=str(preserved_cache),
        declared_trials=54,
        declared_epochs=126,
        worker_order=[2, 0, 8],
        overlaps=["yes", "no"],
        scope="Full lifecycle/parity grid, not repeated throughput or independent final artifact audit.",
    )
    save()
    # Run imports and the full corpus fingerprint in a short-lived process;
    # the launcher itself stays framework-free during measurements.
    preflight = """import hashlib, importlib.metadata as md, json, sys
from pathlib import Path
sys.path.insert(0, "bench")
from coco_loader import COCO_FINGERPRINT, fingerprint, validate_profile
import ultrafast_maskops as mask
import ultrafast_yolo_dataset as dataset
from ultrafast_yolo_dataset.ultralytics import check_profile
import torch
validate_profile(); check_profile()
assert fingerprint(Path("/home/yonghye/ultrafast-vision-build/coco-val2017-yolo/segment")) == COCO_FINGERPRINT
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.multiprocessing.get_sharing_strategy() == "file_descriptor"
for module in (mask, dataset):
    assert Path(module.__file__).is_relative_to(sys.prefix)
print(json.dumps({"python":sys.version,"torch":torch.__version__,"gpu":torch.cuda.get_device_name(0),"sharing_strategy":torch.multiprocessing.get_sharing_strategy(),"packages":{m.__name__:str(Path(m.__file__).resolve()) for m in (mask,dataset)},"extension_sha256":{m.__name__:hashlib.sha256(Path(m._native.__file__).read_bytes()).hexdigest() for m in (mask,dataset)},"fixture":COCO_FINGERPRINT},indent=2))
"""
    runtime = json.loads(run("runtime-preflight", [PYTHON, "-c", preflight]))
    # Match exact wheel payloads recorded by the combined installed-byte audits.
    for package, audit_name in (
        ("ultrafast_maskops", "combined-mask-audit"),
        ("ultrafast_yolo_dataset", "combined-dataset-audit"),
    ):
        audit_path = ROOT / ("dataset-notices-linux-v2-" + audit_name + ".json")
        audit = json.loads(audit_path.read_text())
        assert audit["clean"]
        (expected,) = [
            v["sha256"] for k, v in audit["files"].items() if k.startswith(package + "/_native.") and k.endswith(".so")
        ]
        assert runtime["extension_sha256"][package] == expected
    gpu_processes = run(
        "gpu-before", ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"]
    )
    assert not gpu_processes.strip(), "GPU compute process active; do not overlap workloads"
    STATE.update(runtime_preflight=runtime, preflight_passed=True)
    save()
    run(
        "coordinator",
        [
            PYTHON,
            SOURCE / "bench/lifecycle_coco_gpu.py",
            "--corpus",
            CORPUS,
            "--fresh-check",
            fresh,
            "--out",
            OUT,
            "--workers",
            "2",
            "0",
            "8",
            "--overlaps",
            "yes",
            "no",
        ],
    )
    result = json.loads(OUT.read_text())
    assert result["complete"] and result["full_protocol_requested"] and result["full_protocol_complete"]
    assert len(result["records"]) == 54 and all(r["validated"] and r["returncode"] == 0 for r in result["records"])
    STATE.update(
        complete=True,
        passed=True,
        grid_sha256=sha(OUT),
        finished_at_ns=time.time_ns(),
        independent_final_audit_passed=False,
    )
    save()


if __name__ == "__main__":
    assert not STATE_PATH.exists()
    try:
        main()
    except BaseException:
        STATE.update(complete=True, passed=False, error=traceback.format_exc(), finished_at_ns=time.time_ns())
        if OUT.exists():
            STATE["grid_sha256"] = sha(OUT)
        save()
        raise
