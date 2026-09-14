"""Fixed synthetic feasibility suite. No training-throughput claims.

Run from a source checkout: python bench/feasibility.py --out bench/out/m2.json
Each backend/case/round runs in a fresh process. Reference hashes are computed
only in the parent, so reference allocations cannot inflate native-worker RSS.
"""

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import reference
import ultrafast_maskops as native

CASES = [{"n": n, "h": 640, "w": 640, "ratio": 4, "vertices": 16} for n in (0, 1, 5, 20, 100, 255, 256, 500)] + [
    {"n": 100, "h": 640, "w": 640, "ratio": 1, "vertices": 16}
]


def fixture(case):
    rng = np.random.default_rng(20260912 + case["n"])
    centers = rng.uniform(0.1, 0.9, (case["n"], 1, 2))
    angles = np.linspace(0, 2 * np.pi, case["vertices"], endpoint=False)
    radii = rng.uniform(0.015, 0.2, (case["n"], 1, 1))
    return (
        (centers + radii * np.stack([np.cos(angles), np.sin(angles)], -1)) * np.array([case["w"], case["h"]])
    ).astype(np.float32)


def digest(result):
    return [hashlib.sha256(a.tobytes()).hexdigest() + f":{a.dtype}:{a.shape}" for a in result]


def environment():
    root = Path(__file__).resolve().parents[1]
    source_hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ("src", "python", "tests", "bench")
        for p in (root / folder).rglob("*")
        if p.suffix in (".py", ".cpp") and "out" not in p.parts
    }
    cpu = (
        subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if sys.platform == "darwin"
        else next(
            (
                s.split(":", 1)[1].strip()
                for s in Path("/proc/cpuinfo").read_text().splitlines()
                if s.startswith("model name")
            ),
            platform.processor(),
        )
    )
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "cpu": cpu,
        "logical_cpus": psutil.cpu_count(),
        "source_sha256": source_hashes,
        "extension_sha256": hashlib.sha256(Path(native._native.__file__).read_bytes()).hexdigest(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_bytes": psutil.virtual_memory().total,
        "available_ram_bytes": psutil.virtual_memory().available,
        "loadavg": os.getloadavg(),
        "numpy": np.__version__,
        "cv2": cv2.__version__,
        "native": native.backend_info(),
        "cv2_build": cv2.getBuildInformation(),
    }


def worker(index, backend, expected, samples):
    cv2.setNumThreads(0)
    case = CASES[index]
    segments = fixture(case)
    shape, r = (case["h"], case["w"]), case["ratio"]
    if backend == "reference":
        fn = lambda: reference.polygons2masks_overlap(shape, segments, r)
    elif backend == "wrapper":
        fn = lambda: native.polygons2masks_overlap(shape, segments, r)
    else:
        fn = lambda: native.Rasterizer().overlap(
            shape, native.PackedPolygons.from_segments(segments), r, mode="bounded"
        )
    assert digest(fn()) == expected
    for _ in range(10):
        fn()
    raw = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        fn()
        raw.append(time.perf_counter_ns() - start)
    assert digest(fn()) == expected
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux ru_maxrss may retain the parent's pre-exec high-water mark. VmHWM
    # belongs to the current exec image and avoids oracle-parent contamination.
    peak = (
        int(Path("/proc/self/status").read_text().split("VmHWM:")[1].split()[0]) * 1024
        if sys.platform.startswith("linux")
        else usage.ru_maxrss
    )
    return {
        "case": index,
        "backend": backend,
        "samples_ns": raw,
        "median_ns": float(np.median(raw)),
        "p95_ns": float(np.percentile(raw, 95)),
        "peak_rss_bytes": peak,
        "legacy_ru_maxrss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        "rss_method": "/proc/self/status VmHWM" if sys.platform.startswith("linux") else "getrusage ru_maxrss bytes",
        "user_s": usage.ru_utime,
        "system_s": usage.ru_stime,
        "input_sha256": hashlib.sha256(segments.tobytes()).hexdigest(),
        "output_hashes": expected,
        "parity": True,
        "available_ram_bytes": psutil.virtual_memory().available,
        "loadavg": os.getloadavg(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--worker", type=int)
    p.add_argument("--backend")
    p.add_argument("--expected")
    args = p.parse_args()
    if args.worker is not None:
        print(json.dumps(worker(args.worker, args.backend, json.loads(args.expected), args.samples)))
        return
    if not args.out:
        p.error("--out required")
    cv2.setNumThreads(0)
    report = {
        "environment": environment(),
        "cases": CASES,
        "rounds": args.rounds,
        "samples": args.samples,
        "scope": "synthetic overlap wrappers including packing/sorting, no inference or training",
        "memory_scope": "fresh-process whole-process peak RSS, including imports and inputs; not native allocation tracing",
        "os_cache": "uncontrolled",
        "warmups": 10,
        "results": [],
    }
    expected = [digest(reference.polygons2masks_overlap((c["h"], c["w"]), fixture(c), c["ratio"])) for c in CASES]
    for round_ in range(args.rounds):
        for index in range(len(CASES)):
            backends = ["reference", "wrapper", "bounded"]
            shift = round_ % 3
            backends = backends[shift:] + backends[:shift]
            for backend in backends:
                output = subprocess.check_output(
                    [
                        sys.executable,
                        __file__,
                        "--worker",
                        str(index),
                        "--backend",
                        backend,
                        "--samples",
                        str(args.samples),
                        "--expected",
                        json.dumps(expected[index]),
                    ],
                    text=True,
                )
                row = json.loads(output)
                row["round"] = round_
                report["results"].append(row)
            print(f"round {round_ + 1}/{args.rounds}, case {index + 1}/{len(CASES)}", flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
    summaries = []
    for index in range(len(CASES)):
        base = [r for r in report["results"] if r["case"] == index and r["backend"] == "reference"]
        for backend in ["wrapper", "bounded"]:
            rows = [r for r in report["results"] if r["case"] == index and r["backend"] == backend]
            ratios = np.array([b["median_ns"] / r["median_ns"] for b, r in zip(base, rows)])
            rng = np.random.default_rng(42)
            boot = np.median(rng.choice(ratios, (10000, len(ratios))), axis=1)
            summaries.append(
                {
                    "case": index,
                    "backend": backend,
                    "speedup_median": float(np.median(ratios)),
                    "speedup_paired_bootstrap_ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
                    "reference_median_ms": float(np.median([b["median_ns"] for b in base]) / 1e6),
                    "native_median_ms": float(np.median([r["median_ns"] for r in rows]) / 1e6),
                    "reference_peak_rss_bytes_median": float(np.median([b["peak_rss_bytes"] for b in base])),
                    "native_peak_rss_bytes_median": float(np.median([r["peak_rss_bytes"] for r in rows])),
                }
            )
    report["summary"] = summaries
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
