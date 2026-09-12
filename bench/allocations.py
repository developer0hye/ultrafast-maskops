"""Separate traced-allocation run; never use these samples for latency claims."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import memray
from feasibility import CASES, digest, fixture, native, reference


def worker(index, backend, destination):
    cv2.setNumThreads(0)
    case = CASES[index]
    segments = fixture(case)
    shape, ratio = (case["h"], case["w"]), case["ratio"]
    if backend == "reference":
        fn = lambda: reference.polygons2masks_overlap(shape, segments, ratio)
    elif backend == "wrapper":
        fn = lambda: native.polygons2masks_overlap(shape, segments, ratio)
    else:
        fn = lambda: native.Rasterizer().overlap(
            shape, native.PackedPolygons.from_segments(segments), ratio, mode="bounded"
        )
    # Warm each selected implementation before enabling allocation tracking.
    for _ in range(10):
        fn()
    with memray.Tracker(destination, native_traces=True, trace_python_allocators=True):
        result = fn()
    # Oracle allocations occur after tracking, never inside the measurement.
    assert digest(result) == digest(reference.polygons2masks_overlap(shape, segments, ratio))
    reader = memray.FileReader(destination)
    peak = sum(record.size for record in reader.get_high_watermark_allocation_records())
    assert peak == reader.metadata.peak_memory
    return {
        "case": index,
        "backend": backend,
        "peak_tracked_allocation_bytes": peak,
        "output_hashes": digest(result),
        "parity": True,
        "trace": str(destination),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--backend")
    args = parser.parse_args()
    if args.worker is not None:
        print(json.dumps(worker(args.worker, args.backend, args.out)))
        return
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "memray": memray.__version__,
        "cases": CASES,
        "results": [],
        "scope": "peak tracked Python and native heap allocations during one warmed wrapper call; existing input excluded; output included",
        "timings": "instrumented runs are excluded from latency results",
    }
    for round_ in range(args.rounds):
        for index in (4, 5, 6, 7, 8):
            backends = ["reference", "wrapper", "bounded"]
            shift = round_ % 3
            for backend in backends[shift:] + backends[:shift]:
                destination = args.out / f"{round_}-{index}-{backend}.bin"
                result = json.loads(
                    subprocess.check_output(
                        [
                            sys.executable,
                            __file__,
                            "--worker",
                            str(index),
                            "--backend",
                            backend,
                            "--out",
                            str(destination),
                        ],
                        text=True,
                    )
                )
                result["round"] = round_
                report["results"].append(result)
            (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            print(round_ + 1, index, flush=True)


if __name__ == "__main__":
    main()
