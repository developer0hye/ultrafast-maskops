"""Isolate removed area reduction cost; this is not a training benchmark.

Five fresh processes compare the old prepacked masks path with the new one.
Each process counterbalances the two calls per sample, verifies both against
the frozen OpenCV reference, and retains all timings. Memory figures describe
the removed uint64 area array only, not RSS or total allocation savings.
"""

import argparse
import hashlib
import json
import operator
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import ultrafast_maskops as native

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import reference

CASES = [(640, 1, 4), (640, 100, 4), (640, 500, 4), (1280, 100, 4), (640, 100, 1), (640, 100, 16)]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def profile():
    root = Path(__file__).resolve().parents[1]
    build = native._native.build_profile()
    for key, name in (("bindings_sha256", "src/bindings.cpp"), ("cmake_sha256", "CMakeLists.txt")):
        assert build[key] == sha((root / name).read_bytes()), f"installed binary does not match {name}"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "source_sha256": {
            path: sha((root / path).read_bytes())
            for path in ("src/bindings.cpp", "CMakeLists.txt", "python/ultrafast_maskops/__init__.py", "bench/masks_only.py")
        },
        "extension_sha256": sha(Path(native._native.__file__).read_bytes()),
        "native": native.backend_info(),
    }


def worker(repeat, samples):
    cv2.setNumThreads(0)
    rng = np.random.default_rng(912)
    order = random.Random(912 + repeat)
    results = []
    for size, count, ratio in CASES:
        centers = rng.uniform(0.05, 0.95, (count, 1, 2)) * size
        angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        offsets = np.stack([np.cos(angles), np.sin(angles)], -1)
        polygons = (centers + rng.uniform(0.01, 0.2, (count, 1, 1)) * size * offsets).astype(np.float32)
        packed = native.PackedPolygons.from_segments(polygons)
        engine = native.Rasterizer()

        def legacy(engine=engine, packed=packed, size=size, ratio=ratio):
            h, w, r = native._dimensions((size, size), ratio)
            with engine._lock:
                return engine._core.raster(packed._native, h, w, r, operator.index(1), True)[0]

        def current(engine=engine, packed=packed, size=size, ratio=ratio):
            return engine.masks((size, size), packed, 1, ratio)

        calls = {"legacy": legacy, "masks_only": current}
        expected = reference.polygons2masks((size, size), polygons, 1, ratio)
        for fn in calls.values():
            got = fn()
            assert got.dtype == expected.dtype and got.shape == expected.shape
            assert got.tobytes() == expected.tobytes()
            for _ in range(5):
                fn()
        timings = {name: [] for name in calls}
        for _ in range(samples):
            names = list(calls)
            order.shuffle(names)
            for name in names:
                start = time.perf_counter_ns()
                calls[name]()
                timings[name].append(time.perf_counter_ns() - start)
        for fn in calls.values():
            assert fn().tobytes() == expected.tobytes()
        results.append(
            {
                "size": size,
                "instances": count,
                "ratio": ratio,
                "input_sha256": sha(polygons.tobytes()),
                "output_sha256": sha(expected.tobytes()),
                "removed_area_array_bytes": count * 8,
                "removed_reduced_mask_pixels_summed": count * (size // ratio) ** 2,
                "samples_ns": timings,
                "medians_ns": {name: statistics.median(values) for name, values in timings.items()},
            }
        )
    return {"repeat": repeat, "profile": profile(), "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    assert not args.out.exists() and args.samples >= 10
    if args.worker is not None:
        args.out.write_text(json.dumps(worker(args.worker, args.samples), indent=2) + "\n")
        return
    folder = args.out.with_suffix(".runs")
    folder.mkdir(parents=True, exist_ok=False)
    reports, raw = [], []
    for repeat in range(5):
        path = folder / f"r{repeat}.json"
        subprocess.run(
            [sys.executable, __file__, "--worker", str(repeat), "--samples", str(args.samples), "--out", str(path)],
            check=True,
        )
        result = json.loads(path.read_text())
        assert result["profile"] == profile()
        reports.append(result)
        raw.append({"path": str(path), "sha256": sha(path.read_bytes())})
    summary = []
    for index, case in enumerate(CASES):
        values = [r["results"][index] for r in reports]
        assert len({(v["input_sha256"], v["output_sha256"]) for v in values}) == 1
        legacy = statistics.median(v["medians_ns"]["legacy"] for v in values)
        current = statistics.median(v["medians_ns"]["masks_only"] for v in values)
        summary.append({"case": case, "legacy_ns": legacy, "masks_only_ns": current, "ratio": legacy / current})
    report = {
        "scope": "counterbalanced same-process prepacked mechanism comparison, five fresh processes; not full API, RSS or training evidence",
        "raw_reports": raw,
        "reports": reports,
        "summary": summary,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
