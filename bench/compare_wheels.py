"""Counterbalanced installed-wheel comparison on the frozen nine-case suite.

Each version/case/operation/repetition runs in a fresh process. Separate oracle
processes compute expected outputs; measured processes do not execute reference
mask calls. This measures public call cost and whole-process RSS, not training
throughput or traced working allocation. Run alone on the measurement host.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from zipfile import ZipFile

OPERATIONS = ("overlap", "bounded", "masks")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hashes(output):
    return [{"shape": list(a.shape), "dtype": str(a.dtype), "sha256": sha(a.tobytes())} for a in output]


def descriptor(root, wheel):
    import cv2
    import numpy as np
    import psutil
    import ultrafast_maskops as native

    profile = native._native.build_profile()
    for key, name in (
        ("bindings_sha256", "src/bindings.cpp"),
        ("cmake_sha256", "CMakeLists.txt"),
        ("template_sha256", "src/build_profile.h.in"),
    ):
        require(profile[key] == sha((root / name).read_bytes()), f"compiled source mismatch: {name}")
    extension = Path(native._native.__file__)
    wrapper = Path(native.__file__)
    require(wrapper.resolve().is_relative_to(Path(sys.prefix).resolve()), "requires an installed wheel")
    with ZipFile(wheel) as archive:
        require(
            archive.read("ultrafast_maskops/" + extension.name) == extension.read_bytes(), "extension/wheel mismatch"
        )
        require(archive.read("ultrafast_maskops/__init__.py") == wrapper.read_bytes(), "wrapper/wheel mismatch")
    require(
        wrapper.read_bytes() == (root / "python/ultrafast_maskops/__init__.py").read_bytes(), "wrapper/source mismatch"
    )
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "cv2_build": cv2.getBuildInformation(),
        "private_opencv_build": native._native.opencv_build_info(),
        "profile": profile,
        "extension_sha256": sha(extension.read_bytes()),
        "wheel_sha256": sha(wheel.read_bytes()),
        "wrapper_sha256": sha(wrapper.read_bytes()),
        "oracle_sha256": sha((Path(__file__).parents[1] / "tests/reference.py").read_bytes()),
        "fixture_code_sha256": sha(Path(__file__).with_name("feasibility.py").read_bytes()),
        "harness_sha256": sha(Path(__file__).read_bytes()),
        "private_opencv_threads": native._native.opencv_threads(),
        "ram_bytes": psutil.virtual_memory().total,
        "logical_cpus": psutil.cpu_count(),
    }


def worker(args):
    import cv2
    import psutil
    from feasibility import CASES, fixture, native, reference

    cv2.setNumThreads(0)
    info = descriptor(args.root, args.wheel)
    require(info["private_opencv_threads"] == 1, "private OpenCV must use one thread")
    if args.worker == "describe":
        return {"descriptor": info}
    if args.worker == "oracle":
        expected = {}
        for index, case in enumerate(CASES):
            segments = fixture(case)
            shape, ratio = (case["h"], case["w"]), case["ratio"]
            overlap = hashes(reference.polygons2masks_overlap(shape, segments, ratio))
            expected[str(index)] = {
                "input_sha256": sha(segments.tobytes()),
                "overlap": overlap,
                "bounded": overlap,
                "masks": hashes([reference.polygons2masks(shape, segments, 1, ratio)]),
            }
        return {"descriptor": info, "cases": CASES, "expected": expected}
    case = CASES[args.case]
    segments = fixture(case)
    expected = json.loads(args.oracle.read_text())["expected"][str(args.case)]
    require(sha(segments.tobytes()) == expected["input_sha256"], "fixture changed")
    shape, ratio = (case["h"], case["w"]), case["ratio"]
    if args.operation == "overlap":
        call = lambda: native.polygons2masks_overlap(shape, segments, ratio)
    elif args.operation == "masks":
        call = lambda: [native.polygons2masks(shape, segments, 1, ratio)]
    else:
        call = lambda: native.Rasterizer().overlap(
            shape, native.PackedPolygons.from_segments(segments), ratio, mode="bounded"
        )
    require(hashes(call()) == expected[args.operation], "pre-measurement output mismatch")
    for _ in range(10):
        call()
    memory_before = psutil.Process().memory_info().rss
    available_before = psutil.virtual_memory().available
    load_before = os.getloadavg()
    samples = []
    for _ in range(args.samples):
        start = time.perf_counter_ns()
        call()
        samples.append(time.perf_counter_ns() - start)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak = (
        int(Path("/proc/self/status").read_text().split("VmHWM:")[1].split()[0]) * 1024
        if sys.platform.startswith("linux")
        else usage.ru_maxrss
    )
    require(hashes(call()) == expected[args.operation], "post-measurement output mismatch")
    require(all(v > 0 for v in samples), "invalid clock samples")
    return {
        "descriptor": info,
        "case": args.case,
        "operation": args.operation,
        "samples_ns": samples,
        "median_ns": statistics.median(samples),
        "input_sha256": expected["input_sha256"],
        "output_hashes": expected[args.operation],
        "peak_rss_bytes": peak,
        "rss_after_warmup_bytes": memory_before,
        "available_ram_before_bytes": available_before,
        "available_ram_after_bytes": psutil.virtual_memory().available,
        "load_before": load_before,
        "load_after": os.getloadavg(),
    }


def summary(pairs):
    ratios = []
    for selection in itertools.product(range(len(pairs)), repeat=len(pairs)):
        chosen = [pairs[i] for i in selection]
        ratios.append(statistics.median(v[0] for v in chosen) / statistics.median(v[1] for v in chosen))
    ratios.sort()

    def percentile(p):
        position = p * (len(ratios) - 1)
        lower = int(position)
        return ratios[lower] + (ratios[min(lower + 1, len(ratios) - 1)] - ratios[lower]) * (position - lower)

    baseline, candidate = (statistics.median(v[i] for v in pairs) for i in (0, 1))
    return {
        "baseline_median": baseline,
        "candidate_median": candidate,
        "baseline_over_candidate": baseline / candidate,
        "paired_bootstrap_ci95": [percentile(0.025), percentile(0.975)],
        "ordered_resamples": len(ratios),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-python", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--baseline-wheel", type=Path)
    parser.add_argument("--candidate-python", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--candidate-wheel", type=Path)
    parser.add_argument("--worker", choices=("describe", "oracle", "measure"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--case", type=int)
    parser.add_argument("--operation", choices=OPERATIONS)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists() and args.samples >= 10, "use new output paths and at least ten samples")
    if args.worker:
        with args.out.open("x") as stream:
            stream.write(json.dumps(worker(args), indent=2) + "\n")
        return
    paths = {
        label: [getattr(args, f"{label}_{field}") for field in ("python", "root", "wheel")]
        for label in ("baseline", "candidate")
    }
    require(
        all(p is not None and p.exists() for values in paths.values() for p in values),
        "both installed-wheel environments required",
    )
    runs = args.out.with_suffix(".runs")
    runs.mkdir(parents=True, exist_ok=False)
    report = {
        "complete": False,
        "repeats": 5,
        "samples": args.samples,
        "operations": OPERATIONS,
        "harness_sha256": sha(Path(__file__).read_bytes()),
        "records": [],
        "oracles": {},
        "scope": "same public calls including packing, new Rasterizer and output allocation; nine frozen synthetic cases",
        "memory_scope": "fresh-process peak RSS includes imports, descriptor verification, inputs and warmups; oracle output computation runs separately; not working allocation",
        "statistics": "ratio of medians, paired percentile bootstrap enumerating all 3125 ordered five-pair resamples",
    }

    def save():
        args.out.write_text(json.dumps(report, indent=2) + "\n")

    def run(label, name, extra):
        python, root, wheel = paths[label]
        output = runs / f"{name}.json"
        command = [
            str(python),
            __file__,
            "--root",
            str(root),
            "--wheel",
            str(wheel),
            "--samples",
            str(args.samples),
            "--out",
            str(output),
            *extra,
        ]
        with (runs / f"{name}.log").open("x") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        data = output.read_bytes()
        return {"path": str(output), "sha256": sha(data), "command": command, "report": json.loads(data)}

    save()
    try:
        for label in paths:
            report["oracles"][label] = run(label, f"oracle-{label}", ["--worker", "oracle"])
        base, candidate = [report["oracles"][label]["report"] for label in paths]
        require(
            base["cases"] == candidate["cases"] and base["expected"] == candidate["expected"], "oracle/input mismatch"
        )
        common = (
            "python",
            "numpy",
            "opencv",
            "cv2_build",
            "wrapper_sha256",
            "oracle_sha256",
            "fixture_code_sha256",
            "harness_sha256",
            "private_opencv_threads",
            "ram_bytes",
            "logical_cpus",
        )
        require(
            all(base["descriptor"][k] == candidate["descriptor"][k] for k in common), "comparison environment differs"
        )
        for key in ("cmake_sha256", "template_sha256", "compiler", "build_type"):
            require(
                base["descriptor"]["profile"][key] == candidate["descriptor"]["profile"][key],
                "comparison build configuration differs",
            )
        require(base["descriptor"]["wheel_sha256"] != candidate["descriptor"]["wheel_sha256"], "identical wheels")
        report["cases"] = base["cases"]
        save()
        for repeat in range(5):
            for index in range(len(base["cases"])):
                for operation in OPERATIONS:
                    for label in tuple(paths) if repeat % 2 == 0 else tuple(paths)[::-1]:
                        record = run(
                            label,
                            f"r{repeat}-c{index}-{operation}-{label}",
                            [
                                "--worker",
                                "measure",
                                "--oracle",
                                report["oracles"][label]["path"],
                                "--case",
                                str(index),
                                "--operation",
                                operation,
                            ],
                        )
                        require(
                            record["report"]["descriptor"] == report["oracles"][label]["report"]["descriptor"],
                            "worker identity drift",
                        )
                        # The immutable child retains its full descriptor. Avoid
                        # repeatedly writing build-info strings in the parent.
                        record["report"].pop("descriptor")
                        record.update(repeat=repeat, version=label, case=index, operation=operation)
                        report["records"].append(record)
                        save()
                    print(f"paired repeat={repeat} case={index} operation={operation}", flush=True)
        report["summary"] = []
        rows = {(r["repeat"], r["case"], r["operation"], r["version"]): r["report"] for r in report["records"]}
        require(len(rows) == 5 * len(base["cases"]) * len(OPERATIONS) * 2, "incomplete comparison")
        for index in range(len(base["cases"])):
            for operation in OPERATIONS:
                value = {"case": index, "operation": operation}
                for metric in ("median_ns", "peak_rss_bytes"):
                    paired = [
                        (
                            rows[(r, index, operation, "baseline")][metric],
                            rows[(r, index, operation, "candidate")][metric],
                        )
                        for r in range(5)
                    ]
                    require(all(math.isfinite(v) and v > 0 for pair in paired for v in pair), "bad aggregate input")
                    value[metric] = summary(paired)
                report["summary"].append(value)
        report["complete"] = True
        save()
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
