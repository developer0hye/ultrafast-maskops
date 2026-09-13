"""Overlap-mask stage on captured Format inputs: reference, full-resolution native, sampled native.

Inputs come from bench/capture_format_inputs.py-style captures: an .npz with
points, offsets, instances, vertices, hw and ratio per Format call. Every
backend's output is compared with the unmodified Ultralytics function first.
"""

import argparse
import json
import statistics
import time

import numpy as np
import ultrafast_maskops as um
from ultralytics.data.utils import polygons2masks_overlap


def load(path):
    data = np.load(path)
    points, offsets, counts, verts, hw, ratio = (data[k] for k in ("points", "offsets", "instances", "vertices", "hw", "ratio"))
    samples = []
    for i in range(len(counts)):
        n, v = int(counts[i]), int(verts[i])
        if n:  # Format only rasterizes images with instances
            seg = np.ascontiguousarray(points[offsets[i] : offsets[i + 1]].reshape(n, v, 2))
            samples.append((seg, (int(hw[i][0]), int(hw[i][1])), int(ratio[i])))
    return samples


def backends():
    engine = um.Rasterizer()

    def full(seg, shape, r):
        return engine.overlap(shape, um.PackedPolygons.from_segments(seg), r)

    def sampled(seg, shape, r):
        return engine.overlap_segments(shape, seg, r)

    return {"reference": lambda seg, shape, r: polygons2masks_overlap(shape, seg, r), "full": full, "sampled": sampled}


def full_resolution(fn):
    """Run fn with the sampled path disabled (the previous native path)."""

    def run(*args):
        saved = um._sampled_table
        um._sampled_table = lambda h, w, r: None
        try:
            return fn(*args)
        finally:
            um._sampled_table = saved

    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out")
    args = parser.parse_args()
    samples = load(args.inputs)
    fns = backends()
    names = list(fns)
    for seg, shape, r in samples:  # parity, and table calibration outside timing
        expected = polygons2masks_overlap(shape, seg, r)
        for name in ("full", "sampled"):
            got = full_resolution(fns[name])(seg, shape, r) if name == "full" else fns[name](seg, shape, r)
            assert got[0].dtype == expected[0].dtype and np.array_equal(got[0], expected[0]), name
            assert np.array_equal(got[1], expected[1]), name
    seconds = {name: [] for name in names}
    for round_index in range(args.rounds):
        order = names if round_index % 2 == 0 else names[::-1]
        for name in order:
            fn = fns[name]
            saved = um._sampled_table
            if name == "full":
                um._sampled_table = lambda h, w, r: None
            try:
                start = time.perf_counter()
                for seg, shape, r in samples:
                    fn(seg, shape, r)
                seconds[name].append(time.perf_counter() - start)
            finally:
                um._sampled_table = saved
    instances = sum(len(s[0]) for s in samples)
    report = {"samples": len(samples), "instances": instances, "rounds": args.rounds}
    ref = statistics.median(seconds["reference"])
    for name in names:
        med = statistics.median(seconds[name])
        report[name] = {
            "seconds": seconds[name],
            "us_per_sample_median": 1e6 * med / len(samples),
            "reference_over_backend": ref / med,
        }
        print(f"{name:10s} {1e6 * med / len(samples):8.1f} us/sample  {ref / med:6.2f}x")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
