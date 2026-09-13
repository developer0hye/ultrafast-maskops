"""Audit the sealed Linux paired-wheel experiment without importing a backend.

Usage: python audit-paired-wheels-linux-v1.py ARCHIVE > RECEIPT
Independent weighted-multiset bootstrap; includes every measured condition.
"""

import hashlib
import io
import itertools
import json
import math
import statistics
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bootstrap(pairs):
    distribution = []
    for indices in itertools.combinations_with_replacement(range(5), 5):
        weight = math.factorial(5)
        for index in range(5):
            weight //= math.factorial(indices.count(index))
        ratio = statistics.median(pairs[i][0] for i in indices) / statistics.median(pairs[i][1] for i in indices)
        distribution.extend([ratio] * weight)
    assert len(distribution) == 3125
    distribution.sort()

    def percentile(p):
        x = p * 3124
        lo, hi = math.floor(x), math.ceil(x)
        return distribution[lo] if lo == hi else distribution[lo] * (hi - x) + distribution[hi] * (x - lo)

    left, right = [statistics.median(pair[i] for pair in pairs) for i in (0, 1)]
    return {
        "baseline_median": left,
        "candidate_median": right,
        "baseline_over_candidate": left / right,
        "paired_bootstrap_ci95": [percentile(0.025), percentile(0.975)],
        "ordered_resamples": 3125,
    }


path = Path(sys.argv[1])
with tarfile.open(path) as archive:
    members = archive.getmembers()
    assert all(member.isfile() for member in members)
    assert len({member.name for member in members}) == len(members)
    files = {member.name: archive.extractfile(member).read() for member in members}
manifest = json.loads(files["manifest.json"])
assert set(manifest) == set(files) - {"manifest.json"}
assert all(sha(files[name]) == digest for name, digest in manifest.items())
report = json.loads(files["series.json"])
assert report["complete"] is True and "error" not in report
assert report["repeats"] == 5 and report["samples"] == 30
operations = ["overlap", "bounded", "masks"]
assert report["operations"] == operations
assert report["harness_sha256"] == sha(files["sources/harness.py"])
cases = [{"n": n, "h": 640, "w": 640, "ratio": 4, "vertices": 16} for n in (0, 1, 5, 20, 100, 255, 256, 500)]
cases.append({"n": 100, "h": 640, "w": 640, "ratio": 1, "vertices": 16})
assert report["cases"] == cases
assert set(report["oracles"]) == {"baseline", "candidate"}
expected_binaries = {
    "baseline": (
        "b5257f36505db01e1d1c0e46db2a059c3b64fc25c4f1b9281095bbc550383026",
        "99e10b49b6ac531eabd9797c4f19d3254ff72767aa5cac1d4afbb961425e9a10",
        151,
    ),
    "candidate": (
        "1b08974c11b1bc00e7e3f85f68fb6f8505e9e2edc5ca55b941ca6ef09b789fad",
        "c63940b67b291123d6d5d895858cb981afaa17bd1aea94856cd9f40dcebb1813",
        201,
    ),
}
raw_names, oracles = set(), {}
for label, entry in report["oracles"].items():
    name = f"runs/oracle-{label}.json"
    assert Path(entry["path"]).name == Path(name).name
    assert sha(files[name]) == entry["sha256"] and json.loads(files[name]) == entry["report"]
    raw_names.add(name)
    oracle = entry["report"]
    oracles[label] = oracle
    assert oracle["cases"] == cases and set(oracle["expected"]) == {str(i) for i in range(9)}
    for index, case in enumerate(cases):
        expected = oracle["expected"][str(index)]
        assert set(expected) == {"input_sha256", "overlap", "bounded", "masks"}
        height, width = case["h"] // case["ratio"], case["w"] // case["ratio"]
        overlap = [([height, width], "uint8" if case["n"] <= 255 else "int32"), ([case["n"]], "int64")]
        # Upstream np.array([]) intentionally preserves shape (0,) / float64.
        masks = [([case["n"], height, width], "uint8")] if case["n"] else [([0], "float64")]
        shapes = {"overlap": overlap, "bounded": overlap, "masks": masks}
        for operation, layout in shapes.items():
            assert len(expected[operation]) == len(layout)
            for value, (shape, dtype) in zip(expected[operation], layout):
                assert value["shape"] == shape and value["dtype"] == dtype
                assert len(value["sha256"]) == 64 and all(c in "0123456789abcdef" for c in value["sha256"])
    descriptor = oracle["descriptor"]
    wheel_sha, kernel_sha, test_count = expected_binaries[label]
    assert descriptor["wheel_sha256"] == sha(files[f"wheels/{label}.whl"]) == wheel_sha
    with ZipFile(io.BytesIO(files[f"wheels/{label}.whl"])) as wheel:
        extensions = [
            name for name in wheel.namelist() if name.startswith("ultrafast_maskops/_native.") and name.endswith(".so")
        ]
        assert len(extensions) == 1 and sha(wheel.read(extensions[0])) == descriptor["extension_sha256"]
        wrapper = wheel.read("ultrafast_maskops/__init__.py")
        assert sha(wrapper) == descriptor["wrapper_sha256"]
        assert wrapper == files[f"sources/{label}/python/ultrafast_maskops/__init__.py"]
    assert descriptor["profile"]["bindings_sha256"] == kernel_sha
    for key, source in (
        ("bindings_sha256", "src/bindings.cpp"),
        ("cmake_sha256", "CMakeLists.txt"),
        ("template_sha256", "src/build_profile.h.in"),
    ):
        assert descriptor["profile"][key] == sha(files[f"sources/{label}/{source}"])
    for key, source in (
        ("harness_sha256", "harness.py"),
        ("fixture_code_sha256", "feasibility.py"),
        ("oracle_sha256", "reference.py"),
    ):
        assert descriptor[key] == sha(files["sources/" + source])
    suites = ET.fromstring(files[f"validation/{label}-tests-v1.xml"]).findall("testsuite")
    assert sum(int(s.attrib["tests"]) for s in suites) == test_count
    assert all(int(s.attrib[key]) == 0 for s in suites for key in ("errors", "failures", "skipped"))
    smoke = json.loads(files[f"validation/{label}-wheel-validation.json"])
    assert smoke["clean"] is True and smoke["notice_files_checked"] == 9
    command = entry["command"]
    assert command[2:3] == ["--root"] and command[4:5] == ["--wheel"]
    assert command[6:] == ["--samples", "30", "--out", entry["path"], "--worker", "oracle"]
    assert Path(command[0]).name == "python" and Path(command[1]).name == "compare_wheels.py"
assert oracles["baseline"]["expected"] == oracles["candidate"]["expected"]
a, b = [oracles[label]["descriptor"] for label in ("baseline", "candidate")]
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
assert all(a[key] == b[key] for key in common) and a["private_opencv_threads"] == 1
assert all(
    a["profile"][key] == b["profile"][key] for key in ("cmake_sha256", "template_sha256", "compiler", "build_type")
)
normalize = lambda text: [
    line for line in text.splitlines() if not line.strip().startswith(("Timestamp:", "Install to:"))
]
assert normalize(a["private_opencv_build"]) == normalize(b["private_opencv_build"])
freeze = lambda label: [
    line
    for line in files[f"validation/{label}-freeze-v1.txt"].decode().splitlines()
    if not line.startswith("ultrafast-maskops")
]
assert freeze("baseline") == freeze("candidate")
order = [
    (r, c, op, version)
    for r in range(5)
    for c in range(9)
    for op in operations
    for version in (("baseline", "candidate") if r % 2 == 0 else ("candidate", "baseline"))
]
assert len(report["records"]) == len(order) == 270
rows = {}
for entry, key in zip(report["records"], order):
    repeat, case, operation, label = key
    assert (entry["repeat"], entry["case"], entry["operation"], entry["version"]) == key
    name = f"runs/r{repeat}-c{case}-{operation}-{label}.json"
    raw_names.add(name)
    data = files[name]
    assert sha(data) == entry["sha256"]
    raw = json.loads(data)
    assert raw["descriptor"] == oracles[label]["descriptor"]
    assert {k: v for k, v in raw.items() if k != "descriptor"} == entry["report"]
    oracle_entry = report["oracles"][label]
    assert entry["path"] == str(Path(oracle_entry["path"]).parent / Path(name).name)
    assert entry["command"] == oracle_entry["command"][:8] + [
        "--out",
        entry["path"],
        "--worker",
        "measure",
        "--oracle",
        oracle_entry["path"],
        "--case",
        str(case),
        "--operation",
        operation,
    ]
    assert raw["case"] == case and raw["operation"] == operation
    assert len(raw["samples_ns"]) == 30 and all(isinstance(v, int) and v > 0 for v in raw["samples_ns"])
    assert raw["median_ns"] == statistics.median(raw["samples_ns"])
    expected = oracles[label]["expected"][str(case)]
    assert raw["input_sha256"] == expected["input_sha256"] and raw["output_hashes"] == expected[operation]
    assert all(
        math.isfinite(raw[k]) and raw[k] > 0
        for k in ("peak_rss_bytes", "rss_after_warmup_bytes", "available_ram_before_bytes", "available_ram_after_bytes")
    )
    assert raw["peak_rss_bytes"] >= raw["rss_after_warmup_bytes"]
    assert all(
        len(raw[k]) == 3 and all(math.isfinite(v) and v >= 0 for v in raw[k]) for k in ("load_before", "load_after")
    )
    rows[key] = raw
assert {name for name in files if name.startswith("runs/") and name.endswith(".json")} == raw_names
assert {name for name in files if name.startswith("runs/") and name.endswith(".log")} == {
    name[:-5] + ".log" for name in raw_names
}
summaries = []
assert len(report["summary"]) == 27
for recorded, (case, operation) in zip(report["summary"], itertools.product(range(9), operations)):
    assert recorded["case"] == case and recorded["operation"] == operation
    result = {"case": case, "fixture": cases[case], "operation": operation}
    for metric in ("median_ns", "peak_rss_bytes"):
        pairs = [
            (rows[(r, case, operation, "baseline")][metric], rows[(r, case, operation, "candidate")][metric])
            for r in range(5)
        ]
        value = bootstrap(pairs)
        assert set(recorded[metric]) == set(value)
        for k in value:
            left = value[k] if isinstance(value[k], list) else [value[k]]
            right = recorded[metric][k] if isinstance(recorded[metric][k], list) else [recorded[metric][k]]
            assert len(left) == len(right), (case, operation, metric, k)
            assert all(math.isclose(x, y, rel_tol=1e-12) for x, y in zip(left, right)), (case, operation, metric, k)
        result[metric] = dict(value, raw_pairs=pairs)
    summaries.append(result)
print(
    json.dumps(
        {
            "scope": "complete Linux ROI versus masks-only public-call comparison; synthetic masks only, not training or working allocation",
            "complete": True,
            "measured_processes": 270,
            "oracle_processes": 2,
            "latency_samples": 8100,
            "archive_sha256": sha(path.read_bytes()),
            "auditor_sha256": sha(Path(__file__).read_bytes()),
            "harness_sha256": report["harness_sha256"],
            "statistics": "five paired process medians; independent weighted multiset enumeration of 3125 ordered resamples; linear percentile interpolation",
            "summary": summaries,
            "limitations": [
                "Shared host and uncontrolled OS cache; five pairs do not establish generality.",
                "Whole-process RSS includes imports, wheel verification, inputs and warmups; not isolated temporary allocation.",
                "Parity comes from retained source workers checking outputs before/after timing and two separate oracles; this auditor does not rerun native code.",
                "Sanitizer, real-loader timing, GPU throughput, M2 comparison and release portability remain separate gates.",
            ],
        },
        indent=2,
    )
)
