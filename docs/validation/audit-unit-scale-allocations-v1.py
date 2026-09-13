"""Audit sealed Memray traces without running native mask operations.

Usage: python SCRIPT ARCHIVE > RECEIPT
Requires the recorded Memray version; extracts only checked trace basenames.
"""

import hashlib
import io
import json
import shlex
import statistics
import sys
import tarfile
import tempfile
from pathlib import Path
from zipfile import ZipFile

import memray


def sha(data):
    return hashlib.sha256(data).hexdigest()


path = Path(sys.argv[1])
with tarfile.open(path) as archive:
    members = archive.getmembers()
    assert all(m.isfile() for m in members)
    assert len({m.name for m in members}) == len(members)
    files = {m.name: archive.extractfile(m).read() for m in members}
manifest = json.loads(files["manifest.json"])
assert set(manifest) == set(files) - {"manifest.json"}
assert all(sha(files[name]) == digest for name, digest in manifest.items())
assert sha(files["sources/allocations.py"]) == "afce4ed5bf7f164fc032e341485b7e61b187b52f8b8ff54a852a496cc77bfc3f"
assert sha(files["wheel.whl"]) == "2ca1875910cec38fd501af9108455609b6c926c8d14dcf570da7b9776a2e0c71"
before = json.loads(files["descriptor-before.json"])["descriptor"]
after = json.loads(files["descriptor-after.json"])["descriptor"]
assert before == after
assert before["wheel_sha256"] == sha(files["wheel.whl"])
assert (
    before["profile"]["bindings_sha256"]
    == sha(files["sources/bindings.cpp"])
    == "da270a9525d9c64d1e2d802cb8ec44cbe5743ed76ff6087936a9b00466c1f4a4"
)
assert before["fixture_code_sha256"] == sha(files["sources/feasibility.py"])
assert before["oracle_sha256"] == sha(files["sources/reference.py"])
with ZipFile(io.BytesIO(files["wheel.whl"])) as wheel:
    extension = [n for n in wheel.namelist() if n.startswith("ultrafast_maskops/_native.") and n.endswith(".so")]
    assert len(extension) == 1
    assert sha(wheel.read(extension[0])) == before["extension_sha256"]
    assert sha(wheel.read("ultrafast_maskops/__init__.py")) == before["wrapper_sha256"]
oracle = json.loads(files["oracle.json"])
assert oracle["descriptor"] == before
report = json.loads(files["results.json"])
assert report["memray"] == memray.__version__ == "1.20.0"
assert report["cases"] == oracle["cases"]
backends = ["reference", "wrapper", "bounded"]
order = [(r, c, b) for r in range(5) for c in (4, 5, 6, 7, 8) for b in backends[r % 3 :] + backends[: r % 3]]
assert len(report["results"]) == len(order) == 75
expected_traces = {f"traces/{r}-{c}-{b}.bin" for r, c, b in order}
assert {n for n in files if n.startswith("traces/")} == expected_traces
rows, pids, prior_end = {}, set(), None
trace_checks = []
with tempfile.TemporaryDirectory(prefix="mask-allocation-audit-") as tmp:
    for row, key in zip(report["results"], order):
        r, c, b = key
        assert (row["round"], row["case"], row["backend"]) == key
        name = f"{r}-{c}-{b}.bin"
        assert Path(row["trace"]).name == name
        trace = Path(tmp) / name
        trace.write_bytes(files["traces/" + name])
        reader = memray.FileReader(trace)
        metadata = reader.metadata
        assert metadata.has_native_traces and metadata.trace_python_allocators
        assert metadata.total_allocations > 0 and metadata.pid not in pids
        pids.add(metadata.pid)
        command = shlex.split(metadata.command_line)
        assert Path(command[0]).name == "allocations.py"
        assert command[1:] == ["--worker", str(c), "--backend", b, "--out", row["trace"]]
        assert metadata.end_time >= metadata.start_time
        assert prior_end is None or metadata.start_time >= prior_end
        prior_end = metadata.end_time
        high_water = sum(x.size for x in reader.get_high_watermark_allocation_records())
        assert high_water == metadata.peak_memory == row["peak_tracked_allocation_bytes"] > 0
        expected = oracle["expected"][str(c)]["overlap"]
        digests = [f"{x['sha256']}:{x['dtype']}:{tuple(x['shape'])}" for x in expected]
        assert row["parity"] is True and row["output_hashes"] == digests
        rows[key] = high_water
        trace_checks.append(
            {
                "trace": name,
                "pid": metadata.pid,
                "total_allocations": metadata.total_allocations,
                "peak_bytes": high_water,
                "native_traces": True,
                "python_allocators": True,
            }
        )
summary = []
for c in (4, 5, 6, 7, 8):
    values = {b: [rows[r, c, b] for r in range(5)] for b in backends}
    medians = {b: statistics.median(v) for b, v in values.items()}
    summary.append(
        {
            "case": c,
            "fixture": report["cases"][c],
            "peak_bytes_raw": values,
            "peak_bytes_median": medians,
            "reduction_vs_reference": {b: 1 - medians[b] / medians["reference"] for b in ("wrapper", "bounded")},
        }
    )
print(
    json.dumps(
        {
            "scope": "75 separate traced warmed overlap calls; input excluded, output and packing included; not RSS or latency",
            "complete": True,
            "archive_sha256": sha(path.read_bytes()),
            "auditor_sha256": sha(Path(__file__).read_bytes()),
            "memray": memray.__version__,
            "trace_checks": trace_checks,
            "summary": summary,
            "limits": [
                "Instrumentation is not used for timing.",
                "Before/after installed descriptors bind the frozen environment; original workers did not record separate wheel descriptors.",
                "Parity is checked against retained oracle hashes; this auditor does not rerun mask operations.",
                "Memray coverage is tracked Python/native heap allocation, not stack, device or whole-process memory.",
                "Only overlap wrapper and bounded composition are covered; non-overlap allocation is unmeasured.",
            ],
        },
        indent=2,
    )
)
