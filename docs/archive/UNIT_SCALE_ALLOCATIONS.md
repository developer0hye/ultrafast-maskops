# Unit-scale candidate: traced overlap allocation

The tested unit-scale Linux wheel reduces peak tracked allocation versus the
original Python/OpenCV overlap implementation by 61–91% through the public
wrapper and 94–99.6% in explicit bounded mode on the five fixed crowded cases.
This meets the proposed 30% working-allocation target on this Linux synthetic
suite. It does not establish an incremental memory gain over ROI or masks-only;
those native binaries were not included in this allocation series.

The result is from 75 fresh processes: five repetitions × N=100/255/256/500 at
ratio 4 plus N=100 at ratio 1 × reference/public wrapper/explicit bounded mode.
Backend order rotates each repetition. Inputs are fixed 640×640, 16 vertices
per polygon. Each selected backend warms ten calls before tracking one complete
call, including packing, new Rasterizer, output and temporary heap allocation.
The pre-existing fixture is excluded. Output remains alive when tracking stops.
Reference parity hashing occurs after tracking, so verification copies are not
counted in this metric. Instrumented samples are never used for latency claims.

Memray 1.20.0 tracks native allocations and Python allocators. All 75 retained
binary traces were reopened: high-water allocation-record sums equal trace
metadata and the recorded worker peak. Recorded commands, unique worker PIDs,
sequential trace intervals and output hashes agree with the full paired-suite
oracle. Installed descriptors before and after the series are identical and
match the tested wheel. The original allocation workers do not include separate
per-worker wheel descriptors; this is explicitly retained as an evidence limit.

| N | Ratio | Reference peak MiB | Public wrapper peak MiB | Reduction | Bounded peak MiB | Reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 4 | 7.3908 | 2.8753 | 61.10% | 0.4596 | 93.78% |
| 255 | 4 | 18.7673 | 6.6870 | 64.37% | 0.4871 | 97.40% |
| 256 | 4 | 75.1640 | 6.7849 | 90.97% | 0.5605 | 99.25% |
| 500 | 4 | 146.6838 | 12.7829 | 91.29% | 0.6015 | 99.59% |
| 100 | 1 | 117.9865 | 39.8626 | 66.21% | 1.1915 | 98.99% |

Values are medians of the five per-call peaks; all five raw values per backend
and condition are in the audit receipt. Tracked heap allocation is not stack,
GPU/device memory, total process RSS, or DataLoader process-family RSS. Bounded
mode rerasterizes instances and can cost more CPU time; it is a memory tradeoff.
The public wrapper's dispatch policy remains unchanged. Non-overlap allocation
is not measured by this series.

The [87-file archive](../bench/results/mask-unit-scale-linux-allocations-evidence-v1.tar.gz)
contains all 75 traces, results/logs, before/after descriptors, oracle, measured
wheel, validation identity, source code and manifest. SHA-256:
`369ea7a3e4b7d8488d149256d9b432df78b67426de625506536ea72f3a499eb2`.
The downloaded archive was reopened and all member hashes checked.

[Audit receipt](validation/mask-unit-scale-linux-allocations-audit-v1.json) ·
[trace auditor](validation/audit-unit-scale-allocations-v1.py) ·
[sealing script](validation/seal-unit-scale-allocations-v1.py).
Run the auditor with the archive path under Memray 1.20.0; it does not execute
native mask operations. The [negative-case driver](validation/audit-unit-scale-allocations-negative-v1.py)
accepted the valid archive and rejected five separately rehashed forgeries:
wrong peak, missing worker, missing trace, swapped traces and wrong output.
[Negative-case receipt](validation/mask-unit-scale-linux-allocations-negative-v1.json).

All Memray diagnostic output is preserved in `run.log`, including malloc/free
symbol correction messages. The flags, allocation counts, trace metadata and
record-sum consistency were checked; this is not a proof that the tracer
captures every possible allocation mechanism.

## Full-loader follow-up

A separate uninstrumented series is running from frozen source commit `9c4ece5`
and the tested wheel. The [launch script](validation/run-unit-scale-linux-loader-v1.sh)
runs overlap and non-overlap sequentially: 5,000 real COCO val2017 images,
workers 0/2/8, five paired fresh processes per backend/worker count, batch 8,
640-pixel images and mask ratio 4. It uses ordinary YOLODataset on both sides,
replacing only Format mask generation. Each process verifies all batch fields
in a second untimed pass and rechecks input bytes. No full-loader result is
claimed until both matrices and their independent audits are complete.
