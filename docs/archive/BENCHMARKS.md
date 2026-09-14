# Prototype benchmarks — 2026-09-12

These are feasibility measurements, not a release or training-throughput claim. The original PRD gates remain open. Reference: Ultralytics `795a556942a12fe0124cf767888194a1d0b83e2e`, NumPy 2.4.4 and OpenCV 4.13.0. Source and extension hashes, hardware, load, raw samples and confidence intervals are in each JSON. Small subsequent bounds-check, license and formatting edits mean these are source snapshots, not exact release-candidate measurements.

The separate [completed RTX 3070 training series](GPU_RESULTS.md) reports all
45 jobs / 90 epochs on the later alignment-corrected snapshot: modest epoch-time
gains and no consistent process-family RSS reduction.

## Mask wrapper latency and process memory

Nine fixed synthetic cases at 640×640: 0/1/5/20/100/255/256/500 instances at ratio 4, plus 100 instances at ratio 1; 16 vertices each. Five independent processes per backend/case, ten warmups, thirty timed calls. Wrapper packing and ordering are included; reference parity checks are outside timing. OS cache is uncontrolled. Both libraries use one native worker.

| Host | Nine-case geometric mean | 500 instances reference → wrapper | Peak process RSS reference → wrapper |
|---|---:|---:|---:|
| Apple M2 | 2.05× | 20.223 → 7.134 ms | 217.0 → 75.6 MiB |
| i5-10400 / RTX 3070 server (CPU operation) | 2.41× | 59.294 → 16.450 ms | 224.6 → 88.9 MiB |

The small one-instance server case is effectively tied (its confidence interval includes 1). Explicit bounded mode trades additional rasterization and wrapper overhead for less memory; see every case in the raw JSON, including regressions. The initial M2 wrapper geometric mean was 0.887× and is preserved in `bench/results/m2-initial.json`. Dirty-rectangle clearing and conservative composition bounds improved the measured suite; rasterization and interpolation remain the reference OpenCV operations.

**Historical data quality issue:** `server-support.json` used inherited Linux `ru_maxrss`, which reported the parent process high-water mark in all workers. Its memory fields are invalid and must not be used. `server-vmhwm.json` reruns the suite with each process’s `/proc/self/status` VmHWM. macOS uses byte-valued getrusage and was not affected.

## Independent peak allocation measurement

Memray 1.20.0, native tracing and Python allocator tracing, five fresh processes for each row/backend. One warmed call is traced; existing inputs are excluded, output allocation is included. Instrumented samples are never used for latency. Every output is checked against the oracle after tracing.

| Case | Reference peak | Auto wrapper peak | Bounded peak | Auto reduction |
|---|---:|---:|---:|---:|
| N=100, ratio=4 | 7.391 MiB | 2.874 MiB | 0.464 MiB | 61.1% |
| N=255, ratio=4 | 18.767 MiB | 6.683 MiB | 0.489 MiB | 64.4% |
| N=256, ratio=4 | 75.164 MiB | 6.781 MiB | 0.563 MiB | 91.0% |
| N=500, ratio=4 | 146.684 MiB | 12.775 MiB | 0.600 MiB | 91.3% |
| N=100, ratio=1 | 117.986 MiB | 39.861 MiB | 1.189 MiB | 66.2% |

The table uses M2 medians; the independent Linux run agrees within 0.003 MiB for these cases. This is working heap allocation, not whole-process RSS: the 100-instance ratio-4 process RSS falls much less than 30%. Raw results are `m2-allocations.json` and `server-allocations.json`; trace paths name retained local/server artifacts and are not public downloads.

## Full COCO DataLoader baseline — 2026-09-13

All 5,000 val2017 images, 36,335 instances, ordinary YOLODataset on both sides;
only the final Format is replaced. Batch 8, 640×640, overlap masks at ratio 4,
no augmentation or image RAM cache. Five alternating fresh processes per backend
and worker count on each host. The first epoch includes worker startup and queue
fill. [Protocol and reproduction](COCO_LOADER.md) define the exact scope.

| Host | Workers | Epoch reference → native (s) | Reference/native 95% paired bootstrap interval | Sampled family RSS reference → native (MiB) |
|---|---:|---:|---:|---:|
| Apple M2 | 0 | 15.335 → 14.862 | 1.026–1.042 | 315.5 → 315.1 |
| Apple M2 | 2 | 9.914 → 9.809 | 0.985–1.044 | 1142.0 → 1139.4 |
| Apple M2 | 8 | 10.526 → 10.175 | 0.978–1.059 | 3583.9 → 3583.6 |
| i5-10400 server | 0 | 21.555 → 23.477 | 0.904–1.003 | 365.0 → 364.9 |
| i5-10400 server | 2 | 15.477 → 15.515 | 0.987–1.042 | 1099.8 → 1091.6 |
| i5-10400 server | 8 | 14.916 → 14.916 | 0.993–1.006 | 3275.2 → 3280.0 |

Values are medians; a time ratio above 1 favors native. With only five pairs on
shared hosts, small differences remain uncertain. The server workers=0 median is
8.9% slower and is retained. These results do **not** meet a 10% full-loader
improvement gate or demonstrate a substantial process-memory reduction. Summed
family RSS double-counts shared pages and is sampled, not unique memory or peak
working allocation. Eight-worker first delivery alone takes about 6.8–6.9 s on
M2 and 9.7 s on the server. The remainder is not a separately warmed epoch.

Every one of the 60 samples verifies all 5,000 outputs in a separate untimed
pass, including masks, semantic targets, classes, ordering and metadata. Each
host also rebuilt the label cache with the original upstream scanner after
timing; all 30 outputs match that fresh reference. Raw samples and fingerprints
are in `coco-loader-{m2,server}.json` and `coco-fresh-{m2,server}.json` under
`bench/results`. The server snapshot differs from the local baseline only in
the C++ SPDX comment and packaging/license/lint metadata; the two differing
files are preserved under `coco-baseline-server-source`. This is not a claim
that the two hosts used byte-identical source trees or binaries.

### Why component gains do not transfer directly

Separate single-pass diagnostics use direct wall-clock probes with validated
call counts: 5,000 image loads, 4,952 mask calls and 625 collations per backend.
Both diagnostics verify complete outputs after restoring the original methods.

| Host | Image loading reference → native (s) | Mask formatting reference → native (s) | Native packing / rasterizer overlap (s) |
|---|---:|---:|---:|
| Apple M2 | 9.115 → 9.100 | 1.995 → 1.552 | 0.197 / 1.314 |
| i5-10400 server | 10.599 → 10.607 | 3.650 → 2.869 | 0.374 / 2.387 |

These are instrumented diagnostic totals, not repeated performance estimates;
nested inclusive times must not be added. Image loading and other unchanged
preprocessing account for much of the epoch. The native mask stage is faster
in these diagnostic runs, but that cannot override the repeated full-loader
results above. Stage JSONs and every individual duration are retained as
`coco-stages-{reference,native}-{m2,server}{,-samples}.json`.

An earlier cProfile attempt lost native-side outer call accounting: only 673 of
5,000 `__getitem__` calls and 1 of 626 `__next__` calls were recorded on both
hosts. Those native component times are invalid. The original profile reports
and explicit `coco-cprofile-audit-*.json` rejection records are retained. Direct
probes do not enable cProfile and assert every expected call count. The repeated
full-loader timing runs also do not enable cProfile.

## Validation and remaining evidence

113 tests passed on macOS and Linux, including 10,000 seeded differential cases and actual YOLODataset collated outputs with workers 0 and 2. Subsequent float32 int32-boundary/empty-packed regressions passed all 98 core tests on both hosts. ASan/UBSan passed the 98 core tests on macOS with leak detection disabled; only project native code was instrumented, not the private OpenCV archive.

The synthetic and full COCO results do not establish the representative-workload
PRD performance gate. Further work includes optimizing resampled real geometry,
augmented and non-overlap workloads, higher resolutions, full GPU epochs,
Windows/wheel validation, and release-candidate exact-source reruns.
