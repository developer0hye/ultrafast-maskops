# Prototype benchmarks — 2026-09-12

These are feasibility measurements, not a release or training-throughput claim. The original PRD gates remain open. Reference: Ultralytics `795a556942a12fe0124cf767888194a1d0b83e2e`, NumPy 2.4.4 and OpenCV 4.13.0. Source and extension hashes, hardware, load, raw samples and confidence intervals are in each JSON. Small subsequent bounds-check, license and formatting edits mean these are source snapshots, not exact release-candidate measurements.

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

## Validation and remaining evidence

113 tests passed on macOS and Linux, including 10,000 seeded differential cases and actual YOLODataset collated outputs with workers 0 and 2. Subsequent float32 int32-boundary/empty-packed regressions passed all 98 core tests on both hosts. ASan/UBSan passed the 98 core tests on macOS with leak detection disabled; only project native code was instrumented, not the private OpenCV archive.

These nine synthetic cases do not establish the real-corpus PRD performance gate. Required next evidence includes real COCO polygons, high-vertex and resolution distributions, CPU DataLoader throughput, full GPU epochs, workers 8, Windows/wheel validation, and release-candidate exact-source reruns.
