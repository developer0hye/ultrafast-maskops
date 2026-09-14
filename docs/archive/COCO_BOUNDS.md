# Resampled polygon optimization candidate

Ultralytics resamples every nonempty polygon in the current full COCO fixture
to 1,000 vertices before Format. This candidate caches per-contour inclusive
bounds while taking the native snapshot and removes consecutive identical
integer vertices. It does not simplify nonzero or collinear edges, reorder
contours, change the public instance segments, or remove the terminal copy of
the first vertex. The latter preserves nonzero edge insertion order. An
all-identical contour retains one point and remains a distinct instance.

Rasterization, interpolation, area sorting and overlap composition retain their
previous contracts. Input point/offset/order buffers are copied with memcpy into
aligned native storage before typed access, including C-contiguous but unaligned
NumPy arrays. Bounds are clipped for each raster size with widened arithmetic,
so a packed snapshot remains reusable at different sizes and downsample ratios.

The extension also embeds the SHA-256 values of its binding source, CMake file
and build-profile template. `backend_info()["build"]` exposes those values and
the compiler/build type. Tests compare the compiled values to the source files;
these are build provenance, not a claim of cross-platform binary reproducibility.

## Measured bounds/deduplication snapshot

The following experiment measures source `01247c5`, before the subsequent
untyped input-access correction described below. `coco-bounds-source.tar.gz`
under `bench/results` contains every file named by the measured source hashes.
Five alternating fresh processes per backend/worker count, all 5,000 COCO images,
batch 8, 640×640, ratio 4, overlap masks, no augmentation or image RAM cache.
The epoch includes worker startup. Both hosts rebuilt the original label cache
after timing; all 60 full output hashes match their host's fresh reference.

| Host | Workers | Reference → native epoch median (s) | Reference/native paired bootstrap 95% interval |
|---|---:|---:|---:|
| M2 | 0 | 15.117 → 14.198 | 1.035–1.093 |
| M2 | 2 | 9.912 → 9.356 | 1.008–1.087 |
| M2 | 8 | 9.921 → 9.449 | 0.966–1.101 |
| i5-10400 server | 0 | 21.191 → 22.894 | 0.887–1.140 |
| i5-10400 server | 2 | 15.595 → 14.918 | 0.999–1.095 |
| i5-10400 server | 8 | 14.915 → 14.853 | 1.001–1.017 |

M2 median time reductions are 6.1%, 5.6% and 4.8%. The server workers=0 median
is 8.0% slower; the wide interval includes 1. The 10% full-loader gate remains
unmet. Sampled summed family RSS is approximately unchanged and double-counts
shared pages. Raw reports retain every unfavorable result and uncertainty;
`coco-bounds-{m2,server}-runs.tar.gz` additionally retain per-process output,
logs and memory traces. `coco-bounds-evidence-audit.json` binds the source and
run archives and records the consistency check of medians, intervals, fresh
references and stage samples. These remain shared-host measurements with five
pairs, not proof of a general speedup.

Separate direct probes validate 5,000 image loads, 4,952 mask calls and 625
collations. M2 mask formatting took 1.923 s in the reference versus 1.238 s
native; image loading took 8.775 versus 8.970 s. Native packing consumed 0.234 s
and rasterizer overlap 0.964 s. These are one instrumented diagnostic pass per
backend, with nested inclusive times; they do not replace the repeated epoch
results. Both complete outputs were independently checked after instrumentation.

## Input alignment correction and current checks

The new unaligned-input regression exposed a sanitizer failure at the typed
`array_t<int64_t>::data()` path used as a memcpy source. The current code obtains
input addresses through untyped `py::array::data()` for offsets, points and
composition order, then copies bytes into aligned native storage. The original
failure and the passing isolated reproduction are retained in `docs/validation`.
This correction was made after the measured snapshot: the table above must not
be presented as an exact measurement of the corrected binary.

The corrected source passed **139 tests on each host**, including 10,000 seeded
differential cases, actual DataLoader integration and four deterministic CPU
YOLO11n-seg training checks. Those compare all loss components, parameter
gradients and model states after two SGD steps for foreground/background and
both overlap modes. JUnit reports are `coco-byte-access-tests-{m2,server}.xml`.

Project-code ASan/UBSan passed **120 core tests** on M2. Leak detection was
disabled; the private OpenCV archives were not instrumented. The isolated
sanitized package and its subprocesses both loaded the instrumented extension.
The macOS sanitizer removed DYLD_INSERT_LIBRARIES from the startup environment;
the runner restored it inside Python before running pytest so import-order child
processes also received the runtime. The prior interceptor-initialization failure
is not treated as a passing test or as a kernel bug.

The preceding measured snapshot also passed a combined FastYOLODataset+
FastFormat augmentation pilot: 64 primary outputs at workers 0 and 2, with
independent reference replay and native processes. The native cache was cold in
the first case and warm in the second. Its original verifier source is retained
as `coco-augmented-pilot-script.py.txt`. This is not full-corpus evidence for the
corrected kernel.

The corrected build subsequently completed the full overlap augmentation check:
all nine fresh processes at workers 0/2/8 produced 5,000 outputs each, with exact
reference/replay/native agreement within every worker count. Complete evidence
is `bench/results/coco-byte-access-augmented-both-m2.json`, including verifier,
runtime, source and input hashes. It measures correctness, not throughput.
Full non-overlap verification also completed: nine independent processes,
5,000 images and 625 batches each, with reference/replay/native agreement at
workers 0/2/8. Its report and log are
`bench/results/coco-byte-access-augmented-both-nonoverlap-m2.{json,log}`;
`docs/validation/coco-nonoverlap-artifact-audit.json` independently compares all
child reports and retains their hashes. Both augmentation results predate the
subsequent masks-only kernel and MSVC build configuration changes.
Exact-source performance reruns,
the repeated GPU comparison and portable wheels remain open. GPU pilots are
documented separately in [CUDA_WHEEL_VALIDATION.md](CUDA_WHEEL_VALIDATION.md).
This candidate is not release-ready.
