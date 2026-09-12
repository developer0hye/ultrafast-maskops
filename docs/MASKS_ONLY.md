# Skip unused area reduction for non-overlap masks

`Rasterizer.masks()` previously called the overlap preparation primitive and
discarded its uint64 area vector. It now uses a native masks-only entry point
with the same dimension/color checks, output ownership, mutex, scratch reuse,
OpenCV fill and resize sequence. It does not allocate the area vector or scan
every resized mask with `cv::sum`. Overlap preparation still needs these sums
and retains its existing implementation.

The new CPython 3.12 M2 wheel passed **139 tests**, including the 10,000 seeded
differential cases, worker integration and deterministic CPU model updates.
NumPy-only import/native smoke, wheel RECORD, installed bytes and all nine
notice files in wheel/sdist/install also passed. Source hashes, wheel hashes,
JUnit and logs are retained under `docs/validation/masks-only-*` and
`docs/validation/mask-masks-only-*`. The wheel is tagged macOS 26 arm64; Windows,
other Python versions and older macOS execution are not established.

## Isolated mechanism measurement

`bench/masks_only.py` uses five fresh processes. Each tests six deterministic
synthetic polygon sets, verifies both paths against the frozen OpenCV reference,
warms both paths, then randomizes their order for each of 30 paired samples.
Polygon packing is outside the measurement. The legacy callable reproduces the
previous wrapper's validation, locking and area-discard behavior using the
unchanged native `raster` method in the same binary. This isolates the removed
work; it is not an old-wheel versus new-wheel comparison, a full API benchmark,
or a measurement of training throughput. Ratios are ratios of the medians of
five process medians; the raw timings and all process results are retained in
`bench/results/masks-only-m2-v1.json`.

| Image | Instances | Mask ratio | Previous path (ms) | Masks-only (ms) | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 640×640 | 1 | 4 | 0.013917 | 0.012729 | 1.093× |
| 640×640 | 100 | 4 | 1.426063 | 1.321938 | 1.079× |
| 640×640 | 500 | 4 | 7.016187 | 6.492730 | 1.081× |
| 1280×1280 | 100 | 4 | 4.831209 | 4.445083 | 1.087× |
| 640×640 | 100 | 1 | 2.567250 | 0.970521 | 2.645× |
| 640×640 | 100 | 16 | 0.533355 | 0.520375 | 1.025× |

The removed area array is exactly `8*N` payload bytes. This does not imply a
comparable RSS change; mask output and raster scratch sizes are unchanged. No
process-memory improvement is claimed from this experiment. The larger ratio-1
effect is consistent with avoiding a scan of every full-resolution output, but
cache/bandwidth attribution has not been separately measured.

Full real COCO non-overlap verification of this new installed wheel with the
dataset wheel is running separately. The completed 18-process overlap/non-overlap
experiment in [COCO_BOUNDS.md](COCO_BOUNDS.md) belongs to the earlier corrected
binary. The ongoing server GPU series also retains its original binaries, so
neither experiment may be relabeled as this optimization's throughput result.
Actual workload performance and exact-candidate platform validation remain open.
