# Segment geometry: where augmented segmentation loading spends CPU time

## Profile of the unmodified pipeline

cProfile of 600 augmented COCO val2017 segmentation samples (training
defaults: mosaic, random perspective, HSV, flips), single process, Apple M2:
11.34 ms per sample under the profiler.

| Function | Share | Kind |
|---|---:|---|
| `RandomPerspective.apply_segments` (incl. per-instance `segment2box`, 16,508 calls) | 20.7% | per-instance Python loop over small NumPy calls |
| JPEG decode (`imdecode`) + file read (`fromfile`) | 21.9% | already C |
| `warpAffine` | 14.0% | already C |
| `resample_segments` (per polygon `insert`/`linspace`/`interp`) | 10.2% | per-polygon Python loop |
| HSV (`cvtColor` + `LUT`) | 11.4% | already C |
| mask rasterization (`fillPoly` + `resize` + NumPy glue) | ~6% | the stage this project first targeted |

Measured directly on 2,004 captured samples, the mask stage is **6.3%** of
`__getitem__` time. An infinitely fast mask kernel therefore bounds the loader at
**1.067×**, which is why earlier mask-only work measured at most about 5% loader
gains. The per-instance and per-polygon geometry loops are about five times
larger. Both are dominated by interpreter dispatch rather than arithmetic, so a
native batch per sample removes most of their cost.

Inside the mask stage (27,497 real instances, 1,000-point polygons after
resampling), the reference spends per instance: `fillPoly` 18.9 µs (42%),
`resize` 11.4 µs (25%), `astype`+`sum` 6.5 µs (14%), full-size `zeros` 3.1 µs
(7%), overlap composition and sorting 3.4 µs (8%).

## Exactness strategy

Every accelerated function returns byte-identical results to the pinned
reference, verified by `tests/test_geometry.py`:

- Native arithmetic follows the reference operation by operation in the input
  precision. The extension is compiled with `-ffp-contract=off`, so each multiply
  and add rounds separately, as separate NumPy ufuncs do.
- `cv2.pointPolygonTest`'s float branch multiplies two float differences, which
  is exact in double; its result is therefore independent of contraction.
- `np.interp` computes `slope*(x-xp)+fp`, which the NumPy build may fuse (NumPy
  2.4.4 on arm64 does). The variant is calibrated against NumPy's own float64
  output before native resampling is enabled; if it cannot be determined,
  resampling stays in NumPy.
- Batches containing NaN, infinity or negative zero run the reference code
  unchanged: NumPy's reductions and `clip` loops give such values a sign of zero
  that depends on array shape, which only identical calls reproduce.

## Results

### M2, COCO val2017 segmentation, augmented `__getitem__`

Measured at commit `490eed2` with 1,000 samples per process, five alternating
rounds and identical seeds, on a quiet host (1-minute load average 1.9–3.0).
The full output digest (image, classes, boxes, masks, metadata) is identical
for every backend and round.

| Backend | ms / sample | Loader speedup |
|---|---:|---:|
| Reference | 10.09 | 1.00× |
| Masks only (`accelerate_dataset`) | 9.40 | 1.07× |
| Geometry + masks (`accelerate_dataset` + `accelerate_geometry`) | **6.95** | **1.45×** |

Per sample, the timed segment stages fall from 3.76 ms to 0.86 ms (4.4×):

| Stage | Reference | Geometry + masks | Speedup |
|---|---:|---:|---:|
| Resampling | 0.82 ms | 0.18 ms | 4.6× |
| `apply_segments` | 2.24 ms | 0.61 ms | 3.7× |
| `Format._format_segments` | 0.69 ms | 0.07 ms | 10.3× |

Removing the segment stages entirely would bound this loader at
10.09 / 6.33 = 1.59×. The rest is JPEG decoding, `warpAffine`, HSV conversion
and file reads, which already run in OpenCV. Before the sampled mask path
(commit `134184c`, three rounds), the same loader measured 7.35 ms (1.40×).

### i5-10400 (Linux), with the sampled mask path

Measured at commit `54fc9f6` with 1,000 samples per process and five
alternating rounds, on an otherwise idle host. The output digest is identical
for every backend and round.

| Backend | ms / sample | Loader speedup |
|---|---:|---:|
| Reference | 14.15 | 1.00× |
| Masks only | 13.03 | 1.09× |
| Geometry + masks | **8.71** | **1.63×** |

Per sample, the timed segment stages fall from 6.52 ms to 1.08 ms (6.0×):

| Stage | Reference | Geometry + masks | Speedup |
|---|---:|---:|---:|
| Resampling | 1.66 ms | 0.39 ms | 4.3× |
| `apply_segments` | 3.60 ms | 0.56 ms | 6.5× |
| `Format._format_segments` | 1.25 ms | 0.14 ms | 8.9× |

The `_format_segments` time includes the instance reordering that both
implementations perform. Removing the segment stages entirely would bound this
loader at 14.15 / 7.64 = 1.85×.

### One augmented epoch at COCO scale (i5-10400, Linux)

`bench/coco_epoch.py` measures what a training run waits for: dataset
construction plus one full epoch of the train-mode loader. Both sides build the
dataset and loader as Ultralytics' trainer does:

- `init_seeds(0)`;
- the `build_yolo_dataset` arguments with the default configuration, which
  includes mosaic and every default augmentation at 640 px;
- `build_dataloader` on CPU with 8 workers and batch 16.

The accelerated side is `FastYOLODataset(annotation_cache="fast")`
(ultrafast-yolo-dataset `ad1d738`) with `accelerate_dataset` and
`accelerate_geometry` in persistent mode (commit `54fc9f6`).

The corpus is the size of COCO train2017: 118,287 images (19.3 GB) and 117,152
label files. It is the 5,000 COCO val2017 images and labels copied under new
names, so its instances follow COCO's distribution. Each measurement is a fresh
process. Before timing, the first 64 batches of each backend were digested in
separate processes and were identical.

| Label cache | Stage | Reference | Accelerated | Speedup |
|---|---|---:|---:|---:|
| Hit (later runs) | Constructor | 3.8 s | 0.6 s | 6.5× |
| | Epoch (7,393 batches) | 376.0 s | 268.2 s | 1.40× |
| | **Total** | **379.7 s** | **268.8 s** | **1.41×** |
| Miss (first run) | Constructor | 53.5 s | 9.8 s | 5.5× |
| | Epoch | 372.0 s | 267.7 s | 1.39× |
| | **Total** | **425.5 s** | **277.5 s** | **1.53×** |

- **Rounds:** hit is the median of three alternating rounds, miss of two. Rounds
  differ by less than 1%.
- **Throughput:** rises from 315 to 441 images per second.
- **Memory:** the peak memory of the process and its workers (PSS) falls from
  4.42 GB to 2.35 GB (1.9×).
- **Workers:** in one process, a sample loads 1.63× faster (above). With eight
  workers sharing six cores, the epoch gains 1.40×.

A second campaign applies only the maskops calls to the unmodified
`YOLODataset` (`--backends reference maskops`, three alternating rounds, same
verification digest). It separates what each library contributes:

| Label cache hit | Reference | maskops only | Both |
|---|---:|---:|---:|
| Constructor | 3.8 s | 3.8 s | 0.6 s |
| Epoch | 374.9 s | 268.4 s | 268.2 s |
| Total | 378.7 s | 272.2 s (1.39×) | 268.8 s (1.41×) |
| Peak memory (PSS) | 4.41 GB | 4.23 GB | 2.35 GB |

The reference and "both" columns come from separate campaigns, whose
reference medians agree within 0.3%.

- **maskops:** provides the whole epoch gain.
- **FastYOLODataset:** shortens construction (53.5 s → 9.8 s without a label
  cache) and keeps labels in a packed cache instead of per-image Python
  objects. This is the only difference between the last two columns, which
  roughly halves the memory of the process and its workers.

### Anatomy of `apply_segments`

On 1,500 captured COCO calls (median 24 instances and 24,000 points per call;
67% of instances cross the image border after mosaic), one call takes 2.09 ms in
the reference and 0.60 ms here: 0.41 ms is the NumPy part kept verbatim (the
homogeneous array and `xy @ M.T`), the rest one native pass that divides by w,
computes every box and clips.

The matrix product stays in NumPy deliberately. Apple's Accelerate reproduces a
fused multiply-add chain in k order for almost every size, but computes the
last row of a 1,001-row product differently; random calibration data cannot
reliably expose a one-row tail, so no calibration can guarantee a BLAS's order
for every size and position.

Inside the native pass, work is skipped only when its result is provably
discarded: a division whose rounded `t` the reference certainly rejects, and a
corner's point-in-polygon loop when the corner lies above, below or right of
every contour vertex (OpenCV then returns -1 without an on-edge match).

## Mask rasterization at the downsampled resolution

The reference overlap-mask stage renders every instance with `cv2.fillPoly` into
a zeroed full-resolution image, downsamples it 4× with `cv2.resize`, sums it
and composes the masks by area. For the default `mask_ratio=4` with sides
divisible by 4, `src/sampled.hpp` computes the same bytes without the
full-resolution image:

- At exactly 4×, `INTER_LINEAR` reads source rows 4k+1, 4k+2 and columns
  4j+1, 4j+2 for each output pixel with equal weights, so every output byte is
  a function of four source pixels. The 16-entry table of that function is
  calibrated against the installed `cv2.resize` for each output size. The
  calibration images show every pattern at every output pixel with random
  unread pixels, and random masks follow; any disagreement disables the path
  for that size. On the M2, OpenCV's Carotene path (both output sides ≥ 8)
  gives 1 when any sample is set, and its generic path when at least two are.
- `fillPoly` is integer arithmetic: 8-connected Bresenham outlines (with
  OpenCV's clipping), then spans between pairs of active edges sorted by 16.16
  fixed-point x. The kernel evaluates both only on sampled rows and columns. It
  computes each edge's crossing at the sampled rows it is active on directly as
  `x0 + (y - y0) * dx`, sorts each row's crossings and pairs them. That equals
  OpenCV's incremental active list, because a closed contour crosses every row
  an even number of times.
- A vertex with a strictly lower neighbour is not marked: it is the upper end of
  an edge whose crossing on its own row is the vertex itself, so the fill draws
  it.
- One native call does the rest:
  - it truncates float coordinates as `np.asarray(..., np.int32)` does;
  - it drops consecutive duplicate vertices (71% of resampled COCO vertices);
  - it computes the area sums and the composition.

  On arm64, NEON handles four vertices per step with table-driven compaction,
  the pattern table lookup (`tbl`) and the composition select. On x86-64, SSE2
  converts two vertices per step and does the span ORs and the composition,
  and SSSE3 `pshufb`, selected at run time, does the table lookup.
- Other inputs run the reference functions with the installed cv2 (see "No
  bundled OpenCV" below): coordinates outside the envelope where every
  intermediate is provably exact (|coordinate| ≥ 2^24, sides > 32,768),
  non-finite values, and other dtypes or memory layouts.

The results below are for 2,000 captured COCO Format calls (27,497 instances)
from `bench/mask_stage.py`. They are medians of five alternating rounds in one
process, and every output was compared with the unmodified function first.

| Backend | M2, µs / call | i5-10400 (Linux), µs / call | M2 speedup | i5-10400 speedup |
|---|---:|---:|---:|---:|
| Reference `polygons2masks_overlap` | 661 | 1,207 | 1.00× | 1.00× |
| Previous native path (full resolution) | 263 | 382 | 2.51× | 3.16× |
| Sampled native path | 50 | 88 | **13.2×** | **13.7×** |
| Sampled native path, restructured kernel (below) | — | 88.5 | — | **13.4×** |

- **M2 wall-clock:** measured at commit `490eed2` on a quiet host (1-minute load
  average 1.9). An earlier run at `4219950`, taken as another job was finishing
  (load 3.7–3.9), gave 1,054, 411 and 69 µs (15.3×); load slowed the Python
  reference more than the native paths.
- **M2 instructions:** retired instructions from `/usr/bin/time -l` do not
  depend on load. Per call they are 9.87 M for the reference, 2.94 M for the
  previous native path and 0.68 M for the sampled path: 14.5× and 4.3× fewer.
- **i5-10400:** measured at commit `54fc9f6` (SSE2/SSSE3 paths) on an otherwise
  idle host with a load average below 1. The same host's run before the x86
  SIMD paths gave 111 µs (11.0×).
- **Restructured kernel:** measured on the i5-10400 at the commit that removed
  OpenCV (load average 2.9). On x86-64 it equals the previous kernel; on the M2
  its phase timers showed about 10% less time, but that host is no longer quiet
  enough for a wall-clock figure, so its instruction count (0.72 M per call,
  13.6× fewer than the reference) is the number recorded.
- **Portability:** the Linux and Windows CI builds (GCC, MSVC) pass the same
  parity tests, and the calibration finds a table on all three CI platforms,
  so the sampled path is active everywhere.

Inside the loader, `Format._format_segments` also reorders the instances
(`instances[order]`, which copies every segment array). Both implementations
do this identically, and it now takes most of the stage's remaining time.

### Where the kernel's time goes, and how far SIMD reaches

A profiling build with phase timers (`-DMASKOPS_ABLATE`, `mach_absolute_time`
around each phase) gives this split of one call of the kernel on the M2 for the
2,000 captured Format calls (13.7 instances, 13,700 float vertices each; the
host was busy, so the absolute numbers are indicative):

| Phase | µs / call | What it does | How it is computed |
|---|---:|---|---|
| Load | 6.7 | float → int32, drop consecutive duplicates, pad each contour | NEON / SSE2, 4 vertices per step, table compaction |
| Bounds | 1.3 | vertex box, early exit for contours beside the image | NEON |
| Pass | 6.7 | per vertex: its mark, its edge's crossing when the edge is short and inside (or beside the image on one side), else a scalar index | NEON / SSSE3, 4 vertices per step, three compacted streams |
| Marks | 0.3 | OR the collected marks into the pattern image | scalar byte writes |
| Pushes | 0.8 | crossings into per-row buckets | scalar scatter |
| Other edges | 9.0 | the 15% of edges longer than one pixel, or crossing the border: LineIterator pixels and 16.16 crossings | scalar; a branchless variant for edges up to 5 px |
| Row fill | 9.8 | per output row: sort each slot's crossings, pair them, set column bit sets, expand 16 columns at a time | scalar spans, NEON / SSE2 expansion |
| Emit | 1.9 | pattern → table value, area sum, clear | NEON `tbl` / SSSE3 `pshufb` |

Per contour that is about 2.5 µs, or 8,500 cycles for 289 distinct vertices.
The vector phases (load, bounds, pass, emit) are at 1–5 cycles per vertex and
are bound by instruction count, not arithmetic. The rest is scatter: every
crossing lands in a bucket, every row is filled between its own crossings, and
every longer edge walks its own pixels. Those steps are data-dependent per
element; they can be made branchless (which is what the "other edges" path
does) but not lane-parallel without a different representation of the
problem. On these inputs the floor of this design is therefore around 60% of
its current time, and the whole kernel is already 13.6× fewer retired
instructions than the reference (0.72 M against 9.8 M per call on the M2).

Existing Rust rasterizers were evaluated as replacements for this replica on
4,224 real polygons and none reproduces `cv2.fillPoly`:

- imageproc 0.27 matched 75.8% of them and kornia-imgproc 73.3%. They differ
  on outline pixels and span-end rounding, including simple polygons.
- tiny-skia, raqote, vello_cpu, agg-rust and other coverage-based fillers
  matched at most 40%, because they do not draw OpenCV's LINE_8 outline.

### Verification of the vector code

Every NEON and SSE path has a scalar twin that computes the same bytes, and
`ULTRAFAST_MASKOPS_SCALAR=1` at import selects the twins everywhere
(`backend_info()["simd"]` reports `neon`, `sse2` or `scalar`). The CI runs the
parity tests in both modes on Linux, Windows and macOS, so the fallbacks are
covered on hosts that have SIMD. Beyond the unit tests, differential fuzzing
of the kernel against `cv2.fillPoly` (random sizes up to 1,024, every contour
family of `tests/test_sampled.py`, whole overlap and mask outputs every eighth
case) ran for 45 CPU-minutes on the M2 and 30 on the i5-10400 without a
difference, and a full COCO-scale epoch (7,393 batches) produced identical
batch digests for the reference, the accelerated and the mask-only pipelines.

### No bundled OpenCV

The extension links nothing but pybind11 (0.27 MB on arm64, 0.31 MB on x86-64;
a C++20 compiler builds it in seconds). It bundles no OpenCV because the mask
kernel replicates `fillPoly` and the 4× downscale in integer arithmetic. cv2
still matters: the kernel is verified against the cv2 that is installed, per
image size, the first time that size is used (the resize pattern table plus
24 probe polygons through the whole native path), and `backend_info()` lists
the sizes it accelerates and the sizes it left alone. Inputs outside the native
profile (`mask_ratio` other than 4, sides not divisible by 4, colors other than
1, several contours in one mask, coordinates outside the exact envelope, a cv2
the calibration rejects) run the unmodified reference functions with that
cv2, so results never depend on which path ran. The adapter no longer pins
NumPy and cv2 to exact versions; it requires NumPy 2 and checks the hashes of
the Ultralytics functions it replaces.

## Reproduce

```sh
python bench/geometry_loader.py --images /path/to/coco/segment/images/val2017 \
  --samples 1000 --rounds 3 --out geometry-loader.json
```

Use `accelerate_dataset(ds)` then `accelerate_geometry(ds, persistent=True)` in a
custom trainer's dataset builder; neither patches Ultralytics globally.
