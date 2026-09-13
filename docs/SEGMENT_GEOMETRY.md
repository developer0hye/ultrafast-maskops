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

1,000 samples per process, three alternating rounds, identical seeds; the full
output digest (image, classes, boxes, masks, metadata) is identical for every
backend and round.

| Backend | ms / sample | Loader speedup |
|---|---:|---:|
| Reference | 10.33 | 1.00× |
| Masks only (`accelerate_dataset`) | 9.68 | 1.07× |
| Geometry + masks (`accelerate_dataset` + `accelerate_geometry`) | **7.35** | **1.40×** |

These loader results predate the sampled mask path described below.

Per 1,000 samples, resampling fell from 0.82 s to 0.18 s (4.5×) and
`apply_segments` from 2.23 s to 0.61 s (3.7×); mask rasterization takes about
0.26 s instead of 0.69 s (2.5×). The segment stage as a whole is about 3.6×
faster. Removing it entirely would bound this loader at 10.33 / 6.60 = 1.57×;
the rest is JPEG decoding, `warpAffine`, HSV conversion and file reads, which
already run in OpenCV.

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

  NEON handles four vertices per step with table-driven compaction, the
  pattern table lookup (`tbl`) and the composition select.
- Some inputs use the previous full-resolution path, which keeps its exact
  error behaviour: coordinates outside the envelope where every intermediate is
  provably exact (|coordinate| ≥ 2^24, sides > 32,768), non-finite values, and
  other dtypes or memory layouts.

The results below are for 2,000 captured COCO Format calls (27,497 instances)
from `bench/mask_stage.py`. They are medians of five alternating rounds in one
process, and every output was compared with the unmodified function first.

| Backend | µs / call | Instructions / call | Speedup |
|---|---:|---:|---:|
| Reference `polygons2masks_overlap` | 1,054 | 9.87 M | 1.00× |
| Previous native path (full resolution) | 411 | 2.94 M | 2.57× |
| Sampled native path | 69 | 0.68 M | **15.3×** |

- **Wall-clock:** the run started as another job was finishing (1-minute load
  average 3.7–3.9). That inflates the absolute times of every backend in the
  alternating rounds. With a quiet host, the reference measured about 620 µs.
- **Instructions:** retired instructions from `/usr/bin/time -l` do not depend on
  load. They give 14.5× against the reference and 4.3× against the previous
  native path.
- **x86-64:** the Linux and Windows CI builds (MSVC, scalar fallbacks) pass the
  same parity tests. The calibration finds a table there too, so the sampled
  path is active on all three CI platforms.

Inside the loader, `Format._format_segments` also reorders the instances
(`instances[order]`, which copies every segment array). Both implementations
do this identically, and it now takes most of the stage's remaining time.

Existing Rust rasterizers were evaluated as replacements for this replica on
4,224 real polygons and none reproduces `cv2.fillPoly`:

- imageproc 0.27 matched 75.8% of them and kornia-imgproc 73.3%. They differ
  on outline pixels and span-end rounding, including simple polygons.
- tiny-skia, raqote, vello_cpu, agg-rust and other coverage-based fillers
  matched at most 40%, because they do not draw OpenCV's LINE_8 outline.

## Reproduce

```sh
python bench/geometry_loader.py --images /path/to/coco/segment/images/val2017 \
  --samples 1000 --rounds 3 --out geometry-loader.json
```

Use `accelerate_dataset(ds)` then `accelerate_geometry(ds, persistent=True)` in a
custom trainer's dataset builder; neither patches Ultralytics globally.
