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

## Reproduce

```sh
python bench/geometry_loader.py --images /path/to/coco/segment/images/val2017 \
  --samples 1000 --rounds 3 --out geometry-loader.json
```

Use `accelerate_dataset(ds)` then `accelerate_geometry(ds, persistent=True)` in a
custom trainer's dataset builder; neither patches Ultralytics globally.
