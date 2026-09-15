# How much faster can the kernels still get?

A review of every native path (Ultralytics mask rasterization, Ultralytics
segment geometry, RF-DETR deferred rasterization) against its theoretical floor
and against what C++20 and the target ISAs allow, with measurements. Numbers are
per training sample unless stated; "sample" is one augmented `__getitem__`.

The short version: after this review the mask kernels are within 1–3% of
their sample's total time, and the Ultralytics geometry kernels at about 12%
with perhaps a third of that recoverable. The remaining time in both
frameworks is image work (JPEG decode, resize, colour conversion,
normalization) that these kernels do not touch.

## Ultralytics: mask rasterization (`src/sampled.hpp`)

Per `Format` call (one per sample), M2, 43 captured COCO calls with 15.4
instances of 1,000 vertices each, minimum of 15 repeats (the host was busy;
relative shares are what matter):

| Part | µs | Share |
|---|---:|---:|
| `sampled_raster` (native; load, bounds, pass, marks, pushes, long edges, row fill, emit) | 33.9 | 81% |
| `sampled_compose` (native: zero the 160×160 output, paint each mask in rank order) | 5.2 | 12% |
| `np.argsort(-areas)` + `astype` (kept in NumPy so tie order matches the reference's introsort) | 1.1 | 3% |
| pybind11 call overhead (measured with empty input) | 0.7 | 2% |
| pattern-table lookup, Python glue | 1.6 | 4% |
| **total** | **41.8** | |

Inside the native raster the phase split recorded in
[SEGMENT_GEOMETRY.md](SEGMENT_GEOMETRY.md#where-the-kernels-time-goes-and-how-far-simd-reaches)
holds: about 45% is vector code at 1–5 cycles per vertex (load, bounds, pass,
emit), about 55% is data-dependent scatter (long edges 9 µs, row fill 10 µs,
pushes and marks 1 µs).

What is left, and what it would buy:

- **Row fill, common case (≈10 µs).** Most sampled rows hold exactly one span
  per half-row. They still go through the general path: sort, column bit sets,
  a 16-column expansion with four scalar shifts per group, and the clearing of
  the bit sets. A direct two-span path (compare lane indices against the four
  span bounds, one load-OR-store per 16 columns) would cut this to about 4 µs.
- **Long edges (≈9 µs).** The branchless handler for edges up to 5 px does
  about 30 scalar operations and 7 read-modify-writes per edge. The arithmetic
  for four edges fits in NEON/SSE lanes; the scatters do not. Perhaps 3 µs.
- **AVX2 on x86-64.** The x86 paths are SSE2/SSSE3 so a manylinux wheel runs
  everywhere; an AVX2 twin of pass, load, emit and expand_row behind the same
  runtime dispatch already used for SSSE3 would roughly halve those phases on
  the i5 (about 8 of 89 µs).
- **Fusing load and bounds** saves one pass over the vertices, about 1 µs.

All of it together is about 15 µs per sample: 0.2% of an 8.7 ms sample on the
i5, 0.1% on the M2. The kernel's floor for this design (scatter-bound) was
already estimated at about 60% of its current time; reaching it would not be
visible in training. Not done.

Language notes. The kernel relies on C++20 guarantees that were
implementation-defined before: two's complement integers and arithmetic right
shift of negative values (`y >> 2` for rows above the image), `std::bit_cast`
for the vertex-key trick, `std::countr_zero`, `constexpr` tables built at
compile time, `[[likely]]` on the short-edge branch. The allocator
`DefaultInit` keeps `std::vector` from zero-filling scratch. pybind11 costs
0.7 µs per call and the GIL is released around every kernel.

## Ultralytics: segment geometry (`src/geometry.hpp`)

Per sample on the i5 (from the loader-stage measurements, accelerated side):
`resample_segments` 0.39 ms, `apply_segments` 0.56 ms, `_format_segments`
0.14 ms, of an 8.7 ms sample. The kernels themselves, on a COCO-like sample
(14 polygons of 40–400 vertices resampled to 1,000 points), M2, minimum of
300 repeats:

| Kernel | µs per sample |
|---|---:|
| `resample_stack` (27,000 `np.interp` evaluations) | 58 |
| `project_boxes_f32` (27,000 divisions, boxes, clip) | 57 |
| `segment_boxes_f32` (boxes, clip) | 56 |

So about 0.17 ms of the 1.1 ms is native; the rest is what surrounds the
kernels in Python: packing the list of per-instance arrays into points and
offsets, `np.concatenate`, the NumPy matrix product that `apply_segments`
keeps (a BLAS's summation order is not reproducible from C++), and
Ultralytics' own `Instances` bookkeeping. Levers, in order of size:

- Keep segments packed between stages instead of re-packing per stage
  (the shape ultrafast-yolo-dataset already produces): perhaps 0.2–0.3 ms.
- `resample_one`: the division `(y1 - y0) / (x1 - x0)` is by exactly 1.0 and
  can go; the query positions are monotone, so a 2-wide (NEON) or 4-wide
  (AVX2) `np.interp` is exact and straightforward. `project_boxes`: SIMD
  division is correctly rounded, so a `divps` version is exact. Together
  under 0.1 ms.
- `segment_box` and the clip are min/max loops the compiler already
  vectorizes with compare-and-select.

At most 0.3–0.4 ms per sample, 4% on the i5, most of it in Python rather
than in the kernels. Not done in this pass; the packed-segments change is the
one worth making if the loader is ever the bottleneck again.

## RF-DETR: deferred rasterization (`src/rfdetr.hpp`)

Per materialize step, M2, seven instances (690 vertices, 2.2 MB of output,
5.6% filled), minimum of 200 repeats:

| Version | Kernel | Of which allocate + zero output | `index_maps` (NumPy) |
|---|---:|---:|---:|
| first commit (`195d3e1`) | 299 µs | 19 µs | 18 µs |
| after this review | ~150 µs | 19 µs | 18 µs |

On the i5, per real COCO sample, the materialize step went from 0.34 to
0.23 ms; the sample from 10.26 to 10.20 ms.

A standalone harness split the first version into the toggle walk (181 µs,
3.6 ns per dense step) and the render (273 µs, most of it a `std::sort` of
about 700 toggles per polygon). What changed, each verified against
pycocotools by the 4,000-seed fuzz on x86-64 and by a 200,000-polygon
differential fuzz of the new walk against the old:

- The C code tests `((u + .5) / 5 - .5)` for integrality and takes the `ceil`
  of a similar expression for the row. Both are exact rationals with fractional
  parts in fifths, so they are integer tests: a column toggles exactly when its
  adjusted index is `5n + 2`, and the ceil is a floor division. No division per
  step.
- Along an x-major edge the column advances by one per step, so only every
  fifth step can toggle; the walk evaluates those steps and their predecessors
  with the C formula and skips the rest.
- Along a steep edge (`dy >= 4 dx`) the column is a monotone function of the
  step (one correctly rounded multiply, two adds, a truncation, all monotone;
  a magnitude guard keeps `INT_MIN` out), so the next column change is found
  by galloping and bisection instead of one step at a time.
- Toggles are bucketed by row with a counting sort (the order within a row
  does not matter, XOR commutes); rows with no set bit are skipped; the column
  scan is bounded to the words that can hold bits; repeated output rows copy
  only the painted span.

Floor: writing the instance bytes. Zeroing 2.2 MB and painting 5.6% of it is
about 25 µs; the walk's remaining dense steps (diagonal y-major edges) are
another 40 µs. So the kernel could still drop by about half, to 70–80 µs,
which is 0.7% of a 10 ms sample. Two levers were left unpulled:

- The y-major dense loop could evaluate two (NEON) or four (AVX2) steps per
  vector; exactness holds as long as the product and the sums are not fused,
  which the intrinsics guarantee.
- The output could stay untouched where it stays zero (a `calloc`-backed
  array maps zero pages lazily), but RF-DETR's collate copies every mask
  tensor anyway, so the pages are touched a moment later.

Language notes. `-ffp-contract=off` is what makes the double arithmetic
match pycocotools' x86-64 build; the arm64 macOS wheel of pycocotools
contracts `ys + s*t + .5` and differs from the C source's unfused result by
one pixel in one of 4,000 seeds on the M2, which an unfused pure-Python port
of `rleFrPoly` confirms. Integer conversions emulate x86-64's `cvttsd2si`
(NaN and overflow to `INT_MIN`) explicitly, because arm64 saturates and
returns 0 for NaN. `std::countr_zero` drives the run scan.

## Beyond the mask paths

Where the samples' time actually is after this work:

| Framework | Sample | Mask paths | Rest |
|---|---:|---:|---|
| Ultralytics (i5, augmented COCO) | 8.7 ms | ≈1.1 ms (geometry 0.95, masks 0.14) | `imdecode`, `warpAffine`, HSV `cvtColor`/`LUT`, mosaic copies (OpenCV C++) |
| RF-DETR (i5, default train transforms) | ≈10 ms | ≈0.5 ms | PIL JPEG decode 2.2 ms, PIL bilinear resize ≈4.6 ms, `ToImage`/`ToDtype`/`Normalize` ≈2 ms |

Anything larger now has to come from the image path. Two options that keep
outputs bit-identical are `ToDtype` + `Normalize` fused into one uint8 →
normalized-float pass (RF-DETR's two passes over 3×560×560 floats) and a
faster JPEG decoder with identical output (libjpeg-turbo is what PIL already
uses, so little is left there). One option that does not keep outputs
identical is DCT-domain downscaling at decode (`PIL.Image.draft`), which
RF-DETR supports for detection but refuses for mask datasets; with deferred
rasterization the masks could follow the drafted polygons, but the result
would be RF-DETR's draft pipeline, not its full-resolution one.
