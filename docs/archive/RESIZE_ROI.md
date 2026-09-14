# Experimental resize region candidate

This worktree is based on the tested masks-only implementation. The first fresh
wheel built and installed, but failed 14 of 179 parity/integration/auditor tests
(165 passed). That candidate is rejected. A correction preserving the Arm resize
dispatch threshold passed all 191 tests in a fresh M2 wheel installation. The
[Linux paired comparison](PAIRED_WHEEL_RESULTS.md) shows ratio-4 public-call gains,
a ratio-1 non-overlap regression and less than 1 MiB RSS differences; M2 timing
and isolated working allocation remain unmeasured for this candidate. The server GPU series
uses its original installed wheels; earlier results are not candidate evidence.

## Linux installed-wheel validation

The corrected candidate also built as a CPython 3.12 `linux_x86_64` wheel using
GNU 13.3.0 and private OpenCV 4.13.0. A fresh Python 3.12.14 environment first
installed only NumPy 2.4.4 and the wheel: standalone output checks, wheel RECORD,
installed bytes, and all nine bundled notices against both wheel and sdist passed.
The subsequent pinned Ultralytics/Torch environment passed all **201 tests**,
with no failures, errors or skips (pytest wall time 20.52 seconds). This includes
the 191 prior checks and ten new no-copy output-hashing checks. It is separate
Linux evidence; those ten new hashing checks have not yet run on M2.

The extension reports the exact current binding/CMake/template hashes, including
binding SHA `c63940b67b291123d6d5d895858cb981afaa17bd1aea94856cd9f40dcebb1813`.
The wheel SHA is
`1b08974c11b1bc00e7e3f85f68fb6f8505e9e2edc5ca55b941ca6ef09b789fad`.
The [preserved archive](../bench/results/resize-roi-linux-v1.tar.gz) contains the
source archive, wheel/sdist, build and installation logs, environment freeze,
installed-wheel audit, test logs/XML and source/binary receipt. Its SHA is
`5e80166deb3a07aa5f64df638061d6f8b1ca0280d4248cd3096b3230453267b3`;
server/local copies and the recorded artifact hashes match. See also
[the receipt](validation/resize-roi-linux-passed-v1.json) and
[test report](validation/resize-roi-linux-tests-v1.xml).

An initial offline dependency install could not resolve the uncached pinned
PyTorch URL. Its log is retained; installation succeeded with the same frozen
requirements after allowing downloads. No version was changed to pass tests.
This wheel has not been manylinux-repaired. The full real-corpus overlap and
non-overlap checks below passed; sanitizer, M2 and end-to-end performance checks remain outstanding.
Neither Linux parity nor the earlier GPU results establish ROI speed or memory benefits.

## Linux full augmented overlap and non-overlap verification

The corrected installed wheel passed nine fresh-process runs on all 5,000 COCO
val2017 images: reference, reference replay and native for each of workers 0/2/8.
Each run produced 625 batches at batch size 8, image size 640 and mask ratio 4.
Mosaic, mixup, copy-paste and the other fixed stress augmentations were enabled
with seed 912. A separate non-overlap series then passed the same nine-process
matrix, for 18 completed fresh-process runs in total. Only mask preparation was replaced; the original dataset scanner
was retained. All batch and whole-stream output hashes match within each worker
group, including images, masks, semantic masks, boxes, classes and instance order.

| Loader workers | Augmented instances per run | Maximum instances in an image | Mask dtype |
| --- | ---: | ---: | --- |
| 0 | 225,016 | 222 | uint8 |
| 2 | 222,594 | 227 | uint8 |
| 8 | 220,191 | 184 | uint8 |

The instance counts and dtypes above apply to each mode. Different worker counts produce different random streams; equality is checked
against the two corresponding reference runs, not across worker counts.
An independent standard-library audit checked all nine raw files against the
parent report, all 625 batch records per run, source and compiled-profile hashes,
the tested extension, package versions, augmentation settings and original cache
provenance. This audit does not rerun the corpus; the retained worker source
performs pre/post corpus and cache fingerprint checks during each actual run.

The [32-file evidence archive](../bench/results/resize-roi-linux-overlap-v1.tar.gz)
contains all nine raw results and logs, the complete parent report, source files,
original cache provenance and installed-wheel receipt. Its SHA-256 is
`402ed199a3ebd0c559012617aba1b6ec0ce64da41f085343f707484fc694bf95`.
See the [audit receipt](validation/resize-roi-linux-overlap-audit-v1.json) and
[reproduction script](validation/audit-resize-roi-augmented-v1.py). Run the latter
with the archive path as its single argument. Hashing is inside loader iteration,
so these runs establish output parity only, not timing or memory improvement.

The separately [preserved non-overlap archive](../bench/results/resize-roi-linux-nonoverlap-v1.tar.gz)
also contains 32 files, including all nine raw results and logs. Its SHA-256 is
`58edf832e37908b272f1099847fc6ebdb653c47f1c1858a1e9ee96fe9ece273f`.
The same auditor passed all nine non-overlap runs; see its
[audit receipt](validation/resize-roi-linux-nonoverlap-audit-v1.json).
Both series used the same tested installed extension and unchanged source,
upstream packages and original cache. Each mode is compared against its own
reference streams; overlap and per-instance masks are not expected to share
output hashes. The sequential server command finished with exit code zero.

## Change and candidate invariants

Polygon filling still uses the complete raster scratch image, original integer
coordinates and unchanged contour order. After filling, the cached inclusive
polygon extents give a conservative rectangle containing every possible nonzero
pixel. The candidate aligns that rectangle outward to the source sampling
blocks, extends it to preserve the 8×8 HAL eligibility threshold when applicable,
resizes only that crop into its corresponding output rectangle, and
sets the rest of the output to zero. Overlap area reduction visits only the
returned support rectangle. Instance ordering still uses the existing unsigned
area vector and NumPy sort.

Let the effective scale be `s`, with source coordinates mapped from destination
coordinate `d` by `(d + 0.5)*s - 0.5`. For integer power-of-two downsampling,
the two interpolation neighbors lie inside the associated source block of
width `s`; scale 1 is a direct copy. Translating a crop origin by `k*s` translates
the destination by `k`. Both resizes have the same exactly representable
reciprocal scale. This is the sampling-lattice argument, not a substitute for
testing each actual OpenCV dispatch path.

Eligibility requires the effective horizontal and vertical scales to be the
same integer power of two. Both source dimensions must be at most `2**23`, so
the generic OpenCV resize implementation can represent the required half-integer
coordinates exactly in its intermediate `float`. Other scales use the existing
full resize. This concerns the effective source/output ratio; an API ratio that
truncates output dimensions can produce a different effective ratio.

An empty support rectangle produces zeros. A fixed initial cost guard retains
full resize when the aligned output crop covers at least half of the output:
zeroing the output and then rewriting a large crop could otherwise regress.
This threshold is declared before candidate measurements and has not been tuned.

## Dispatch and boundary review

The inspected pinned OpenCV 4.13.0 source switches equal 2× linear downsampling
to its fast area path. Its single-channel scalar tail and vector kernels use
the same integer rounding form for each 2×2 block. In the macOS build, the
KleidiCV 0.7.0 adapter can instead handle this operation above 150,000 source
pixels; a crop can cross that dispatch threshold. KleidiCV falls back to the
previously defined Carotene HAL, then generic OpenCV. Carotene requires at least
eight output pixels along each axis for single-channel linear resize. Below that
threshold the generic path can round differently at scales 4/8/16. The initial
candidate exposed these differences: mask bytes and overlap area order changed.
The failures and exact binary identity are preserved in
`validation/resize-roi-failed-v1.json` and `resize-roi-tests-v1.*`.

The correction extends small crops to at least 8×8 output pixels whenever the
full output meets that threshold. Extension stays inside the image and on the
sampling lattice; it includes only additional zero-valued scratch outside the
support. The original half-output cost guard still applies after extension.
This preserves eligibility rather than replacing the reference's rounding.

`tests/test_resize_roi.py` now contains 40 checks covering crop widths around SIMD
boundaries, unaligned origins, thin shapes, clipping, colors 0/1/127/255,
retained/bounded modes, zero/equal areas with more than 255 instances, combined
contours, noninteger/other integer scales, the half-scale Arm dispatch threshold
and coordinates above the float guard, including 12 added combinations spanning
the Carotene width/height threshold and image edges. The corrected wheel's full
suite passed in 95.90 seconds: 179 native/parity/integration checks and 12 artifact
auditor checks, with no failures, errors or skips. This includes 10,000 seeded
differential cases and deterministic CPU model updates. The NumPy-only installed
wheel smoke, RECORD/runtime audit and all nine bundled notice files also passed.
Exact wheel/source hashes and reports are in `validation/resize-roi-passed-v2.json`
and `resize-roi-tests-v2.*`. Linux full real-corpus output validation is recorded
above. Sanitizer, M2 and end-to-end benchmark checks remain necessary before merging.

The first build's 573 selected dependency files, including 61 Carotene files,
match the existing notice inventory. Their collected notices are present in the
wheel; no missing Carotene notices were found. Selected HAL/CPU/build options
match the earlier masks-only build. This evidence and the compressed compiler
graph are retained in `validation/resize-roi-dependency-audit-v1.json` and
`resize-roi-compiled-deps-v1.txt.gz`; this is not a complete redistribution audit.

The corrected Linux build's dependency graph was checked separately after its
201-test run. All 375 selected OpenCV/pybind11 source and header files match the
existing notice inventory by byte count and SHA-256, with no missing or changed
entries. The inventory's provenance hash and bundled notice hashes also match.
The graph's build extension matches the tested installed extension
(`9b86c9dc…d78325`). The receipt, compressed graph and read-only audit script are
retained as `validation/resize-roi-linux-dependency-audit-v1.json`,
`resize-roi-linux-compiled-deps-v1.txt.gz` and
`audit-resize-roi-linux-dependencies-v1.py`. Toolchain, system, Python and generated
headers remain outside this selected-source coverage check.

## Performance and memory scope

The candidate aims to reduce resize coefficient/row-buffer work and area-scan
traffic for small polygons. Every public per-instance mask still needs its full
output shape and defined zero values. Full-resolution raster scratch remains
allocated and filling still uses that image, so this is **not** a claim of
polygon-sized raster scratch or lower process RSS. Actual allocation and timing
measurements must accompany any result.

The [paired wheel protocol](PAIRED_WHEELS.md) completed the Linux counterbalanced
comparison against the tested masks-only baseline across the frozen nine cases
and three public call paths. All 270 measured processes, separate oracle checks
and 54 independently recomputed timing/RSS summaries passed the artifact audit.
The [complete results](PAIRED_WHEEL_RESULTS.md) retain the ratio-1 non-overlap
regression along with the ratio-4 gains. M2 remains unmeasured while its 500k
dataset experiment runs; isolated working-allocation and real-loader timing
still need their own measurements.

After this candidate's correctness and cost are established, a separate step can
consider cropping the raster scratch itself. That needs a distinct proof of
integer translation, clipping and fill-edge ordering, plus new memory-budget
and lifetime checks. It must not be inferred from resize-crop parity alone.

Source-review hashes are in `validation/resize-roi-source-review.json`. They
identify the inspected implementation files and do not certify this candidate.
