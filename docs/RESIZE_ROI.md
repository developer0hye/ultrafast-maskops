# Experimental resize region candidate

This worktree is based on the tested masks-only implementation. The first fresh
wheel built and installed, but failed 14 of 179 parity/integration/auditor tests
(165 passed). That candidate is rejected. A correction preserving the Arm resize
dispatch threshold passed all 191 tests in a fresh M2 wheel installation. No timing
or memory benefit has been measured for either candidate. The server GPU series
uses its original installed wheels; earlier results are not candidate evidence.

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
and `resize-roi-tests-v2.*`. Sanitizer, real-corpus output, Linux and benchmark
checks remain necessary before merging.

The first build's 573 selected dependency files, including 61 Carotene files,
match the existing notice inventory. Their collected notices are present in the
wheel; no missing Carotene notices were found. Selected HAL/CPU/build options
match the earlier masks-only build. This evidence and the compressed compiler
graph are retained in `validation/resize-roi-dependency-audit-v1.json` and
`resize-roi-compiled-deps-v1.txt.gz`; this is not a complete redistribution audit.

## Performance and memory scope

The candidate aims to reduce resize coefficient/row-buffer work and area-scan
traffic for small polygons. Every public per-instance mask still needs its full
output shape and defined zero values. Full-resolution raster scratch remains
allocated and filling still uses that image, so this is **not** a claim of
polygon-sized raster scratch or lower process RSS. Actual allocation and timing
measurements must accompany any result.

The [paired wheel protocol](PAIRED_WHEELS.md) now prepares a counterbalanced
comparison with the tested masks-only baseline across the frozen nine cases and
three public call paths. Both installed-wheel descriptors passed byte/source and
shared-profile checks. Actual oracle, timing, RSS and aggregate execution remain
pending; no benchmark is queued while the host runs the 500k dataset experiment.

After this candidate's correctness and cost are established, a separate step can
consider cropping the raster scratch itself. That needs a distinct proof of
integer translation, clipping and fill-edge ordering, plus new memory-budget
and lifetime checks. It must not be inferred from resize-crop parity alone.

Source-review hashes are in `validation/resize-roi-source-review.json`. They
identify the inspected implementation files and do not certify this candidate.
