# Experimental resize region candidate

This worktree is based on the tested masks-only implementation. The changes in
this document are **not built, executed or benchmarked yet**. The ongoing main
worktree augmentation verification and the server GPU series use their original
installed wheels. No earlier result is evidence for this candidate.

## Change and candidate invariants

Polygon filling still uses the complete raster scratch image, original integer
coordinates and unchanged contour order. After filling, the cached inclusive
polygon extents give a conservative rectangle containing every possible nonzero
pixel. The candidate aligns that rectangle outward to the source sampling
blocks, resizes only that crop into its corresponding output rectangle, and
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
pixels; a crop can cross that dispatch threshold. Those paths still require
actual bitwise comparison. Generic downsampling at larger ratios also needs
SIMD-tail and row-stride coverage.

`tests/test_resize_roi.py` prepares 28 checks covering crop widths around SIMD
boundaries, unaligned origins, thin shapes, clipping, colors 0/1/127/255,
retained/bounded modes, zero/equal areas with more than 255 instances, combined
contours, noninteger/other integer scales, the half-scale Arm dispatch threshold
and coordinates above the float guard. These tests are not yet run. Full core,
framework/model-update parity, sanitizer, real-corpus output and benchmark checks
remain necessary before merging.

## Performance and memory scope

The candidate aims to reduce resize coefficient/row-buffer work and area-scan
traffic for small polygons. Every public per-instance mask still needs its full
output shape and defined zero values. Full-resolution raster scratch remains
allocated and filling still uses that image, so this is **not** a claim of
polygon-sized raster scratch or lower process RSS. Actual allocation and timing
measurements must accompany any result.

After this candidate's correctness and cost are established, a separate step can
consider cropping the raster scratch itself. That needs a distinct proof of
integer translation, clipping and fill-edge ordering, plus new memory-budget
and lifetime checks. It must not be inferred from resize-crop parity alone.

Source-review hashes are in `validation/resize-roi-source-review.json`. They
identify the inspected implementation files and do not certify this candidate.
