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

## Verification checkpoint — 2026-09-13

- The complete suite passed 135 tests on Apple M2 / CPython 3.12.13 and Linux
  i5-10400 / CPython 3.12.14. The Linux JUnit report is retained in
  `validation/coco-bounds-server.xml`.
- Coverage includes 10,000 seeded differential cases, repeated vertices,
  degenerate contours, holes, 255/256 instance dtype boundaries, reused dirty
  scratch buffers, unaligned input and actual DataLoader integration.
- Both hosts are measuring all 5,000 COCO images at workers 0/2/8 with five
  alternating fresh processes per backend. No final performance conclusion is
  available at this checkpoint. Earlier baseline results remain in the sibling
  `feat/native-core` branch, including unfavorable measurements.
- ASan/UBSan for these new changes, final full-corpus fresh-reference verification,
  augmented training targets, GPU epochs and wheel validation remain open. This
  candidate is not release-ready.
