# Experimental unit-scale mask-only copy path

The frozen Linux ROI comparison found a 3.7% regression for public non-overlap
masks at 640×640, 100 instances and ratio 1. All measurements and the prior
binary remain preserved in PAIRED_WHEEL_RESULTS.md. The new Linux wheel passed all 209 tests in 20.34 s in a fresh
environment, including the eight added mixed-caller cases. Two separate full
270-process comparisons are complete and independently audited; see
[UNIT_SCALE_RESULTS.md](UNIT_SCALE_RESULTS.md). The candidate does not yet
establish that the ratio-1 regression is fixed.

Mask-only calls do not use the support rectangle returned for area summation.
When source and destination dimensions match, they now call the original full
`cv::resize` path instead of clearing the destination and resizing a crop into
it. This covers batched masks and the single-mask combined-contour operation.
Area-producing rasterization and overlap composition retain their prior path;
downsampled mask calls still use the corrected ROI logic. Scratch clearing,
fill coordinates/order, output ownership and the public API are unchanged.

This is a performance hypothesis, not an attribution of the entire measured
regression to crop overhead. A new wheel must pass the full suite, including
eight mixed-caller scratch/ownership cases across unit and downsampled scales,
and fresh-process paired measurements against the frozen ROI and masks-only
wheels. No result from the earlier binary validates this change automatically.

## Installed Linux validation

Source commit `9c4ece56f8a15a251c36a07a0e02af238a7df81b` produced wheel
`2ca1875910cec38fd501af9108455609b6c926c8d14dcf570da7b9776a2e0c71`,
with extension `9da4432a4b3fe7c753568d82780886812360783bfbbf33e8727e1dde94f49cf2`
and kernel `da270a9525d9c64d1e2d802cb8ec44cbe5743ed76ff6087936a9b00466c1f4a4`.
Python 3.12.14, NumPy 2.4.4 and GNU 13.3.0 Release match the Linux comparison
baselines. All other frozen installed packages and private OpenCV build settings
match; OpenCV build timestamp and temporary installation prefix differ.

The initial NumPy-only installation passed wheel RECORD, installed-byte and
nine bundled-notice checks. After installing the frozen framework dependencies,
`pip check` passed and all 209 tests passed without failures, errors or skips.
The [receipt](validation/mask-unit-scale-linux-passed-v1.json) records the
source archive, seven test-source hashes and artifact identities; all seven
source files also match this worktree. The
[JUnit output](validation/mask-unit-scale-linux-tests-v1.xml),
[test log](validation/mask-unit-scale-linux-tests-v1.log) and
[installed descriptor](validation/mask-unit-scale-linux-descriptor-v1.json)
are retained. This is Linux correctness evidence; sanitizer, M2, portability,
working-allocation and real-loader performance gates remain open.
