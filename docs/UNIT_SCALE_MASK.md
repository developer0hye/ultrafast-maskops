# Experimental unit-scale mask-only copy path

The frozen Linux ROI comparison found a 3.7% regression for public non-overlap
masks at 640×640, 100 instances and ratio 1. All measurements and the prior
binary remain preserved in PAIRED_WHEEL_RESULTS.md. This new candidate is not
built, tested or measured yet.

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
