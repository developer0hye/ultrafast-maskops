# Test fixtures

## `coco-format-inputs.npz`

43 calls of `ultralytics.data.augment.Format._format_segments` (664 instances)
captured from the augmented COCO val2017 segmentation pipeline: the float32
`(N, 1000, 2)` segment arrays exactly as `polygons2masks_overlap` receives
them after mosaic, random perspective and resampling, with their image sizes
and mask ratios. They were chosen from a 2,000-call capture: the two calls
with the most instances (67), the one with the fewest, and 40 at random
(`source_index` records their positions in the capture).

The polygons derive from the COCO 2017 annotations
(https://cocodataset.org, Creative Commons Attribution 4.0). The images are
not included.

Keys: `points` float32 `[P, 2]`, `offsets` int64 `[44]`, `instances`,
`vertices`, `hw`, `ratio`, `overlap`, `source_index`.
