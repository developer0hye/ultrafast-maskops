# RF-DETR: deferred mask rasterization

RF-DETR's segmentation datasets (`rfdetr.datasets.coco.CocoDetection` with
`include_masks=True`) spend most of a training sample's CPU time on masks.
`ConvertCoco` rasterizes every polygon at full resolution with pycocotools,
then every `Resize`, `RandomSizedCrop` and `RandomHorizontalFlip` resamples the
whole `(N, H, W)` mask tensor. The adapter in `ultrafast_maskops.rfdetr` keeps
the polygons instead, records what the random transforms did, and rasterizes
once, only at the output pixels. The result is the same bytes RF-DETR produces.

## Profile of the unmodified pipeline

Intel i5-10400, RF-DETR develop `2398ce3c` (1.11.0.dev0), torch 2.10 CPU,
COCO val2017, default training transforms (`make_coco_transforms_square_div_64`,
multi-scale, resolution 560), single process, 400 samples (6.3 instances each):

| Stage | Time per sample | Share | Mask-related part |
|---|---:|---:|---|
| JPEG decode (PIL) | 2.4 ms | 11% | — |
| `ConvertCoco` (prepare) | 5.3 ms | 25% | `convert_coco_poly_to_mask`: pycocotools `frPyObjects` + `decode` at full resolution, `any` over polygons, `[keep]`, `.bool()` |
| transforms | 13.3 ms | 63% | nearest resize of the mask tensor (once or twice), crop, flip: about 5.9 ms; the rest is the PIL image resize, `ToImage`/`ToDtype`/`Normalize` |
| **total** | **21.0 ms** | | about 11 ms of mask work |

## What the adapter does

`accelerate_dataset(dataset)` replaces two attributes of the dataset in place:

- `dataset.prepare` becomes `FastConvertCoco`. It runs RF-DETR's own
  `ConvertCoco` with `include_masks=False` for boxes, labels, areas, keypoints
  and sizes, recomputes the instance filter the way `ConvertCoco` does, and
  stores the kept instances' polygons as float64 arrays under
  `target["maskops_polygons"]` together with an empty `Trace`. Samples whose
  segmentations pycocotools would not read as polygons (RLE dicts, a four-number
  first entry, missing `segmentation` on the first annotation) go through the
  original `ConvertCoco` unchanged.
- `dataset._transforms` becomes a copy of the transform tree whose geometric
  leaves are traced subclasses: `TracedResize`, `TracedRandomResize`,
  `TracedRandomSizedCrop` and `TracedRandomHorizontalFlip` call the RF-DETR
  implementation (same random draws, same image, boxes and keypoints) and
  append `("resize", old, new)`, `("crop", top, left, h, w)` or `("hflip",)` to
  the trace. `ToImage`, `ToDtype` and `Normalize` pass through; any other
  transform type raises, since its effect on masks would be unknown.
  `MaterializeMasks` is appended last: it composes the trace into two index maps
  (output row → source row, output column → source column, −1 where a crop
  padded) and calls the native kernel.

`_filter_per_instance_fields` in RF-DETR's `crop` filters every list whose
length equals the number of boxes, so the polygon list follows the boxes
through crops for free; the `Trace` is not a sequence and is left alone.
torchvision's tree transforms (`ToImage`, `ToDtype`) only touch tensors, PIL
images and NumPy arrays, so `Polygons` leaves survive them.

## The kernel

`src/rfdetr.hpp` replicates pycocotools' `rleFrPoly` followed by `decode`:

1. Vertices are scaled by 5 and rounded as the C code does (`(int)(5x + .5)`,
   with the x86-64 conversion of NaN or out-of-range values to `INT_MIN`).
2. Every edge is walked as a dense integer line with the same double
   arithmetic (`(int)(ys + s*t + .5)`; the extension is compiled with
   `-ffp-contract=off`, matching pycocotools' x86-64 build, which has no FMA).
3. Consecutive points that change column produce a toggle at column `x` and
   row `y ∈ [0, h]`, exactly the "y-boundary points" of `rleFrPoly`.
4. `decode` alternates 0/1 between the sorted column-major positions
   `x*h + y`, so pixel `(y, x)` is set when an odd number of toggles lie at or
   before that position. The kernel keeps one bit per source column, initialized
   to the parity of the toggles in earlier columns (a column with an odd number
   of toggles leaks into every later column, and a toggle at `y == h` counts
   for later columns only, both as in pycocotools), sweeps the sampled source
   rows once flipping bits as toggles pass, and paints each 1-run through the
   column map with `memset`. Output rows that read the same source row are
   copied. Only ones are written, so several polygons of one instance union in
   the same zeroed plane, as `mask.any(dim=2)` does in RF-DETR.

The torch `nearest` index rule is `min(floor(float32(i) * (float32(in) /
float32(out))), in - 1)`; it was checked against
`torch.nn.functional.interpolate` on 5,993 size pairs with no mismatch.

## Results

![RF-DETR sample time and DataLoader throughput](assets/rfdetr-loader.svg)

Same host and configuration as the profile ([raw report](../bench/results/rfdetr-loader-linux-v1.json)):

| | RF-DETR reference | with ultrafast-maskops | Speedup |
|---|---:|---:|---:|
| `__getitem__`, single process | 20.99 ms | **10.26 ms** | **2.05×** |
| of which prepare / transforms / materialize | 5.34 / 13.28 / — | 0.26 / 7.42 / 0.34 | |
| DataLoader, 8 workers, batch 8 | 119.6 img/s | **206.0 img/s** | **1.72×** |

The remaining 10.3 ms are JPEG decoding, the PIL image resize and
normalization; the mask work went from about 11 ms to 0.6 ms. Whether a
training run speeds up depends on whether its GPU was waiting on the loader.

## Verification

- `tests/test_rfdetr.py`: the kernel against pycocotools + torchvision on
  random polygons (star-shaped, self-intersecting, partly outside, integer and
  half-integer vertices, repeated points, degenerate) through random
  resize/crop/flip chains, the identity chain against the full raster, and
  padded or flipped index maps. With RF-DETR installed, a synthetic COCO dataset
  (polygons, RLE, crowd and degenerate boxes) is drawn through an unmodified
  `CocoDetection` and through an accelerated one under the same seeds, for the
  square and the non-square training pipelines: images, boxes, labels, areas,
  sizes and masks are `torch.equal`.
- `bench/verify_rfdetr.py`: the whole val2017 split (5,000 images, 65,177
  instances) under two seeds each, 0 mismatches, and 2,000 further
  random-polygon fuzz cases on canvases up to 400 px (445 M output pixels)
  without a difference ([report](../bench/results/rfdetr-verify-linux-v1.json)).
- `bench/rfdetr_loader.py` checks 400 samples for identical digests before
  timing anything.

## Where to call it in RF-DETR training

`RFDETRDataModule.setup("fit")` builds the training dataset with
`build_dataset("train", ...)` and keeps it as `_dataset_train`; the
DataLoader is created later by `train_dataloader()`. A subclass that calls
`accelerate_dataset(self._dataset_train)` after `super().setup(stage)` (when
`_dataset_train.prepare.include_masks` is set) accelerates every worker,
because the traced transforms and `FastConvertCoco` pickle like RF-DETR's own.
`RFDETR.train()` constructs the data module internally and offers no hook for
a subclass yet, so this currently applies to Lightning runs that build the data
module themselves.

## Compatibility

- RF-DETR develop `2398ce3c` (1.11.0.dev0). `check_profile()` hashes the
  sources of `ConvertCoco`, `convert_coco_poly_to_mask`,
  `CocoDetection.__getitem__`, the four geometric transforms, `crop`,
  `_filter_per_instance_fields`, `_apply_to_masks` and the three combinators,
  and refuses anything else with a `RuntimeError` naming the function.
- The torchvision default backend only. Albumentations (`aug_config`) and the
  YOLO-format dataset (`rfdetr.datasets.yolo`, PIL `ImageDraw` masks) are not
  accelerated; `trace_transforms` raises on transform types it does not know.
- `gpu_postprocess=True` (kornia) is compatible: masks are materialized at the
  end of the CPU pipeline, before collation.
- Exactness is defined against pycocotools as built for x86-64 Linux. Builds
  of pycocotools that contract `ys + s*t` into a fused multiply-add could differ
  on a rounding boundary; none appeared in the tests run on an Apple M2.
