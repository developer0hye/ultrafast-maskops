# ultrafast-maskops

[![Wheel compatibility](https://github.com/developer0hye/ultrafast-maskops/actions/workflows/wheels.yml/badge.svg)](https://github.com/developer0hye/ultrafast-maskops/actions/workflows/wheels.yml)

Exact native replacements for the slow parts of segmentation data pipelines:
mask rasterization and segment geometry for Ultralytics, deferred mask
rasterization for RF-DETR. Every accelerated function returns the **same
bytes** as the pinned framework code, and the batches a trainer receives are
identical, so the training run does not change.

**Ultralytics: 13× faster mask rasterization · 4–9× faster segment geometry ·
1.4× faster augmented epochs. RF-DETR: 2.05× faster segmentation samples,
1.76× DataLoader throughput**, measured on real COCO data against the
unmodified pipelines.

The extension depends on nothing but pybind11 and NumPy. It bundles no OpenCV
and no pycocotools: the kernels replicate `cv2.fillPoly` with the 4× linear
downscale, and pycocotools' `rleFrPoly` with torchvision's nearest resize, crop
and flip, in integer arithmetic evaluated only where the output reads.

**Status:** alpha. Validated against Ultralytics commit
`795a556942a12fe0124cf767888194a1d0b83e2e` (8.4.149) and RF-DETR develop
`2398ce3c8730c88643f89d8e0283e71d14580bc7` (1.11.0.dev0); each adapter checks
the source hashes of the functions it replaces and refuses others.

## Installation

Prebuilt wheels for CPython 3.10–3.13 on Linux x86-64/aarch64, macOS and
Windows are built by the [release workflow](docs/publishing.md); until the
first PyPI release, install from source:

```bash
python -m pip install git+https://github.com/developer0hye/ultrafast-maskops.git
```

Requirements: Python 3.10+, a C++20 compiler and CMake 3.20+. Nothing is
downloaded during the build; the extension is one translation unit and about
0.3 MB.

## Quick start

Train with the Ultralytics `SegmentationTrainer` whose datasets are accelerated
after construction. Nothing global is patched.

```python
from ultralytics import YOLO
from ultrafast_maskops.training import FastSegmentationTrainer

YOLO("yolo11n-seg.pt").train(data="coco.yaml", trainer=FastSegmentationTrainer)
```

The same two calls work on any segmentation `YOLODataset` you build yourself,
for example inside a custom trainer's `build_dataset`:

```python
from ultrafast_maskops.ultralytics import accelerate_dataset
from ultrafast_maskops.geometry import accelerate_geometry

accelerate_dataset(dataset, persistent=True)   # mask rasterization
accelerate_geometry(dataset, persistent=True)  # resampling, affine boxes and clipping
```

`persistent=True` keeps the acceleration when Ultralytics rebuilds the
transforms for `close_mosaic`. The standalone wrappers `polygon2mask`,
`polygons2masks` and `polygons2masks_overlap` keep the reference signatures.

For RF-DETR, one call on a `CocoDetection` built with `include_masks=True`:

```python
from ultrafast_maskops.rfdetr import accelerate_dataset

accelerate_dataset(dataset)  # polygons ride through the transforms; masks are rasterized once, at the end
```

## Mask rasterization: 13× faster, byte for byte

`polygons2masks_overlap` is the reference's costliest per-instance step: it
fills every instance into a full-resolution image, downsamples it and sums it.
The native kernel computes the same 160×160 bytes from the two rows and two
columns of every 4×4 block that the downscale actually reads.

![Mask rasterization time per Format call on the M2 and the i5-10400](docs/assets/mask-stage.svg)

| Host | Reference | ultrafast-maskops | Speedup |
|---|---:|---:|---:|
| Apple M2 | 661 µs | **50 µs** | **13.2×** |
| Intel i5-10400 | 1,184 µs | **89 µs** | **13.4×** |

2,000 captured COCO `Format` calls (27,497 instances). The kernel retires
0.72 M instructions per call against the reference's 9.8 M.

## Segment geometry: 4–9× faster per stage, 1.45–1.63× per sample

Three stages of an augmented sample are per-instance Python loops over small
NumPy and OpenCV calls. Each is replaced by one native batch that follows the
reference operation by operation, in the input precision.

![Per-stage time of one augmented COCO sample on the M2 and the i5-10400](docs/assets/loader-stages.svg)

| Stage | M2 reference → ours | i5-10400 reference → ours |
|---|---:|---:|
| `resample_segments` | 0.82 → 0.18 ms (4.6×) | 1.66 → 0.39 ms (4.3×) |
| `RandomPerspective.apply_segments` | 2.24 → 0.61 ms (3.7×) | 3.60 → 0.56 ms (6.5×) |
| `Format._format_segments` | 0.69 → 0.07 ms (10.3×) | 1.25 → 0.14 ms (8.9×) |
| whole `__getitem__` | 10.09 → 6.95 ms (**1.45×**) | 14.15 → 8.71 ms (**1.63×**) |

The rest of a sample is JPEG decoding, `warpAffine` and HSV conversion, which
already run in OpenCV; removing the segment stages entirely would bound the
sample at 1.6–1.85×.

## One training epoch at COCO scale: 1.4× faster

118,287 COCO-distributed images, the Ultralytics train-mode dataset and
DataLoader (8 workers, batch 16, imgsz 640, every default augmentation), one
fresh process per measurement.

![Epoch time and peak memory for the reference, ultrafast-maskops, and with ultrafast-yolo-dataset](docs/assets/coco-epoch.svg)

| Pipeline | Constructor + epoch | Throughput | Peak memory (PSS) |
|---|---:|---:|---:|
| Ultralytics reference | 379 s | 316 img/s | 4.41 GB |
| ultrafast-maskops | **272 s (1.39×)** | 441 img/s | 4.23 GB |
| + [ultrafast-yolo-dataset](https://github.com/developer0hye/ultrafast-yolo-dataset) | 269 s (1.41×) | 441 img/s | **2.35 GB (−47%)** |

Before timing, every batch of a whole epoch (7,393 batches) was digested for
all three pipelines: the digests are identical.
[Method, per-stage anatomy and raw reports](docs/SEGMENT_GEOMETRY.md).

## RF-DETR: 2× faster segmentation samples, 1.7× DataLoader throughput

RF-DETR rasterizes every polygon at full resolution with pycocotools and then
resamples the whole mask tensor in each resize, crop and flip. The adapter
keeps the polygons, records what the random transforms chose, and evaluates
pycocotools' fill once, only at the output pixels.

![RF-DETR sample time and DataLoader throughput](docs/assets/rfdetr-loader.svg)

| | RF-DETR reference | with ultrafast-maskops | Speedup |
|---|---:|---:|---:|
| `__getitem__`, single process | 20.78 ms | **10.20 ms** | **2.04×** |
| DataLoader, 8 workers, batch 8 | 119.7 img/s | **210.9 img/s** | **1.76×** |

Intel i5-10400, COCO val2017, RF-DETR's default training transforms
(`square_resize_div_64`, multi-scale, resolution 560). The mask work of a
sample went from about 11 ms to 0.5 ms; what remains is JPEG decoding, the PIL
image resize and normalization. Under the same seeds, images, boxes, labels
and masks are byte-identical for every image of the split.
[Design, kernel and verification](docs/RFDETR.md).

## DataLoader workers: the gain holds at every worker count

Both frameworks, COCO val2017, one pass per measurement in a fresh process,
worker start-up excluded, best of two rounds, on the i5-10400 (6 cores, 12
threads).

![Images per second against num_workers for Ultralytics and RF-DETR](docs/assets/workers-sweep.svg)

| num_workers | Ultralytics reference → ours (img/s) | RF-DETR reference → ours (img/s) |
|---:|---:|---:|
| 0 | 69 → **112** (1.62×) | 41 → **76** (1.84×) |
| 2 | 128 → **196** (1.53×) | 67 → **118** (1.77×) |
| 4 | 240 → **355** (1.48×) | 108 → **187** (1.73×) |
| 6 | 321 → **453** (1.41×) | 119 → **205** (1.72×) |
| 8 | 333 → **452** (1.36×) | 122 → **212** (1.74×) |
| 12 | 360 → **464** (1.29×) | 124 → **216** (1.74×) |

RF-DETR's samples are heavier, so its loader scales with the physical cores
and the per-sample gain carries through (1.72–1.84×). Ultralytics saturates
the six cores at 6–8 workers; beyond that the main process's collation and
the shared memory bandwidth cap both pipelines, which narrows the gap.
Reports: [bench/results/workers-sweep-linux-v1](bench/results/workers-sweep-linux-v1).

## Exactness

- The mask kernel follows OpenCV 4.13's `clipLine`, `LineIterator`,
  `CollectPolyEdges` and `FillEdgeCollection` in 16.16 fixed point, evaluated
  only on sampled rows and columns; the 4× resize is a calibrated 16-entry
  table of the four sampled pixels.
- The geometry kernels are compiled with `-ffp-contract=off` and replicate each
  NumPy ufunc separately; `np.interp`'s fused multiply-add is calibrated against
  the installed NumPy, and the affine matrix product stays in NumPy because no
  calibration can guarantee a BLAS's summation order for every shape.
- Batches with NaN, infinity or negative zero, other ratios or sizes, colors
  other than 1, several contours in one mask, coordinates outside the exact
  envelope, and any image size whose cv2 results the calibration cannot
  reproduce run the unmodified reference code. `backend_info()` lists the
  sizes handled natively and the sizes handed back.
- The RF-DETR kernel follows pycocotools' `rleFrPoly` (scaled integer edge
  walk, column-major toggles, even-odd decode) with the C code's double
  arithmetic and integer conversions, and torch's `nearest` index rule; samples
  whose segmentations are not polygon lists take RF-DETR's own path.
- Verification: parity suites in SIMD and scalar modes on Linux, macOS and
  Windows (`ULTRAFAST_MASKOPS_SCALAR=1` selects the scalar twin of every vector
  path); differential fuzzing of the kernel against `cv2.fillPoly` over about
  500,000 random contours and 8 G output pixels without a difference; a whole
  COCO-scale epoch with identical batch digests; and a two-epoch training run
  across `close_mosaic` in which every batch the trainer received was identical
  ([tests/test_trainer.py](tests/test_trainer.py)). For RF-DETR: the kernel
  against pycocotools + torchvision on random polygons and chains, a synthetic
  COCO dataset through the unmodified and the accelerated `CocoDetection` under
  the same seeds ([tests/test_rfdetr.py](tests/test_rfdetr.py)), and the whole
  val2017 split (5,000 images, 65,177 instances) under two seeds each,
  byte-identical.

## Compatibility and limitations

- Ultralytics is pinned to the validated commit through source hashes. Other
  versions raise a `RuntimeError` naming the changed function rather than
  silently running.
- Requires NumPy 2. cv2 is whatever Ultralytics installed; it is not pinned.
- `mask_ratio=4` (the default) with image sides divisible by 4 is the native
  profile; everything else is correct but not accelerated.
- Custom `Format` subclasses and custom `build_transforms` overrides are not
  replaced; the helpers raise instead of guessing.
- RF-DETR: the torchvision default augmentation backend with the COCO-format
  dataset. Albumentations pipelines, the YOLO-format dataset and unknown
  transform types are refused rather than guessed; RLE segmentations run
  RF-DETR's own code. RF-DETR is pinned to the validated commit through source
  hashes, like Ultralytics.

## Development and verification

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[test]"
.venv/bin/python -m pytest -q tests/test_parity.py tests/test_sampled.py
```

The integration suite additionally needs `.[ultralytics]` at the pinned
Ultralytics commit (see `requirements-integration.txt`):

```sh
.venv/bin/python -m pytest -q tests
```

The RF-DETR tests need `pycocotools`, and for the end-to-end part RF-DETR at
the pinned commit (`pip install "rfdetr @ git+https://github.com/roboflow/rf-detr@2398ce3c8730c88643f89d8e0283e71d14580bc7"`).

Benchmarks are reproduced with `bench/mask_stage.py`,
`bench/geometry_loader.py`, `bench/coco_epoch.py` and
`bench/rfdetr_loader.py`; the README charts are
rendered from the recorded reports by `bench/plot_readme.py`. Raw reports live
in [bench/results](bench/results/README.md), earlier development records in
[docs/archive](docs/archive).

## License

[AGPL-3.0-only](LICENSE): the replaced functions derive from Ultralytics, which
is AGPL-3.0. The RF-DETR adapter replicates code from RF-DETR (Apache-2.0) and
pycocotools (BSD-2-Clause). The only compiled third-party component is
pybind11 ([licenses](licenses)).
