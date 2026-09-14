# ultrafast-maskops

Exact native replacements for the Ultralytics segmentation data pipeline's mask
rasterization and segment geometry. Every accelerated function returns the
same bytes as the pinned Ultralytics code; the batches a trainer receives are
identical (see [tests/test_trainer.py](tests/test_trainer.py)).

The extension depends on nothing but pybind11. It bundles no OpenCV: the mask
kernel replicates `cv2.fillPoly` and the 4x linear downscale in integer
arithmetic, evaluated only where the downscale reads, and is verified against
the cv2 you have installed the first time each image size is used. Any input
outside the native profile (`mask_ratio` other than 4, sides not divisible by
4, several contours in one mask, a cv2 whose results cannot be reproduced) runs
the unmodified reference functions with that cv2, so results never depend on
which path ran.

```python
from ultralytics import YOLO
from ultrafast_maskops.training import FastSegmentationTrainer

YOLO("yolo11n-seg.pt").train(data="coco.yaml", trainer=FastSegmentationTrainer)
```

`FastSegmentationTrainer` is the Ultralytics `SegmentationTrainer` whose
datasets are accelerated after construction; nothing global is patched. The
same two calls work on any `YOLODataset` you build yourself:

```python
from ultrafast_maskops.ultralytics import accelerate_dataset
from ultrafast_maskops.geometry import accelerate_geometry

accelerate_dataset(dataset, persistent=True)   # mask rasterization
accelerate_geometry(dataset, persistent=True)  # resampling, affine boxes and clipping
```

Results on a 118,287-image COCO-scale epoch and the per-stage numbers are in
[docs/SEGMENT_GEOMETRY.md](docs/SEGMENT_GEOMETRY.md).

The standalone wrappers `polygon2mask`, `polygons2masks` and
`polygons2masks_overlap` keep the reference signatures; `PackedPolygons` and a
per-worker `Rasterizer` expose the native engine directly.

## Development

Requires CMake, a C++17 compiler, and Python 3.10+. There is nothing to
download: the extension is one translation unit over pybind11.

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-test.txt
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m pytest -q tests/test_parity.py
```

These are macOS/Linux source-development commands. Python commands explicitly
use the environment just created; activation is not assumed. For the framework
adapter and full integration suite, install the pinned requirements separately:

```sh
uv pip install --python .venv/bin/python -r requirements-integration.txt
.venv/bin/python -m pytest -q tests
```

The core package does not install Ultralytics or Torch.

Integration additionally uses the exact Ultralytics commit
`795a556942a12fe0124cf767888194a1d0b83e2e`, Torch 2.10.0 and torchvision 0.25.0.
The adapter checks the full Format and mask-function source hashes and requires
NumPy 2. cv2 is not version-pinned: the native kernel is verified against the
installed cv2 per image size at first use, and `backend_info()` lists the sizes
it accelerates and the sizes it left to the reference code.

The current integration profile cannot be installed on Python 3.10 because
NumPy 2.4.4 requires Python 3.11 or newer. Package metadata allows a Python 3.10
core build with an older NumPy, but that is not a validated adapter combination.
Current execution evidence is on CPython 3.12; the proposed wider wheel/profile
matrix remains open.

```python
from ultrafast_maskops.ultralytics import accelerate_dataset

accelerate_dataset(yolo_segmentation_dataset)
```

Call this explicitly in a dataset factory/custom trainer after construction.
With the default one-shot option, rebuilding transforms requires opting in again. The pinned
trainer's `close_mosaic` step rebuilds them: reapply acceleration after that rebuild
and before DataLoader workers are reset. The helper accelerates the current
transform list; it does not install a persistent training-lifecycle hook. Custom
Format subclasses are not replaced. Rolling back means constructing the original
dataset normally.

The experimental `persistent=True` option instead sets the instance's base
`format_class` hook, so the reference builder can recreate FastFormat itself.
It requires the pinned base segmentation transform builder and default native
mode/budget; it leaves the class-level factory and other datasets unchanged.

## Evidence and remaining work

The differential corpus includes 10,000 seeded cases, zero/tied areas, 127/128/129/
255/256/500 instances, multiple contours, clipping, strided/read-only inputs,
ownership, concurrency, and import-order checks. Integration compares complete
Format outputs and actual YOLODataset/DataLoader batches with workers 0 and 2.

```sh
python bench/feasibility.py --out bench/out/host.json
```

This fixed synthetic suite uses five fresh processes per backend/case, 10 warmups
and 30 samples, alternating backend order. Input/output hashes, raw timings,
paired bootstrap intervals, whole-process peak RSS and host load are retained.
It includes wrapper packing costs; it does not measure training throughput.

This branch's unit-scale candidate passed 209 installed-wheel Linux tests and
the complete 60-process real 5,000-image DataLoader comparison. All outputs match,
but epoch median ratios span 0.984–1.050× and the 10% loader target remains unmet;
see [the loader results](docs/UNIT_SCALE_LOADER.md). Its separate 75-trace overlap
series reduced peak tracked allocation by 61–91% through the public wrapper;
this is not process RSS or training throughput. See
[the allocation results](docs/UNIT_SCALE_ALLOCATIONS.md).

Earlier builds completed CPU and RTX 3070 training validation, including a
45-job GPU series with exact loss/model agreement and limited throughput gains.
Those results do not validate this later kernel. They also keep mosaic enabled
through both measured epochs, so they do not verify a `close_mosaic` transition.
See [the GPU results](docs/GPU_RESULTS.md) and [protocol](docs/GPU_BENCHMARK_PROTOCOL.md).

Still required: exact-current-candidate M2/sanitizer/GPU and lifecycle validation,
representative performance improvements, broader fuzz/platform coverage,
verified Linux/macOS/Windows wheels, CI, and a release report against every PRD gate.
See [docs/STATUS.md](docs/STATUS.md). No training speedup or release readiness is claimed.
