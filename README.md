# ultrafast-maskops

Experimental C++/OpenCV batch polygon rasterization for Ultralytics. This is an
alpha implementation, not yet a published or release-qualified replacement.

Implemented: `polygon2mask`, `polygons2masks`, `polygons2masks_overlap`, owned
`PackedPolygons`, per-worker `Rasterizer`, and an explicit `FastFormat` adapter.
The original PRD and all release gates remain in [docs/PRD.md](docs/PRD.md).

```python
from ultrafast_maskops import polygons2masks_overlap

masks, order = polygons2masks_overlap((640, 640), segments, downsample_ratio=4)
classes = classes[order]
```

The wrapper includes input conversion, native rasterization, resizing, unsigned
area reduction, NumPy's reference sorting, and native composition. It preserves
uint8/int32 IDs at the 255-instance boundary. The packed API additionally permits
empty contours as zero masks; compatible wrappers reject empty contours explicitly.
Nonfinite/out-of-int32 coordinates and colors outside 0..255 raise `ValueError`.
Invalid-input exception types are not yet fully identical to the reference.

`Rasterizer.overlap(..., mode="retained")` stores resized uint8 masks and composes
without the original int32 stacks or indexing copies. `mode="bounded"` renders
twice to keep raster working storage independent of instance count. `auto` uses
the fixed scratch-budget rule in the source. Packed inputs, returned masks and
OpenCV's internal edge structures are additional memory; the scratch budget is
not a whole-process memory cap. Independent masks inherently need N*h*w output bytes.

Private OpenCV core/imgproc 4.13.0 is statically linked and its symbols hidden.
Internal OpenCV parallel regions are disabled using its private `setNumThreads(0)`;
Python `cv2` thread settings are preserved. Separate Rasterizer objects can run
concurrently, and one object serializes calls. No module import patches Ultralytics.

The optimized path clears only the previous polygon's dirty rectangle and composes
within conservatively padded resized support. It still rasterizes at the original
resolution and uses OpenCV's original linear resize; it does not approximate geometry.

## Development

Requires CMake, a C++17 compiler, and Python 3.10+. Building fetches the
SHA256-pinned OpenCV source; users of future wheels will not need a compiler.

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
NumPy 2.4.4 / Python cv2 4.13.0 / private OpenCV 4.13.0. An unknown profile raises
an actionable error. This narrow initial matrix will expand only after validation.

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
Rebuilding dataset transforms requires opting in again. In particular, the pinned
trainer's `close_mosaic` step rebuilds them: reapply acceleration after that rebuild
and before DataLoader workers are reset. The helper accelerates the current
transform list; it does not install a persistent training-lifecycle hook. Custom
Format subclasses are not replaced. Rolling back means constructing the original
dataset normally.

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

Complete real 5,000-image CPU loader and RTX 3070 training experiments are
available, including a 45-job GPU series with exact loss/model agreement.
GPU epoch median ratios were 1.002–1.021×, with several confidence intervals
including no improvement and no consistent family-RSS reduction. These are
results for the frozen measured builds, not later kernel candidates or every
current source change. See [the GPU results](docs/GPU_RESULTS.md) and
[the implementation status](docs/STATUS.md).

The GPU series keeps mosaic enabled through both measured epochs; it does not
verify a `close_mosaic` transition. Still required: exact release-candidate
performance and lifecycle validation, broader sanitizers/fuzz/platform coverage,
verified Linux/macOS/Windows wheels, CI, and a release report against every PRD gate.
See [docs/STATUS.md](docs/STATUS.md). No training speedup or release readiness is claimed.
