# ultrafast-maskops — Initial PRD

- Document version: 0.1 / Date: 2026-09-12
- Status: Proposal for starting implementation. Implementation, benchmarks, and package publishing have not been performed.
- Distribution name: `ultrafast-maskops` / Import name: `ultrafast_maskops`
- Proposed initial stack: C++17 + OpenCV + pybind11 + scikit-build-core
- Primary users: Ultralytics Instance Segmentation training users and developers of CPU annotation processing tools
- Reference source: Ultralytics commit `795a556942a12fe0124cf767888194a1d0b83e2e`.

## 1. Product definition and goals

A Python extension that converts polygons into training masks through a batch CPU operation. It preserves the existing Python API and mask values while reducing per-polygon calls, temporary arrays, and memory traffic during overlap composition.

The principles adopted from `ultrafast-pycocotools` are explicit reference parity, native data lifetime management, and reproducible benchmarks on real workloads. Published speedups from that project are not performance estimates for this project. [Project README](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/README.md)

| ID | Goal | Evidence of success |
|---|---|---|
| G1 | Preserve training targets after replacement | Identical mask bytes, shapes, dtypes, and instance ordering within supported environments |
| G2 | Reduce CPU mask preparation time | Geometric-mean speedup ≥1.5× on a fixed representative overlap suite, including wrapper costs |
| G3 | Reduce temporary memory for crowded images | ≥30% reduction in peak working allocation on a fixed suite with N≥100 |
| G4 | Support practical adoption | Clean wheel installation and passing Ultralytics DataLoader integration |
| G5 | Make performance claims reproducible | Published input hashes, commands, environments, raw samples, and parity reports |

These numbers are proposed product targets, not measured results. Freeze benchmark cases and hardware in M0; do not remove cases because their results are unfavorable. If G2/G3 are not met, remain in alpha or revise the PRD and scope with supporting evidence.

## 2. Scope and non-goals

### Required v0.1 scope

1. Compatible wrappers for `polygon2mask`, `polygons2masks`, and `polygons2masks_overlap`.
2. A separate native API accepting packed polygons.
3. Buffer reuse, bounded-memory overlap generation, and low-overhead paths for small inputs.
4. Version-gated Ultralytics integration examples and a differential test harness.
5. Verified wheels for Linux x86-64, macOS arm64, and Windows x86-64.

### Follow-up scope

`masks2segments`, `resample_segments`, `segment2box`, and `segments2boxes` remain part of the product direction but are excluded from the v0.1 native release gate. Implement the original six-function concept in stages. Do not export unimplemented functions as if they were native implementations.

### Non-goals

- GPU mask kernels, CUDA/Triton, tensor autograd, or acceleration of model Inference itself.
- Replacement of COCO RLE/evaluation, NMS, dataset TXT parsing, or image decoding.
- Rewriting the entire augmentation pipeline or automatically modifying an Ultralytics installation.
- Direct low-resolution rasterization, different interpolation methods, or polygon simplification in exact mode.
- Guaranteed support for arbitrary GIS polygon topology or general holes/multipolygon specifications.
- Guaranteed speedups for entire epochs or training workloads that already saturate the GPU.

## 3. Target Ultralytics hot paths

| Priority | Location / function | Optimization unit |
|---|---|---|
| P0 | `data/utils.py::polygon2mask` | Dtype conversion, rasterization, resizing, and scratch allocation |
| P0 | `data/utils.py::polygons2masks` | Processing multiple instances in one native batch |
| P0 | `data/utils.py::polygons2masks_overlap` | Downsampled mask areas, ordering, and instance ID composition |
| P0 integration | `data/augment.py::Format._format_segments` | Masks and the matching class/instance index permutation |
| P1 | `utils/ops.py::masks2segments` | Batch contours, contour selection, and merging |
| P1 | `utils/ops.py::resample_segments` | Vertex interpolation and ragged output |
| P1 | `utils/ops.py::segment2box`, `segments2boxes` | Distinct geometry for clipped xyxy boxes and annotation xywh boxes |

The primary references are the [mask functions](https://github.com/ultralytics/ultralytics/blob/795a556942a12fe0124cf767888194a1d0b83e2e/ultralytics/data/utils.py#L404) and [Format implementation](https://github.com/ultralytics/ultralytics/blob/795a556942a12fe0124cf767888194a1d0b83e2e/ultralytics/data/augment.py). Measure full-resolution scratch storage separately from per-instance downsampled mask storage. OpenCV operations already execute natively, so removing Python loops alone may not produce substantial gains.

## 4. Compatibility and parity contract

### 4.1 Reference profile

Record the Ultralytics commit, Python/NumPy/OpenCV versions, OpenCV build options, OS/architecture, and extension build ID in `compatibility.json`. Freeze the first installed and executed combination in M0. This document does not yet declare a verified dependency version combination.

- Exact parity means bitwise equality with the reference in the specified profile.
- Expand the supported version range only through a passing compatibility matrix. There is no blanket guarantee for every OpenCV/NumPy version.
- Floating-point tolerance comparisons during development are diagnostic tools. They cannot replace the v0.1 integer mask and index gates.

### 4.2 Required invariants

| ID | Contract | Verification |
|---|---|---|
| C01 | Image size is `(height, width)` | Rectangular images and odd dimensions |
| C02 | Preserve the reference sequence: coordinates → int32 → fill → resize | Negative, fractional, and out-of-bounds coordinates; dtype-specific conversion |
| C03 | Resized output shape is `(H // r, W // r)` | r=1,2,4,8 and dimensions not evenly divisible by r |
| C04 | Preserve reference OpenCV interpolation and rounding | Default linear resize, thin objects, and one-pixel boundaries |
| C05 | Compute overlap areas from resized masks | Do not order by geometric polygon area or full-resolution mask area |
| C06 | Match the reference area-based index permutation | Preserve area dtype, negation, zero areas, and ties; do not substitute an arbitrary stable sort |
| C07 | IDs are sorted rank+1; background is 0 | Verify composition order and higher IDs at overlapping pixels |
| C08 | Overlap output dtype is uint8 for N≤255 and int32 for N>255 | Fully overlapping cases with N=127,128,129,255,256 |
| C09 | Do not modify original inputs | Read-only, strided, and sliced arrays |
| C10 | Preserve each function's empty-input contract | Legacy empty results versus packed results, as specified below |
| C11 | Preserve correspondence between classes, boxes, instances, and mask IDs | Compare complete Format outputs, not only masks |

The oracle includes NumPy `argsort` ordering for area ties. It must also preserve the unsigned dtype produced by area reduction and the `-areas` operation. In particular, zero-area masks can make a simple mathematical descending sort produce different results. The initial implementation may perform one NumPy sort of an area vector with the same dtype in the wrapper. Include all raster batch → vector sort → native composition round-trip costs in measurements. Enable a fully native sort only after it passes zero-area and tie cases.

In the current reference, `polygons2masks(..., polygons=[])` returns a float64 array with shape `(0,)`. The compatible wrapper must preserve this behavior. Empty overlap input returns a downsampled 2D uint8 mask and an empty index array. Only the packed API may provide a normalized contract such as uint8 `(0, h, w)`.

The v0.1 supported domain limits `color` to integers from 0 through 255 and reproduces OpenCV results. Route out-of-range colors, NaN/Inf or out-of-int32-range coordinates, and malformed polygons before entering native kernels. The compatible wrapper must either explicitly fall back to the installed reference or raise a documented unsupported-input error. Do not silently repair coordinates to make an input succeed.

`downsample_ratio` must be a positive integer. The packed API raises `ValueError` when the resulting width or height is zero. Record differences in wrapper exception types in the compatibility table. Do not hide native memory safety errors or allocation failures behind ordinary compatibility fallback.

### 4.3 Integration contract

- Provide direct import replacement and a test adapter first. Importing the package must not apply a global patch.
- Preserve the class/instance index permutation applied by `Format._format_segments`.
- Do not assume that changing a `data.utils` attribute replaces symbols already imported into `augment`. Verify the actual call path through integration tests.
- Do not automatically enable the native backend for unknown Ultralytics profiles. Run the reference or return an explicit error with a reason.
- If an optional `torch` adapter is added later, include GPU→CPU copies and synchronization in its performance accounting.

## 5. Proposed Python API

The following signatures are design proposals, not APIs of an already installable package.

```python
from ultrafast_maskops import (
    polygon2mask,
    polygons2masks,
    polygons2masks_overlap,
)

polygon2mask(imgsz, polygons, color=1, downsample_ratio=1)
polygons2masks(imgsz, polygons, color, downsample_ratio=1)
polygons2masks_overlap(imgsz, segments, downsample_ratio=1)

# The same two return values as the reference.
mask, order = polygons2masks_overlap((640, 640), segments, 4)
sorted_classes = classes[order]
```

The packed API serves callers that want to reuse packed inputs across calls.

```python
from ultrafast_maskops import PackedPolygons, Rasterizer

# points: contiguous int32[P, 2]
# offsets: contiguous int64[N + 1], 0 ... P, monotonic
packed = PackedPolygons(points=points_i32, offsets=offsets_i64)
engine = Rasterizer(num_threads=1, scratch_limit_bytes=64 * 1024**2)
masks = engine.masks((640, 640), packed, color=1, downsample_ratio=4)
overlap, order = engine.overlap((640, 640), packed, downsample_ratio=4)
```

- Packed v0.1 represents one contour per instance. Supporting multiple contours per instance requires separate contour and instance offsets.
- Multiple contours passed to `polygon2mask` must preserve the semantics of one reference fill call. Do not replace this with an OR of independently rendered masks.
- The general wrapper must not reduce float64 coordinates to float32 before int32 conversion; doing so can change truncation near boundaries.
- A packed object is an owned, immutable snapshot. Copy on construction by default; introduce zero-copy input only after defining an explicit ownership contract.
- Returned ndarrays must have safe Python ownership. Later calls must not overwrite earlier results.
- `Rasterizer` is an explicit per-worker object. Serialize concurrent calls to the same object with a lock in v0.1; do not use global scratch storage.
- Expose the profile, native backend, thread settings, and fallback counts through `backend_info()`.

## 6. Native architecture

```text
Python compatible wrapper / PackedPolygons
    → shape, dtype, and offset validation; input ownership
    → C++ batch rasterization using OpenCV core/imgproc
    → area collection → reference-compatible ordering
    → bounded-memory ID composition
    → owned NumPy outputs
```

### 6.1 Language and dependency decisions

Start with C++ to reuse OpenCV fill/resize semantics. Proposed tooling is pybind11 for Python bindings, scikit-build-core/CMake for builds, and cibuildwheel for wheels. Rust/PyO3 with OpenCV bindings is an alternative, but the initial release must not maintain two FFI stacks simultaneously. Explore a pure Rust rasterizer after the reference corpus exists.

The core package requires NumPy; place Ultralytics/PyTorch in an integration extra. Compatible fallback may require separate Python OpenCV/reference dependencies. Freeze actual minimum dependency versions after build verification in M0.

### 6.2 Memory strategy

Notation: N=instance count, P=total vertex count, A=H×W, a=(H//r)×(W//r), T=worker count.

- Independent mask outputs inherently require O(Na) bytes. Do not claim O(a) storage for the output itself.
- Reuse full-resolution scratch in the single-mask path and resize into an output slice.
- The initial overlap baseline retains uint8 downsampled masks while eliminating unnecessary full int32 stacks and advanced-indexing copies.
- The bounded-memory path first rasterizes to collect only areas, then rasterizes, resizes, and composes again in sorted order. Peak storage is approximately O(TA + Ta + a + P + N), but rasterization runs twice.
- Streaming composition one instance at a time avoids shared-output races. Parallel rasterization uses fixed-size chunks and an ordered queue.
- `scratch_limit_bytes` is the target scratch budget. Fail clearly when it cannot accommodate even the required single scratch buffer. Report input and output storage separately.
- Compressed runs/tiles are a v0.2 experiment. Preserve pixels after resize and include inputs that compress poorly.
- `auto` selection uses only predetermined criteria based on N, dimensions, and budget. Report the fast and low-memory paths separately.

### 6.3 Threading and error handling

Release the GIL only in C++ regions that do not access Python objects. Start with one native thread by default to avoid oversubscription with DataLoader workers. Do not silently change OpenCV's process-global thread settings. Benchmark the parallel backend together with the OpenCV thread policy.

Use checked dimension multiplication and offset arithmetic. Translate C++ exceptions into Python exceptions. Do not overwrite user ndarrays as OpenCV scratch or return freed buffers. Check for `KeyboardInterrupt` at chunk boundaries.

### 6.4 Packaging

First validate bundling a pinned minimal build of OpenCV `core`/`imgproc` as a private native dependency. Do not assume that installing `opencv-python` supplies C++ headers/libraries or a stable C++ ABI. Test different import orders of Python `cv2` and private OpenCV to detect symbol conflicts.

Proposed v0.1 wheel matrix: CPython 3.10–3.13 × Linux x86-64 / macOS arm64 / Windows x86-64. The first release does not promise Ultralytics' entire, broader Python support range. Explicitly exclude combinations that fail OpenCV build or NumPy ABI checks and update release scope. Linux arm64, macOS x86-64, and Python 3.14 are follow-up targets.

## 7. Benchmark plan

### 7.1 Workloads

| Layer | Fixed workload | Measurements |
|---|---|---|
| Micro | N=0,1,5,20,100,255,256,500; vertices=3,16,100,1000 | Per-function latency, throughput, and allocation |
| Geometry stress | No overlap, full overlap, equal areas, long thin polygons, out-of-bounds polygons | Parity and worst-case time/memory |
| Resolution | 320²,640²,1280²,1920×1080; r=1,2,4,8 | Scaling with pixel count |
| Real corpus | Fixed COCO instance subset preserving per-image polygon distributions | Representative median/p95 |
| Integration | Fixed Ultralytics segmentation dataset and augmentation seed | `_format_segments`, batch wait, images/s |
| End-to-end | One epoch with fixed model, batch, workers, and device | Startup, steady state, and total wall time |

Publish COCO conversion rules and annotation hashes. A synthetic suite requiring no downloads is mandatory. Maintain required boundary cases and fixed representative combinations in a manifest instead of requiring the full Cartesian product.

### 7.2 Comparison method

- Distinguish reference Python+NumPy+OpenCV, native wrapper, packed native, and bounded-memory modes.
- Include packing, input copying, sorting, ndarray wrapping, and fallback in wrapper performance. Label packed-only results separately.
- Use reasonable OpenCV/worker settings for the reference. Provide both single-thread comparisons and comparisons with equal total CPU budgets.
- Run 10 warmups followed by ≥30 samples for microbenchmarks; use ≥5 independent process runs for heavy/integration cases. Alternate execution order.
- Save median, p95, bootstrap 95% confidence intervals, speedup, wall time, peak RSS, and peak native allocation.
- Check parity for every raster mode comparison and retain input/output hashes. Perform correctness checks outside timing boundaries, consistently for both backends.
- Compare DataLoader workers=0/2/8, native threads=1, and bounded parallel settings. GPU utilization is a secondary metric.
- Record how model downloads, JIT/warmup, and image decoding are handled within timing boundaries.

Estimate overall acceleration using `1 / ((1-f) + f/s)`, where f is the fraction of time in the target path and s is its speedup. A 2× microbenchmark result does not imply 2× training throughput.

### 7.3 Proposed release performance gates

- Geometric-mean speedup ≥1.5× on the representative overlap suite, including wrapper costs; stretch target: 2×.
- ≥30% reduction in peak working allocation on the N≥100 memory suite; also publish total process peak RSS.
- For representative N≤5 cases, median latency must not regress by more than the larger of 10% of reference latency or 20µs.
- ≥10% reduction in batch preparation wall time on a fixed CPU-bound DataLoader case.
- No minimum speedup gate for GPU-dominated epochs. Document the causes and avoidance conditions for significant regressions before release.

## 8. Testing strategy

| ID | Test | Acceptance criteria |
|---|---|---|
| T1 | Differential golden corpus | 100% shape/dtype/byte/order equality for supported inputs |
| T2 | Property-based random polygons | ≥10,000 seeded cases; store minimized mismatch fixtures |
| T3 | Empty/invalid/strided/multi-contour cases | Wrapper contract and safe failures |
| T4 | N=128/255/256 overlap and area ties | No overflow; identical reference IDs |
| T5 | Input lifetime, repeated calls, and concurrency | No mutation, use-after-free, or races |
| T6 | DataLoader integration | Correct masks/classes/boxes/sem_masks correspondence and identical results across workers |
| T7 | Native sanitizers / fuzzing | No crashes under ASan/UBSan and offset/dimension/geometry fuzzing |
| T8 | Built wheels | Golden smoke tests pass in fresh environments without a source checkout |

Include polygons with 0/1/2 points, repeated vertices, self-intersections, reversed winding, coordinates exactly on boundaries, large coordinates, and small polygons that disappear after resizing. Verify that nonfinite inputs never reach native kernels.

Validate exact training targets first and use short deterministic CPU training loss checks as supporting evidence. Do not use nondeterministic GPU training loss as a bitwise acceptance criterion.

## 9. Milestones and implementation sequence

| Stage | Estimated development effort | Deliverables / exit criteria |
|---|---|---|
| M0: Oracle and feasibility | 3–5 days | Pinned environment, corpus, profile, baseline, OpenCV wheel spike, and bottleneck breakdown |
| M1: Exact MVP | 1 week | Three wrappers and C++ batch baseline; T1–T4 pass |
| M2: Memory/CPU improvements | 1–2 weeks | Reuse/two-pass/chunk strategies; evaluate G2/G3 gates |
| M3: Integration | 1 week | Opt-in actual Format path, DataLoader tests, unsupported-profile fallback |
| M4: Release candidate | 1 week | All wheel smoke tests, sanitizers, reproducible report, release checklist |

Estimates assume one developer and may change with OpenCV packaging or parity issues. If M0 shows little room to improve total wrapper cost, record a go/no-go decision rather than expanding into contours/resampling to obscure the result.

Initial issue sequence: reference manifest → overlap/empty/tie fixtures → packaging spike → three-function binding → allocation baseline → two-pass benchmark → Format adapter → wheel CI.

## 10. Risks and mitigations

| Risk | Impact | Mitigation / decision point |
|---|---|---|
| OpenCV raster/resize differences | Changed mask targets | Pin build profiles; create golden images in M0; disable mismatching backends |
| NumPy area tie ordering | Changed ID/class mapping | Start with reference vector sort; gate native sorting separately |
| Existing native operations dominate | Insufficient speedup | M0 profiling; packing-inclusive microbenchmarks and DataLoader measurements |
| Two-pass recomputation | Lower memory but higher CPU cost | Separate memory/performance modes and publish automatic-selection rationale |
| Wheel size and symbol conflicts | Installation failures | Minimal private OpenCV; cv2 import-order tests |
| Thread oversubscription | Regressions with multiple workers | One thread by default; cap total CPU budget |
| Upstream function changes | Silent semantic drift | Immutable oracle, scheduled compatibility CI, version gates |
| Licensing and distribution setup | Delayed public repository readiness | Finalize licenses, NOTICE, and native dependency inventory in M0 |

The proposed initial license is AGPL-3.0-or-later; finalize it when creating the repository. Preserve attribution and applicable licensing when porting reference code. This PRD does not grant separate permission to use a permissive license. [Upstream license declaration](https://github.com/ultralytics/ultralytics/blob/795a556942a12fe0124cf767888194a1d0b83e2e/pyproject.toml)

## 11. v0.1 release criteria

- [ ] Supported scope, reference commit, and dependency matrix are fixed.
- [ ] Evidence for C01–C11 and T1–T8 is linked to the release commit.
- [ ] There are zero known mask/index mismatches within the supported input domain.
- [ ] Proposed performance gates are met, or an evidence-backed PRD revision has been approved before release.
- [ ] Published benchmarks include wrapper and fallback costs.
- [ ] Supported Linux/macOS/Windows wheels have been installed and executed in clean environments.
- [ ] Lifetime/concurrency tests pass for Rasterizer reuse and DataLoader workers.
- [ ] README, API reference, supported/unsupported inputs, benchmark reproduction, CONTRIBUTING, and LICENSE/NOTICE are available.
- [ ] CI passes for the exact release candidate commit, and installation of distribution artifacts has been verified.
- [ ] Opt-in, rollback, and profile-mismatch behavior are documented.

## 12. Future roadmap

- **v0.2:** `masks2segments`. Freeze external contour ordering, selection by contour point count rather than area for `largest`, and `merge_multi_segment` connection ordering for `all` in the oracle.
- **v0.2:** `resample_segments`. Define separate contracts for input-list mutation, bypassing inputs already containing n points, endpoint/vertex preservation, and float32 rounding.
- **v0.3:** `segment2box` and `segments2boxes`. Do not collapse clipped image-space xyxy and unclipped annotation xywh semantics into a single min/max kernel contract.
- **v0.3:** Compressed raster runs, tile composition, contour-aware packed representations, and Linux arm64 wheels.
- **Research:** Pure Rust backend, SIMD geometry, and GPU-aware handoff. Keep these in an experimental namespace until they pass the exact contract.
- **Explicit separate modes:** Offer direct low-resolution rasterization or approximate polygon simplification only through separate APIs accompanied by accuracy reports.

The reference for follow-up geometry semantics is the [pinned ops.py](https://github.com/ultralytics/ultralytics/blob/795a556942a12fe0124cf767888194a1d0b83e2e/ultralytics/utils/ops.py).

## 13. Decisions to finalize in M0

1. The first passing Python/NumPy/OpenCV profile and minimum OS versions.
2. Private OpenCV build configuration and binary size budget.
3. The fixed representative manifest and hardware for the 1.5× gate.
4. The cost/accuracy trade-off between retaining wrapper sorting and implementing native tie sorting.
5. Whether the three-function native MVP's performance and memory gains justify continuing the project.

This document defines product requirements and a validation plan. Its signatures, targets, and proposed platforms do not represent completed implementation or measured performance.
