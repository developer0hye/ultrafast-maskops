# Paired installed-wheel comparison preparation

`bench/compare_wheels.py` compares the tested masks-only baseline wheel with the
corrected resize-ROI wheel. It is prepared but **not measured or execution-tested
yet**. Only its read-only descriptor mode has run for both installed environments;
Ruff also passed. Do not interpret this protocol or the descriptors as a speedup.

The baseline is the M2 wheel with SHA-256
`0659680b4ee2098a610d033d08fe47ef6cd8a8612d30821b48b5b7a1b43fdce8`.
The corrected candidate is
`e5b5eb8b09bb66527365c6263e3882a529ca810f0340ba2375c51a180dd79481`.
Both are 0.1.0a1 archives, so package version alone cannot distinguish them.
Their installed extension and wrapper bytes match the respective wheel; compiled
kernel/CMake/profile-template hashes match their declared source roots.

Both descriptors have identical Python, NumPy, Python OpenCV build information,
wrapper/oracle/fixture code, compiler, CMake and profile-template hashes, native
thread count and CPU/RAM counts. The complete private OpenCV build information
is preserved for review; the prior selected build-option comparison is in the
resize-ROI dependency receipt. Compiler/build-file matching does not certify
every external toolchain configuration. See `validation/paired-wheel-*.json`.

## Frozen matrix and measurement scope

Use the unchanged nine cases in `bench/feasibility.py`: 640×640 images at ratio 4
with N=0/1/5/20/100/255/256/500, plus N=100 at ratio 1. The existing seeded
16-vertex polygons, including empty cases and integer dtype transitions, stay
unchanged. Execute three operations for every case:

- `polygons2masks_overlap`, using the public wrapper's automatic scratch mode.
- Explicit bounded overlap, including construction of the packed input and a
  new Rasterizer for each call.
- `polygons2masks`, the public non-overlap wrapper including packing and outputs.

Five paired fresh-process repetitions alternate which wheel goes first. Each
worker verifies full output hashes, warms ten calls, retains 30 latency samples,
and verifies full output hashes again. That is 270 measured processes. Two
separate oracle processes compute complete expected bytes before measurements;
their large reference output allocations do not occur in measured workers.
The benchmark checks wheel/source/environment identity in every worker and
retains raw JSON, logs, commands, hashes and partial results on failure.

Latency includes all wrapper work and output allocation/freeing; packed input
construction is not excluded. These synthetic mask-call measurements do not
establish actual training throughput. Fresh-process peak RSS includes imports,
descriptor/archive verification, inputs and warmups; it is not isolated working
allocation, process-family RSS or a claim of smaller full raster scratch.
Separate Memray measurements and full real-data parity remain necessary.

The aggregate compares medians of the five process medians and computes a paired
percentile bootstrap interval from all 3,125 ordered five-pair resamples. RSS is
reported separately with the same pairing. The script never removes a slower
case or changes the fixed matrix after measuring. Final evidence still requires
independent raw/aggregate auditing and runtime/dispatch review.

## Running after the host is free

Do not launch while the M2 500k-pair job is active. First qualify the oracle,
worker and aggregate paths with retained checks. Then execute the full matrix
from this candidate worktree with the two immutable environments:

```sh
python bench/compare_wheels.py \
  --baseline-python /measurement/mask-masks-only-clean-v1/bin/python \
  --baseline-root /source/ultrafast-maskops \
  --baseline-wheel /measurement/mask-masks-only-dist-v1/ultrafast_maskops-0.1.0a1-cp312-cp312-macosx_26_0_arm64.whl \
  --candidate-python /measurement/mask-resize-roi-clean-v2/bin/python \
  --candidate-root /source/ultrafast-maskops-resize-roi \
  --candidate-wheel /measurement/mask-resize-roi-dist-v2/ultrafast_maskops-0.1.0a1-cp312-cp312-macosx_26_0_arm64.whl \
  --out /measurement/mask-paired-wheels-m2-v1.json
```

Use actual mounted paths and a new output name. The current RSS/platform helpers
target macOS and Linux; this script is not Windows benchmark coverage. Linux
needs independently built and validated baseline/candidate wheels after the
active GPU series finishes. No process is queued automatically by this document.
