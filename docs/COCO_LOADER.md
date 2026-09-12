# Real COCO DataLoader experiment

This experiment measures the ordinary pinned Ultralytics `YOLODataset` on both
sides. The candidate only replaces its final base `Format` with `FastFormat`.
Dataset discovery, Pillow label-cache construction, image decoding/resizing,
1,000-point segment resampling, LetterBox, semantic targets and collation remain
upstream operations. The Rust dataset package is not used or required.

The fixture contains all 5,000 COCO val2017 images, 4,952 converted TXT files and
36,335 instances. There are 48 missing-label background images. Objects per image
have median 4, p95 22 and maximum 62. Converted polygons have median 18, p95 63 and
maximum 362 vertices; every nonempty image therefore reaches Format with 1,000
points per instance under the upstream resampling rule. This is a real-data test
of a different geometry distribution from the earlier 16-point synthetic suite.
It does not replace the N≥100 crowded-image allocation tests.

`bench/results/coco-source-manifest.json` records source archive URLs/hashes,
original annotation SHA-256, all 5,000 image hashes and the pinned upstream
converter. `coco-segment-manifest.json` records the converted fixture digest.
The converter's crowd/invalid-box exclusion, duplicate handling, category mapping
and multipart merge are unchanged. Source images and converted labels are ordinary
physical files. Preparation creates a new directory and refuses to reuse it.

After obtaining the two official archives named in the source manifest:

```sh
python bench/prepare_coco.py --archives /path/to/coco-archives --out /path/to/new-coco-fixture
python bench/coco_loader.py \
  --corpus /path/to/new-coco-fixture/segment \
  --out /path/to/new-loader-report.json --workers 0 2 8 --rounds 5
python bench/verify_coco_loader.py \
  --corpus /path/to/new-coco-fixture/segment \
  --benchmark /path/to/new-loader-report.json \
  --out /path/to/new-fresh-reference-verification.json
```

Each output and its `.runs` directory must be new. Keep the benchmarked package,
extension and scripts unchanged until the job ends. Do not run another benchmark,
build or test job on the same host concurrently. The final verification command
must run **after** timing: it deletes only this generated fixture's derived
`labels/val2017.cache`, rebuilds it with the original upstream scanner, and verifies
that every recorded output matches the fresh reference. Its timings are not
performance samples. It leaves image/annotation input files intact.

A short execution/parity smoke uses `--limit 64 --rounds 1 --workers 0 2`.
It is labelled `pilot` and is not evidence for a representative speedup or a
confidence interval. `--overlap no`, `--imgsz` and `--mask-ratio` support additional
fixed workloads; do not pool different configurations into one aggregate.

## Measurement boundaries

- Five alternating fresh processes per backend and worker count. Both packages
  are imported in every process; imports, source checking and dataset construction
  are outside epoch timing. The six complete upstream source files are checked
  against hashes independently verified against commit
  `795a556942a12fe0124cf767888194a1d0b83e2e`.
- Full ordered dataset, batch 8, 640×640, overlap masks at ratio 4. No augmentation,
  shuffling, image RAM cache or pinned memory. This is a deterministic CPU loader
  workload, not a complete training or augmented-epoch experiment.
- Spawn with 0/2/8 loader workers; persistent workers and prefetch factor 2 when
  workers are enabled. Each worker uses one Torch thread, one Python OpenCV thread
  and one private OpenCV thread. The eight-worker case can exceed physical cores
  on the six-core server; both backends receive the same setting.
- The first timed epoch includes worker creation, dataset transfer, queue fill,
  all batches and iterator exhaustion. `first_batch_s` reports startup-to-first
  delivery; `after_first_batch_s` reports the remaining first-epoch interval.
  The latter is not a separately warmed full epoch: prefetch may already have
  prepared some later batches by the first delivery.
- The consumer requests the next batch immediately. Interarrival times and their
  p50/p95 describe an unpaced consumer, not GPU training wait times. No hashes,
  equality comparisons or target reductions run inside the timed loop.
- A complete second deterministic pass hashes every collated key, Python type,
  tensor dtype/shape/value and metadata field. Output hashes must match between
  backends, all repetitions and all worker counts on the same host. Inputs are
  fingerprinted before/after each process. These full reads pre-warm the OS page
  cache, so there is no cold-storage claim.
- An external parent samples the benchmark process and recursive children every
  50 ms while the epoch marker is active. Reported **summed family RSS counts
  shared pages more than once** and can miss short peaks; it is not unique memory
  or working heap allocation. Worker launch/import memory is included. Actual
  sample count/max gap/process count are recorded. Parent VmHWM/ru_maxrss also
  includes dataset construction, but is captured before output verification.
- CPU/RAM, dependency versions, OpenCV builds, library/script/extension hashes,
  available memory, load averages and host-wide swap changes are retained. Swap
  deltas can include unrelated processes on these shared hosts. No outliers are
  discarded because they are unfavorable.

The raw run logs and 50 ms memory traces remain beside each local/server report;
the committed report retains every epoch time, all batch interarrival samples,
measured memory peaks, sampling diagnostics and output hashes. Per-configuration
medians, p95s and paired bootstrap intervals use a fixed seed (912) with 10,000
resamples. The five-process sample size and shared-host conditions limit inference
about small differences. A ratio labelled `reference_over_native` is reference
value divided by candidate value; for images/s, a smaller ratio favors native.

The full real-data results belong in [BENCHMARKS.md](BENCHMARKS.md). CPU loader
results do not prove GPU utilization, training throughput or accuracy. Augmented
batches, full GPU epochs and higher resolutions remain separate validation tasks.

## Direct stage diagnostics

After the full benchmark finishes, while its exact sources and extension are
still installed, run each backend sequentially:

```sh
python bench/time_coco_stages.py --corpus /path/to/fixture/segment \
  --benchmark /path/to/loader-report.json --backend reference \
  --out /path/to/new-reference-stages.json
python bench/time_coco_stages.py --corpus /path/to/fixture/segment \
  --benchmark /path/to/loader-report.json --backend native \
  --out /path/to/new-native-stages.json
```

This wraps individual methods with `perf_counter_ns`, asserts exact call counts,
then restores them before an untimed complete output check. Raw durations are
saved alongside the report. It is one instrumented diagnostic epoch, not an
additional sample for the repeated benchmark; nested inclusive times cannot be
summed. We rejected native-side cProfile component times after its call counts
failed to cover the full epoch. See the retained audit reports and results.

`python bench/audit_coco_evidence.py` rechecks the retained baseline's 60 samples,
source snapshots, medians/bootstrap intervals, fresh-reference output bindings
and raw stage durations without importing the native extension or requiring the
fixture. This is an artifact consistency check, not a replacement for rerunning
the experiment. Git history at `7c508a6` supplies baseline source bytes after
later implementation changes.
