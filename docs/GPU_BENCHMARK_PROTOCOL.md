# Repeated GPU training protocol

Preparation runs are separate from the comparison series. The completed combined
pilot proves that the installed wheels can complete real YOLO11n-seg training;
it is not included in the repeated timing summary. The matching reference pilot
also completed; its initial/final model hashes and all 1,250 batch loss vectors
are identical to the combined pilot. The paired-bootstrap covariance and
backend/epoch grouping checks passed before launching the first series.
Before starting a series, finish that host's other builds,
tests and benchmark jobs. Do not stop unrelated users' processes; retain system
load and memory observations and qualify shared-host conclusions.

The frozen primary configuration is the hashed 5,000-image COCO segmentation
fixture, random YOLO11n-seg weights, FP32, SGD, batch 4, 640 pixels, overlap masks
at ratio 4, seed 912, default pinned training augmentation, and two complete
epochs. Mosaic remains enabled through both epochs (`close_mosaic=0`). The
single parent process starts fresh reference, mask-only and combined replacement
children sequentially. CPU Torch and OpenCV each use one calling thread; workers
0, 2 and 8 are separate configurations. Upstream validation uses twice the
requested worker count; reports record both actual counts.

```sh
python bench/repeat_coco_gpu.py \
  --corpus /path/to/coco-val2017-yolo/segment \
  --fresh-check /path/to/mask-coco-bounds-fresh-server.json \
  --out /path/to/new-gpu-series.json \
  --workers 0 2 8 --repeats 5 --epochs 2 --batch 4 --imgsz 640
```

This is **45 full training jobs / 90 epochs**, including the normal local
validation/checkpoint/final-validation work in each job. Expect several hours.
Backend order rotates and alternates across five repetitions per worker count;
keep all observations. Five pairs provide limited uncertainty resolution and
must not be described as proof of a general gain. The script fails on incomplete
children or drift in runtime files, fixture, original cache, GPU identity or
initial model weights. Existing series files are never overwritten.

Report the first and second epoch separately. Epoch boundaries synchronize CUDA;
prefetch may already have started during loader construction. First-epoch GPU
warmup is included. Whole-job time includes trainer initialization, validation
and checkpoint saving, but excludes package imports and fixture preflight. The
native content cache starts cold in each job; the original annotation cache is
warm and has a weaker invalidation contract. Therefore, whole-job times are not
an equivalent cache-hit startup benchmark. Dataset constructor measurements
remain separate evidence.

The summary uses ratios of median epoch times and 10,000 **paired** bootstrap
resamples, seed 912, with nearest-rank 2.5/97.5 percentile intervals. Output keeps
each raw report's SHA-256 and command. Raw reports retain all per-batch loss
components, CUDA allocation/reservation peaks, sampled process-family RSS, load,
resolved configuration and source/runtime hashes. The parent keeps compact
summaries so retained traces do not progressively inflate its memory usage.
Sampled summed RSS double-counts shared pages and cannot replace working native
allocation measurements.

`complete=true` means all requested jobs and summaries finished, not that any
performance gate passed. Confirm full target parity separately and audit the raw
reports before publishing a speedup or memory claim. Same-fixture train/val and
random initialization make these throughput experiments, not accuracy studies.
Non-overlap is a separate complete series; it must not be inferred from overlap.
