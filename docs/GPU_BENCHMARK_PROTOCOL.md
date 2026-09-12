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
0, 2 and 8 are separate configurations. Upstream validation requests twice the
training worker count, subject to the loader's CPU/device and batch-count caps.
On this 12-logical-CPU, single-GPU host, the actual training/validation counts are
0/0, 2/4 and 8/12. Reports record both actual counts. The pinned `data/build.py`
source hash was independently matched to the recorded upstream hash before
correcting an audit assumption that had expected 16 validation workers.

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

## Independent artifact audit

`bench/audit_gpu_series.py` uses only the Python standard library. It checks the
parent/trial script hashes, every raw report hash, compact/raw agreement, recorded
commands, counterbalanced execution order, source/runtime/fixture identity,
resolved training settings, actual workers, active replacements and cache state.
Each epoch must have 5,000 images, 1,250 complete finite loss vectors and valid
CUDA allocator peaks. Within every complete worker/repetition group, all three
backends must have exactly matching loss traces and final model hashes. The
sampled RSS maximum is recomputed from the full trace; sample gaps are reported.

The auditor refuses incomplete series by default. `--allow-partial` permits a
clearly labeled interim artifact check and never produces a timing summary.
For a complete series it independently recomputes all paired epoch summaries
and checks the recorded values. Relocated reports can be supplied with
`--runs-dir`; the original recorded commands and paths remain unchanged.

```sh
python bench/audit_gpu_series.py \
  --series /path/to/completed-gpu-series.json \
  --runs-dir /path/to/copied-raw-reports \
  --out /path/to/new-audit.json
```

An immutable eight-trial checkpoint passed the interim audit. Its two complete
groups (repetition 0, workers 0 and 2) have equality for both full epochs and
final models across all three backends. Workers 8 had two completed backends at
the snapshot and is not a complete group. The initial auditor's worker-cap
rejection and its source are retained with the corrected audit and raw reports;
no training report was changed to make the check pass. This is not a final
series result or a speedup claim. The
[checkpoint archive](../bench/results/cuda-series-checkpoint-8-trials.tar.gz)
and `validation/cuda-series-checkpoint-8-{audit,manifest}.json` retain this evidence.

A later immutable 17-trial checkpoint also passed, covering 34 full epochs.
Five complete groups now have exact loss-vector and final-model equality:
repetition 0 at workers 0/2/8, and repetition 1 at workers 0/2. The incomplete
repetition-1/workers-8 group is not counted as a complete comparison. Source,
runtime, command, raw/parent and memory-accounting checks passed for all 17
trials. No aggregate timing is emitted. The original server archive and its
downloaded copy have the same SHA-256; see the
[17-trial archive](../bench/results/cuda-series-checkpoint-17-trials.tar.gz) and
`validation/cuda-series-checkpoint-17-{audit,manifest}.json`. The older checkpoint
and rejected early auditor result remain preserved.

The subsequent immutable 27-trial checkpoint passed the same independent audit,
covering all three backends at workers 0/2/8 for the first three repetitions:
54 full epochs and nine complete backend groups. Every group has identical
loss vectors for every batch and identical final model hashes across reference,
mask-only and combined replacements. Runtime/command/source identity, complete
epochs, raw/compact records and memory accounting also passed. The preserved
[27-trial archive](../bench/results/cuda-series-checkpoint-27-trials.tar.gz) and
`validation/cuda-series-checkpoint-27-{audit,manifest}.json` retain exact bytes.
This is training-equivalence evidence for the frozen original Linux wheels;
it does not validate the later masks-only or resize-ROI candidates. Eighteen
trials remain in the planned comparison. No interim timing summary is emitted,
and matching losses/models do not establish model accuracy or replace direct
target parity tests.

Twelve synthetic corruption/false-completion cases in
`tests/test_gpu_series_audit.py` passed on M2, including the complete-series
bootstrap branch with known constant ratios and the CPU-capped worker case.
They also reject altered loss/model records after recomputing archive hashes,
truncated traces, false completion, reordered trials, altered RSS and forged
statistics. The combined 21-test JUnit report (including nine dataset-fixture
checks) and source hashes are in `validation/benchmark-audits-m2.*`. A complete
real-series audit is still pending; synthetic success is not that broader claim.

The RSS trace spans trainer initialization, training and validation work. It is
not a per-epoch training-RSS peak and does not observe between-sample spikes.
CUDA peaks cover the allocator during each epoch, not all device allocations or
the entire job. These scopes stay separate in comparisons. Full-target parity,
performance gates, real-host contention and generalizability still require
their own evidence after an artifact audit passes.
