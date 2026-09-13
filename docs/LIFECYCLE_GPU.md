# GPU lifecycle qualification: full grid launched, results pending

The full 54-trial, 126-epoch grid has now launched on the RTX 3070 server. The
first `w2-yes-fresh-reference` trial is active. The current dataset wheel and
frozen mask packet wheel passed 166 dataset and 257 mask tests in the same new
Linux runtime, plus standalone packaging/cache checks, before this launch.
No completed cohort or new throughput result is claimed yet. Earlier GPU
measurements used another adapter and `close_mosaic=0`; they still cannot qualify
this lifecycle protocol. The original factory/shared-storage failures and later
`file_system` exit failures remain preserved separately.

The default trial still runs the original full 5,000-image training workload.
Optional arguments enable the separate lifecycle protocol:

- `--close-mosaic N` passes the closing window to the unmodified trainer.
- `--persistent-mask` opts this trial's native datasets into the instance factory.
  The reference backend records the flag without replacing its formatter.
- `--checkpoint-epochs E ...` preserves exact intermediate `last.pt` bytes in
  the trial directory before later epochs or final checkpoint stripping. Epoch
  numbers are zero-based and must precede the final epoch.
- `--resume-from PATH --resume-receipt JSON` resumes a retained checkpoint from
  a completed reference trial with the same script, installed packages, upstream
  sources, resolved corpus path, corpus bytes, cache proof and training settings. The checkpoint must match
  the receipt's recorded path and SHA-256 before PyTorch deserializes it.

All outputs use new directories. Resume passes the upstream-supported `save_dir`
override so restored checkpoint arguments cannot send writes to the original
trial directory. Input checkpoint and receipt bytes must remain unchanged; the
reported receipt hash binds the exact bytes parsed before training. The resolved
corpus path is checked because relative content hashes alone do not bind paths
stored in reference caches and label dictionaries. Intermediate
optimizer state and EMA updates must be present, and their state counts/update
counter must be observed after the unmodified resume routine restores them.
This is not an independent byte-by-byte audit of every restored optimizer tensor.

## Observe the actual transition

The pinned epoch-start callback runs before the trainer closes mosaic and resets
the loader. This harness checks the first batch after those calls: the expected
Format/FastFormat and instance factory must be present, a new Compose must have
replaced the initial one, and the prior worker processes must have stopped with
new PIDs at the boundary when workers are enabled. Only integer transform IDs
are retained; old transform objects and native scratch buffers are not kept alive
for observation. The pinned builder constructs the replacement while the old
Compose still exists, allowing those IDs to distinguish the single rebuild.

The trainer's close/reset/resume methods are not overridden. Observations inspect
the parent dataset and worker processes, not instrumentation inside worker mask
calls. Per-batch loss traces and final model hashes must be compared separately
across backends; this observation alone cannot establish input/model parity.
Partial dataset/epoch/lifecycle/checkpoint records are retained on trial failure.

Trainer and configuration-source hashes are additionally pinned for this path.
The reviewed resume implementation closes mosaic during setup only when
`start_epoch > epochs - close_mosaic`; equality is handled by the normal epoch
loop. The selected protocol exercises both branches.

## Declared full qualification grid

Use the existing real 5,000-image COCO fixture and fresh-reference cache proof.
Keep overlap yes/no and workers 0/2/8 as separate conditions. Within each condition:

1. Run reference, mask-only and combined dataset+mask backends for four epochs,
   `close_mosaic=2`, `persistent-mask`, and retained checkpoint epochs 1 and 2.
   Epochs 0/1 precede closure; epochs 2/3 follow closure.
2. Resume all three backends from the **same reference epoch-1 checkpoint**.
   Expected epochs are `[2, 3]`; closure occurs in the first resumed epoch loop.
3. Resume all three from the **same reference epoch-2 checkpoint**.
   Expected epoch is `[3]`; closure occurs during resume setup.

This is 54 full training trials and 126 epochs across six conditions. It is
lifecycle/parity qualification, not a five-pair throughput experiment. Every
completed epoch must process 5,000 images with finite foreground segmentation
loss and no OOM retry/batch reduction. Compare exact initial weights, every
batch's loss vector and final model state **within each fresh/resume cohort**.
Require all configured epochs and expected lifecycle observations. Do not assume
resumed training equals an uninterrupted run: upstream checkpoints derive from
EMA, serialize optimizer state at reduced precision, and do not establish full
RNG-state continuation. This protocol compares the backends under the same
upstream resume behavior.

For example, the reference fresh trial for one condition is:

```sh
python bench/train_coco_gpu.py \
  --corpus /measurement/coco-val2017-yolo/segment \
  --fresh-check /measurement/fresh-reference.json \
  --out /measurement/lifecycle-reference-w2-yes.json \
  --backend reference --workers 2 --overlap yes \
  --epochs 4 --close-mosaic 2 --persistent-mask --checkpoint-epochs 1 2
```

After that trial completes and its checkpoint/report are checked, one resumed
candidate trial is:

```sh
python bench/train_coco_gpu.py \
  --corpus /measurement/coco-val2017-yolo/segment \
  --fresh-check /measurement/fresh-reference.json \
  --out /measurement/lifecycle-mask-w2-yes-resume-at-boundary.json \
  --backend mask --workers 2 --overlap yes \
  --epochs 4 --close-mosaic 2 --persistent-mask \
  --resume-from /measurement/lifecycle-reference-w2-yes.run/resume-epoch-1.pt \
  --resume-receipt /measurement/lifecycle-reference-w2-yes.json
```

Paths above are placeholders, not existing trial receipts. Run serially on the
reserved host and retain failures. The existing `repeat_coco_gpu.py` coordinator
does not forward these flags or audit resumed cohorts. The
`lifecycle_coco_gpu.py` coordinator passed 63 synthetic tests and has now launched
the grid. Independent final artifact verification remains required before a full
qualification claim.
Single-trial `complete` is insufficient to establish the whole protocol.

Epoch timing includes upstream closure/reset and the small first-batch
observation, and excludes checkpoint saving. Whole-job resource samples include
setup/checkpoint work; summed family RSS double-counts shared pages. No new speed,
memory or accuracy claim is made by preparing this harness.

The current shared-batch revision also observes the actual loader collator and
requires zero exit codes at worker replacement. Native persistent trials install
the shared collator, while the reference trial retains its original collator.
The complete set of these assertions and runtime conditions has not yet passed
the GPU grid; see [the candidate contract](SHARED_BATCHES.md).

The current harness also records the loader's actual multiprocessing
start method, pin-memory setting and prefetch factor. It does not force GPU
training to use the spawned context of the CPU loader experiment. At the first
batch of each epoch, it checks that the image is pinned when pinning is enabled.
At a close-mosaic replacement it retains both old worker PIDs and exit codes in
the lifecycle record, in addition to asserting clean termination and new PIDs.

Resume after the boundary can reset workers during setup, before epoch callbacks.
The harness delegates loader construction to the original factory and retains
only the returned loader's parent Process handles until the pretrain callback.
It then applies the same old-worker exit/PID checks to this setup-time reset and
releases those handles. It does not change loader options, reset or shutdown.
Replacement observations are written before assertions so failures remain visible.

After the unmodified trainer returns, both train and validation loaders must
have the expected worker count, no live workers and zero exit codes. These
observations are retained before assertions, so a shutdown failure remains in
the partial report. No shutdown override is used. The trial now binds both its
own script and `coco_loader.py`, verifies those bytes after training, and requires
the same two-script identity when resuming from a reference receipt. These
source changes alone are not training or reliability results.

## Coordinator synthetic qualification passed; GPU grid in progress

`bench/lifecycle_coco_gpu.py` declares the full plan before launching any trial:
54 fresh processes and 126 total epochs across workers 0/2/8 and both mask modes.
Each condition runs reference, mask-only and combined backends for fresh training,
resume at the closing boundary and resume after it. All resumed backends use the
same retained reference checkpoint for that condition. Each output uses a new
directory. Failures preserve the original raw report, log, return code and hash;
there is no automatic retry or omission of a failed trial.

The coordinator compares complete loss-trace digests, initial/final model digests,
resume identity and loader settings within each three-backend cohort. It also
checks the actual resolved training configuration, full 5,000-image/1,250-batch
epochs, foreground losses, finite values, CUDA peaks and sampled RSS arithmetic.
Formatter/factory/collator identity, pinning, worker continuity, reset history and
final train/validation shutdown must agree with the declared protocol. No
comparison claims that resumed training reproduces uninterrupted training.

Raw loss/resource arrays stay in their hashed trial files; the parent retains
compact summaries so it does not accumulate full traces between trials. State
updates use atomic file replacement. A selected-condition run is explicitly
marked as a subset, even when it completes all its requested trials. Only the
default full grid can set `full_protocol_complete`, and that remains a coordinator
result requiring independent final artifact verification.

All 63 synthetic tests in `tests/test_lifecycle_coordinator.py` passed on Linux,
with no failures, errors or skips. They check the declared plan/checkpoint relationships, all three lifecycle
paths, corruption rejection, cohort comparisons, and failure retention through
a fake child-process driver. They do not execute GPU training or deserialize a
real checkpoint. The wheel workflow now selects these framework-free tests in
every core job. Local actionlint passed; hosted matrix execution remains pending.

Example full-grid invocation after installing qualified wheels and binding the
current original cache to a fresh reference receipt:

```sh
python bench/lifecycle_coco_gpu.py \
  --corpus /measurement/coco-val2017-yolo/segment \
  --fresh-check /measurement/current-fresh-reference.json \
  --out /measurement/lifecycle-grid.json
```

The example paths are placeholders. The actual launched command and frozen
identities are recorded below. Combined runtime qualification and current
fresh-reference binding passed; independent final-grid artifact verification
remains required.

The [queued qualifier](validation/qualify-lifecycle-after-loader-v2.py) waited for
the exact Linux full-loader controller PID/start time and a successful terminal
timing/fresh-reference receipt before running formatting, Ruff, actionlint and
63 synthetic tests. Its [launch identity](validation/mask-lifecycle-coordinator-linux-v2-launch-identity.json)
binds the staged source bytes. It never restarts the measurement and never starts
GPU training. Its [completed receipt](validation/mask-lifecycle-coordinator-linux-v2-qualification.json)
records all five commands returning zero, including 63 synthetic tests. The
[21-member archive](../bench/results/mask-lifecycle-coordinator-linux-v2-evidence.tar.gz)
preserves original and formatted source, logs, JUnit, watcher and dependency
identities. All members passed readback; see the
[preservation receipt](validation/mask-lifecycle-coordinator-linux-v2-preservation.json).
Formatted Python files were copied back only after before/after hash and AST
equivalence checks. This remains synthetic qualification, not actual GPU training.
The previous idle watcher was
[cancelled before any check ran](validation/mask-lifecycle-coordinator-linux-v1-cancelled.json)
to add a signal mock to the fake-child tests; its staged source remains preserved.

## Actual Linux launch, 2026-09-13

[The launcher](validation/launch-lifecycle-gpu-linux-v1.py) verified the completed
combined-runtime qualification, frozen source hashes, wheel payload hashes, the
full corpus fingerprint and Linux `file_descriptor` sharing. No other GPU compute
process was present at launch. It reuses the last successful 30-run original
reference verification because the current cache bytes and source identities
still match, preserving a separate copy of that cache and the exact receipt.
It does not regenerate or relabel the old verification as a new benchmark.

The executed coordinator command selects worker order `2 0 8` and overlaps
`yes no`, covering the same full six-condition grid. Every condition still has
three backends and three lifecycle stages, for 54 trials and 126 epochs. Failures
stop the series and retain their first artifacts; there is no retry or automatic
batch reduction. This is lifecycle/cohort qualification, not a repeated speed
comparison or held-out accuracy experiment.

The dataset runtime is `dataset-notices-linux-v2-clean`; the frozen harness source
is `mask-lifecycle-gpu-linux-v1-source`, from mask commit
`135d0e428dfb55207f7cc1ece0acd033eb952a09`. Its
[source manifest](validation/mask-lifecycle-gpu-linux-v1-source.json) binds the
[12-file archive](../bench/results/mask-lifecycle-gpu-linux-v1-source.tar), SHA-256
`09c94a6ff6f2e1d5dd83c422092c41f59ff239fa0cdfe80d4196ca9531a0aba1`.
The [launch checkpoint](validation/mask-lifecycle-gpu-linux-v1-launch-identity.json)
records live launcher/coordinator/trial PIDs, exact commands, preflight results
and the qualified wheel receipt. This is an initial live observation, not final
status; re-check the remote grid before claiming completion.

Authoritative ongoing state on the server is
`/home/yonghye/ultrafast-vision-build/mask-lifecycle-gpu-linux-v1-grid.json`.
Its per-trial logs/reports/checkpoints are under the sibling `.runs` directory.
All 54 outcomes and the independent final audit remain pending at this checkpoint.
