# GPU lifecycle qualification: prepared, not executed

The extended `bench/train_coco_gpu.py` source parses and passes lint, but its new
flags and lifecycle assertions have not executed. The persistent candidate built
on Linux, but [installed qualification failed](PERSISTENT_LINUX_VALIDATION.md)
at a reference spawned-worker reset. The later packet revision passed
[257 installed tests and 18 fresh-process reset cases](SHARED_PACKET_VALIDATION.md).
Those checks do not execute or qualify this training protocol. Earlier GPU
results used another installed adapter, `close_mosaic=0` and two epochs; they
cannot qualify this change. No new job is automatically queued.

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
does not forward these flags or audit resumed cohorts; a lifecycle coordinator
and independent final artifact audit remain to be qualified before a full grid
claim. Single-trial `complete` is insufficient to establish the whole protocol.

Epoch timing includes upstream closure/reset and the small first-batch
observation, and excludes checkpoint saving. Whole-job resource samples include
setup/checkpoint work; summed family RSS double-counts shared pages. No new speed,
memory or accuracy claim is made by preparing this harness.

The current shared-batch revision also observes the actual loader collator and
requires zero exit codes at worker replacement. Native persistent trials install
the shared collator, while the reference trial retains its original collator.
These new assertions and the new runtime have not yet executed in the full GPU
grid; see [the candidate contract](SHARED_BATCHES.md).

The latest unexecuted harness also records the loader's actual multiprocessing
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
source changes are not new training or reliability results.
