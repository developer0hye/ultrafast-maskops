# Worker shutdown diagnosis and shared-memory control

The [failed persistent-candidate qualification](PERSISTENT_LINUX_VALIDATION.md)
now has a native debugger trace and a controlled reference-only experiment.
The abort occurs while Python finalizes a spawned worker and a queue feeder
thread returns from PyTorch storage sharing. Preparing shared memory on the
worker's main execution path prevented the failure in six diagnostic processes,
with exact post-reset batches and clean worker exit codes. **This is not yet a
production fix or a throughput/memory result.** The original 221/222 suite outcome
remains a failed qualification.

## Observed abort path

GDB followed the original reference-only reproducer and its spawned children.
The aborting thread's stack includes, from caller to termination:

```text
pickle / queue feeder
THPStorage_shareFd
PyEval_RestoreThread
take_gil
PyThread_exit_thread
pthread_exit / forced unwind
std::terminate / SIGABRT
```

At the same stop, the worker's main thread was inside `Py_FinalizeEx` and garbage
collection. Neither new library was imported. The GDB run is diagnostic only:
its scheduling differs, it stopped the inferiors at the abort, and it did not
produce a completed reference report. Its command, return status and full trace
are retained; the debugger and traced processes were confirmed terminated.

The pinned [PyTorch storage-sharing source](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/StorageSharing.cpp#L188)
releases the GIL while copying newly shared CPU storage and reacquires it on
return. Already-shared storage takes the handle-return path. The pinned
[CPython GIL implementation](https://github.com/python/cpython/blob/v3.12.14/Python/ceval_gil.c)
contains the finalization/thread-exit path seen in the trace. Together with the
control below, this supports the diagnosis of overlapping interpreter shutdown
and deferred shared-storage preparation. It does not prove every similar
SIGABRT has this cause.

## Executed differential control

The [v2 diagnostic](validation/reproduce-reference-reset-v2.py) retains the
reference YOLODataset, Format and real close/reset methods. Its optional collator
first calls the original `YOLODataset.collate_fn`, then calls `share_memory_()`
on the resulting batch tensors before returning them to the queue. This moves
the storage-copy work onto the worker's main path. No exception suppression,
sleep, retry, worker-count reduction, fork substitution or shutdown override is
used. Both new libraries remain unimported, including in spawned workers.

Seven fresh processes ran serially on the Linux host with the same installed
environment, two spawned workers and four-image fixture:

| Before reset | Shared preparation | Processes | Result |
|---|---|---:|---|
| Full epoch | Original deferred sharing | 1 | Reference worker SIGABRT |
| Full epoch | Eager sharing | 3 | All passed |
| First batch | Eager sharing | 2 | All passed |
| Setup, before consumption | Eager sharing | 1 | Passed |

Every eager-sharing process compared both post-reset batches against the
reference collator in the parent, including tensor dtypes/values and other batch
fields. All four worker exit codes per process were zero: two old workers after
reset and two replacement workers after final close. A returned reset alone is
insufficient; the earlier control returned despite two `-6` exits.

This small synthetic control does not establish general reliability, actual
training parity, performance or platform coverage. Its extra copy is deliberate
diagnostic instrumentation and must not be advertised as an optimization.

## Next implementation and qualification

Evaluate a profile-guarded instance collation hook that builds standard CPU
stack/cat outputs directly in shared storage. That can address the observed
shutdown path while removing the separate heap-to-shared-memory copy, but both
effects must be measured. Keep the zero-worker reference path and validate dtype
promotion, layout, empty annotations, ordering, ownership and the reference's
batch-index mutation behavior. Unsupported/custom collation must be explicit.
Core standalone mask APIs should retain their NumPy-only dependency boundary.

Preserve the original reference failure. Any functional lifecycle test that
normalizes transport on the reference side must say so, and must independently
compare the new collator against the original output. Do not relabel it as
unmodified-reference throughput evidence. The actual training benchmark must
record exactly which collator each backend uses. Clean exit checks, repeated
spawn lifecycle cases, real-data batches, full installed tests and the declared
training/resume grid remain required before a production claim.

## Retained evidence

The [19-file archive](../bench/results/mask-persistent-linux-v1-shutdown-diagnosis-evidence.tar.gz)
contains GDB command/log, exact v2 script, all seven raw report/log pairs and the
series record. It embeds the previous qualification archive unchanged, binding
the wheel, environment, original failures and v1 controls. The
[manifest](validation/mask-persistent-linux-v1-shutdown-diagnosis-preservation.json)
records all hashes. Archive size is 1,646,215 bytes and SHA-256 is
`e2d469087ea6e1338ebd186a48eb2b1708b72c19aac8ad14031dcd2a443631b4`.
All archive members and all seven reported outcomes were independently checked
after download. No candidate runtime code was changed by these experiments.
