# Named shared-memory lifetime: Linux failure observed

The current Linux packet wheel is **not qualified for `file_system` transport**.
On Linux / CPython 3.12.14 / PyTorch 2.10.0+cu128, both the original Ultralytics
collator and packet collator left named storage alive after repeated resets.
The parent processes then failed to exit naturally. A stock PyTorch DataLoader
control reproduced the exit hang without importing either project or Ultralytics;
the otherwise identical `file_descriptor` control exited normally.
These observations do not establish macOS behavior or a candidate-specific leak.
The subsequent [M2 installed suite](M2_PACKET_VALIDATION.md) passed 366 tests and
exited naturally using macOS's default `file_system` transport. It is bounded
functional evidence, not a named-storage lifetime measurement.

## Actual observations

The [diagnostic](../bench/shared_transport_lifetime.py) used two spawned workers,
batch size 4, prefetch factor 2, no pinning, and 16 synthetic 128-pixel image
samples with variable instance counts. Each requested cycle consumes and checks
one batch, then calls the actual pinned `InfiniteDataLoader.reset()`. No worker
shutdown method or reference collator was replaced. The installed mask wheel is
the [previously qualified Linux packet wheel](SHARED_PACKET_VALIDATION.md).

| Case | Completed resets / requested | Observed names still present 3 s after close | Named payload bytes | Actual process outcome |
| --- | ---: | ---: | ---: | --- |
| Original collator, initial direct run | 6 / 6 | 76 | 2,868,128 | Hung after successful diagnostic body; manually terminated |
| Packet collator, supervised | 6 / 6 | 80 | 3,275,680 | Body passed; exit exceeded 20 s; supervisor sent SIGTERM |
| Original collator, same supervisor | 3 / 6 | 52 | 2,041,872 | Worker SIGABRT during fourth reset, then parent exit hang |

All consumed batches matched the original collator's values, dtypes, shapes,
strides, container types and field order. The initial reference and packet cases
observed zero exit codes for all reset and final workers. The supervised reference
case explicitly failed: worker 1649603 exited with -6. It must not be reported as
a completed six-reset control or omitted from a comparison.

In the initial reference and packet cases, cumulative named payload bytes grew
from 613,928 after the first reset to the final values above. These are small,
instrumented, scheduling-sensitive observations, **not physical RSS, performance
measurements, or a quantified regression between collators**. Only names recorded
by PyTorch's storage reducer were probed read-only; the diagnostic never unlinked
memory. All traced names were absent after the respective parents were terminated,
and the related managers/resource trackers subsequently exited.

## Isolating the exit failure

The [stock PyTorch reproducer](../bench/reproduce_torch_filesystem_exit.py) uses
one complete ordinary DataLoader epoch, two spawned workers, four verified
batches, and no resets, private shutdown calls or reducer instrumentation.
Both workers exited with code 0 in each control:

| Stock PyTorch transport | Body | Natural process exit |
| --- | --- | --- |
| `file_system` | Passed | Hung; terminated after 20 s |
| `file_descriptor` | Passed | Code 0, no timeout or signal |

The descriptor control changes only the sharing-strategy literal and associated
text in the filesystem reproducer; the exact generated source and diff are in
the archive. These are one-run causal controls, not broad platform qualification.

Faulthandler captured the stuck main thread in
`multiprocessing.resource_tracker.ResourceTracker._stop_locked`, called by
`__del__`, waiting for the tracker process. Read-only `/proc` snapshots showed
that both stock-case `torch_shm_manager` processes retained a write descriptor
for the exact pipe the tracker was reading. The parent still held manager
connections. This supports a shutdown dependency cycle in this runtime.

The public PyTorch 2.10.0 [manager launcher](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/lib/libshm/core.cpp)
forks and execs the manager, while the [manager loop](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/lib/libshm/manager.cpp)
waits for client connections to disappear before final cleanup. Tagged source
copies support the explanation; they are not proof of the wheel's complete
binary compilation provenance. The exact installed Python tracker source and
runtime diagnostic sources are also preserved. No global runtime workaround was
installed, and this does not prove a fix for the separate worker abort.

## Evidence quality and retained failures

The [process supervisor](../bench/observe_transport_process.py) requires an actual
child return code in addition to the diagnostic body's `complete` and
`protocol_passed` fields. A successful body is insufficient. On an exit timeout,
it captures process/descriptor state and a SIGUSR1 faulthandler stack, then stops
only its child. It does not signal managers/trackers or alter queue shutdown.

The first unsupervised process required manual termination. Attaching GDB was
denied by ptrace policy; that log is retained. The manual cleanup observer sent
SIGTERM successfully, then hit a permission race while reading `/proc` during
exit. A separate read-only follow-up confirmed terminal processes and absent
handles. That observer failure is preserved rather than labeled successful.

The first summary check also exposed a diagnostic defect: historical
`worker_pids` fields referenced a mutable list and grew as later workers were
appended. Raw reports are unchanged. The corrected summary reconstructs each
probe's scope from recorded retired workers and verifies the frozen per-probe
trace hashes and payload totals. It explicitly flags the alias defect. Current
source copies the PID list; an isolated execution of the actual probe function
reproduces the original bug and verifies the fix. This is not a new complete
loader run of the corrected source.

The [74-member evidence archive](../bench/results/mask-storage-lifetime-linux-v1-evidence.tar.gz)
contains all five runs, JSONL traces, termination/proc observations, failures,
exact executed sources, stock controls, upstream source copies and the frozen
runtime's wheel qualification archive. Its 1,758,577 bytes have SHA-256
`39e4fbba5a9ab16116894a101836efe072514960b8861d6859ec168d2ba0dd3d`.
Every member was read back and hashed; see the
[preservation manifest](validation/mask-storage-lifetime-linux-v1-preservation.json),
[loader observation summary](validation/mask-storage-lifetime-linux-v1-summary.json),
[stock control checks](validation/mask-storage-lifetime-stock-controls.json), and
[snapshot regression](validation/mask-storage-lifetime-snapshot-regression.json).

## Remaining scope

Do not promote Linux `file_system` transport or macOS named-storage lifetime as qualified.
Setup-only and full-epoch repeated-reset comparisons remain unexecuted; the
full-epoch stock control above is not that matrix. Fix or explicitly resolve the
runtime exit dependency and storage lifetime before extending that transport.
The successful Linux `file_descriptor` control does not substitute for actual
close-mosaic/resume GPU training, current combined-wheel validation, or supported
platform/wheel qualification. Those remain separate gates.
