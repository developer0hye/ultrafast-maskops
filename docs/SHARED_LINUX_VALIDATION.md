# Shared collator v1: failed Linux qualification

Source `8e60554097cdba94a834b64730a1d2440d190d2b` built a fresh wheel and
sdist. A new CPython 3.12.14 environment passed the NumPy-only standalone audit,
installed-byte/RECORD checks and all nine source/wheel/sdist notice comparisons.
After installing the pinned framework dependencies, `pip check` passed and all
79 non-mask/non-pip dependency entries matched the preceding environment.

The installed suite then reported **253 passed, one failed, zero skipped** in
50.72 seconds. All 23 shared-collator unit cases passed. The failure was
`test_close_mosaic_retains_format_through_real_worker_reset[first-batch-True-2]`:
reset returned, but an old worker had a nonzero exit code and stderr reported
`terminate called without an active exception`. The zero-exit assertion remains
mandatory. A single focused rerun passed; it does not replace the failed suite.

## Second observed shutdown path

Three bounded, fresh GDB attempts followed the exact failing case and spawned
children. The first two logs report a pass; no JUnit files were produced under
the debugger. The third stopped at SIGABRT. Its aborting queue-feeder thread
contained this caller-to-termination path:

```text
tupledealloc / dict_dealloc / subtype_dealloc
THPVariable_dealloc / THPVariable_clear
PyEval_RestoreThread / take_gil
PyThread_exit_thread / pthread_exit / forced unwind
std::terminate / SIGABRT
```

The worker main thread was in `Py_FinalizeEx` and garbage collection at the same
stop. Unlike the [first trace](WORKER_SHUTDOWN_DIAGNOSIS.md), this path concerns
Tensor destruction, not first-time storage sharing. Eager sharing alone leaves
Tensor references in the queue feeder's payload and is insufficient in this
environment. GDB changes scheduling and its stopped run is diagnostic evidence,
not a completed functional test or a universal root-cause proof.

## Current revision, still unqualified

The new worker collator serializes the shared batch into an internal bytes-only
packet on the worker's main path. Torch's normal `ForkingPickler` reducers carry
shared-storage handles; receiver unpickling returns the ordinary dict. The
queue feeder does not own Tensor payload references. No trainer/worker shutdown
method, exception handling, worker mode or retry policy is changed.

Three new tests check packet round-trip batch parity for both mask modes and
shared-handle semantics: payload size is small relative to image bytes, and
mutations remain visible across original and reconstructed tensors. The existing
real spawned-loader cases exercise actual IPC and strict exit checks. Their
reference collation math is unchanged, but test-only transport now uses its own
packet; these tests are functional parity, not unmodified-reference throughput.

Fresh installed tests, repeated spawned-worker stress, real-data loader/CPU and
GPU lifecycle/resume checks, resource measurements and platform qualification
remain required. No new performance or reliability claim follows from source
changes or passing historical controls.

## Preserved artifacts

The [29-file evidence archive](../bench/results/mask-shared-linux-v1-qualification-evidence.tar.gz)
contains exact source, build outputs, wheel/sdist, clean-environment and standalone
receipts, original failed suite, separate passing rerun and all three GDB logs.
The [preservation manifest](validation/mask-shared-linux-v1-qualification-preservation.json)
records every member's byte count and SHA-256. All members were read back and
matched independently after download. The archive is 1,659,129 bytes with SHA-256
`dbc13e216cb0e8227b1a0b1dc52db20811f4fc97e4a74e576d7b42cc47575e51`.

Wheel SHA-256 is
`67823a4764f910f7eb9fbd2f319b35313e8f6fed59ea2e1668bacf28b5724c20`;
source tar SHA-256 is
`45cb2f5b4499edc71db4621ba786c7fcdc0afdb33c293d49efe6746bceaad794`.
