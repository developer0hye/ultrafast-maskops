# Named shared-memory lifetime: diagnostic prepared, not executed

The packet wheel's Linux qualification and CPU loader measurements do not prove
memory reclamation on every supported transport/platform. The pinned PyTorch
`reduce_storage` implementation increments the shared reference count for
`file_system` transport, while `rebuild_storage_filename` decrements it when a
receiver reconstructs the storage. Queued batches can be abandoned during a
loader reset. Whether their names persist while the parent remains alive must
be measured; this source observation alone does not establish a leak or a
candidate-specific regression.

The new [diagnostic](../bench/shared_transport_lifetime.py) compares the original
Ultralytics collator with the packet collator in separate fresh processes. It
uses the actual pinned `InfiniteDataLoader.reset()` and `close()` methods, two
spawned workers, prefetch factor 2, synthetic 128-pixel image tensors and variable
instance counts. `file_system` is selected explicitly inside every worker.
No worker shutdown method or reference transport is replaced.

A reducer wrapper records only handles returned by PyTorch's original storage
reducer. After each reset, the parent opens those exact names read-only using
`shm_open`, closes the descriptors, and records which names still exist. It never
enumerates unrelated shared-memory objects or unlinks anything. Records include
source hashes, original/replacement worker PIDs and exit codes, traced handle
sizes, and cumulative observations over six resets. Consumed batches must match
the original collator's values, dtypes, shapes, strides and field order. Full
epochs must contain four batches; first-batch cases must consume exactly one.

Run setup-only, first-batch and full-epoch reset points independently for each
backend. Every invocation needs a new output name; keep failures and all JSONL
traces. Setup-only cases consume no batch but wait up to 30 seconds for a storage
trace from each worker, so an empty import-time shutdown cannot count as an
abandoned-prefetch observation. Both immediate and three-second-after-close
observations are retained.
`protocol_passed` means the requested diagnostic completed with parity and clean
worker exits. It deliberately does not mean no surviving handles, bounded memory,
or production qualification. Handle payload sizes are not physical RSS, and
instrumented execution is not a performance benchmark.

The script is not yet linted or executed. Existing measurements must terminate
before qualification or execution on either host. The Linux lifecycle qualifier
already queued behind its CPU benchmark must also finish before another job.
Start with one reference/native pair per reset point; preserve and investigate
worker failures before any repetition. Linux `file_system` observations do not
substitute for an actual macOS run. This isolates batch transport; full-image
FastFormat, close-mosaic training and resume protocols remain separate gates.
