# Shared batch candidate: qualification pending

The candidate adds guarded CPU collation to the optional Ultralytics adapter.
It is motivated by the [observed spawned-worker abort](WORKER_SHUTDOWN_DIAGNOSIS.md)
and the extra heap-to-shared-storage copy performed when queue serialization
receives ordinary batch tensors. The first installed shared-collator revision
failed qualification (253 passed, one worker-reset failure); see the
[failure and second debugger trace](SHARED_LINUX_VALIDATION.md). The current
packet revision passed all 257 installed Linux tests and 18 fresh-process reset
repetitions. See the [bounded qualification evidence](SHARED_PACKET_VALIDATION.md).
Real-data performance, training/resume and other platforms remain unqualified.

`accelerate_dataset(dataset, persistent=True)` now installs both the instance's
FastFormat factory and its shared collator before workers are constructed.
The default one-shot `accelerate_dataset(dataset)` remains unchanged. The
separate `share_dataset_batches(dataset)` entry point in
`ultrafast_maskops.ultralytics` installs only the collator and returns 1 when
changed, or 0 when already installed. It can be used without mask replacement.
Neither entry point changes class/module globals, trainer methods or shutdown.
Callers must let the loader use `dataset.collate_fn`; an explicitly supplied
class-level reference collator bypasses the instance hook.

## Allocation and compatibility contract

In a DataLoader worker, homogeneous contiguous CPU stack/cat fields are written
directly into freshly owned shared storage. The pattern matches PyTorch's default
tensor collator's shared output allocation, extended to the pinned YOLO batch
contract. No output pool is reused while previous batches may remain queued or
alive in the consumer. Zero-worker calls delegate to the original collator.

After collation, the worker serializes the batch using `ForkingPickler`, whose
registered Torch reducers encode shared-storage handles. An internal packet
holds only the resulting bytes. The ordinary queue receiver reconstructs the
original dict before DataLoader pinning or delivery to the consumer. Tensor
serialization and destruction no longer belong to the queue feeder's payload
lifetime. This is a transport change, with metadata serialization overhead;
it does not serialize a copy of the image/mask pixels. Standard trusted local
multiprocessing semantics apply. No externally supplied pickle input is added.

Mixed dtypes, noncontiguous or channel-last inputs, and autograd inputs use the
original Torch operation to preserve promotion/layout/error behavior, followed
by eager sharing. Padded visuals and tensor-valued metadata also take eager
sharing; this fallback still copies and is not a zero-copy claim. The adapter
supports dense strided CPU tensors. Custom collators and non-CPU/sparse/nested
tensor outputs require their own integration. Core mask APIs retain their
NumPy-only dependency boundary; Torch is imported only by framework integration.

PyTorch 2.10.0 and the exact original YOLODataset.collate_fn source are guarded.
The runtime source hash is
`124bfcaa17398abe7749d85f6a63d0303bedbc1400184005ac772621f6c9a420`.
The instance hook is module-level and pickleable. Persistent opt-in validates
the collator before changing its formatter/factory, rejecting custom hooks.

## Required evidence

The new tests independently compare the collator with the reference for values,
dtypes, shapes, strides, key order, batch-index input mutation, dtype promotion,
empty annotations, overlap/non-overlap masks, nested metadata and invalid-shape
errors. They retain a batch while creating/mutating another to check ownership,
and verify common output fields are already shared before final IPC preparation.

The persistent lifecycle tests now cover reset before consumption, after a
first batch and after a full epoch, with workers 0/2 and both mask modes. They
check zero exit codes for old and replacement workers. Their reference side
calls the original collator, eagerly shares its outputs, and uses a separate
test-only bytes packet with the standard Torch reducers. This normalizes
reference transport for **functional parity only**;
the original unmodified-reference failure remains preserved. It is not a speed
comparison against an unchanged full reference loader.

The GPU trial harness records and asserts the actual loader collator. Its
reference backend keeps the original collator; native persistent backends use
the new instance hook. Old GPU evidence cannot qualify this change. Fresh wheel
auditing, all installed tests and the declared 18-case spawn stress passed on
one Linux profile. Real-data batches, actual training/resume and supported-platform
qualification remain required.

There is no measured speed or memory improvement for this implementation yet.
Measure shared allocation and full-loader/epoch behavior before choosing a
default for broader usage. Removing a copy does not by itself establish an
end-to-end gain or a bound on total process-family RSS.
