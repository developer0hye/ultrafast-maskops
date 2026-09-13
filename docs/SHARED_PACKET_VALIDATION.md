# Shared packet: bounded Linux checks passed

The bytes-packet revision at `5f6fc8e4c9da94e33cfd079f35fe3e3f8bec7d52`
passed **257 installed tests, zero failures/skips, in 51.79 seconds**, then
**18 fresh-process spawned-worker reset tests**. These results qualify the
tested Linux functional scope. They do not establish production readiness,
new speed/memory gains, other-platform support or full training/resume parity.
The [preceding 253/254 failure](SHARED_LINUX_VALIDATION.md) remains preserved.

## Exact environment and scope

A fresh wheel/sdist build used CPython 3.12.14 on the Linux RTX 3070 host. A new
venv first installed only the wheel, NumPy 2.4.4 and pip 26.2.1. The standalone
mask smoke result, wheel RECORD/installed-byte audit and all nine bundled notice
comparisons against source/wheel/sdist passed without cv2 or Ultralytics installed.

The pinned framework environment then installed successfully and passed
`pip check`. All 79 normalized non-mask/non-pip dependency entries matched the
preceding environment, including Torch 2.10.0+cu128 and the exact Ultralytics
source. All selected source files and installed runtime Python files matched
their frozen hashes. GPU kernels were not exercised by this qualification.

The full suite includes 26 shared-collation cases, testing exact batch fields,
ownership, invalid/fallback behavior and the new packet's round-trip/shared-handle
semantics. The packet carries handles: changes through a reconstructed image
tensor remain visible through the original shared tensor.

The subsequent stress series declared three fresh-process repetitions of each
reset point (setup, first batch, full epoch) and mask overlap mode, with two
spawned workers and the four-image fixture. All 18 processes completed with exact
post-reset reference/native batch equality and zero exit codes for old and
replacement workers. The reference computation uses the original collator with
test-only normalized packet transport. This is functional parity, not unchanged
reference-loader throughput or a general statistical reliability guarantee.

No failed run was retried into success. The suite and all 18 declared repetitions
passed on their first runs. The older eager-sharing-only failures and passing
rerun remain distinct evidence for their original wheel.

## Retained evidence and remaining work

The [60-file archive](../bench/results/mask-shared-linux-v2-qualification-evidence.tar.gz)
includes exact selected source, wheel/sdist, build, clean environment, standalone
audit, full suite, every stress log/XML and the qualification/stress/collector
scripts. Its [manifest](validation/mask-shared-linux-v2-qualification-preservation.json)
binds every member. All member bytes and SHA-256 values were checked again after
download. Archive size is 1,668,875 bytes, SHA-256
`ecd5ebadb94f4c3dfcdeaa616f57c6914737a4b0bca0e80abe9ed2a33cc6701b`.

Wheel SHA-256:
`8b467c837c323812a0201b81210a6dcd0c180ef50ba0d16531f07ef2c741cd9a`.
Source tar SHA-256:
`5793b268fa28985ac278cd3a9a7f6a71ed8cffba6250db3cc5f5305ac8314783`.

Next gates are the [real-data loader protocol](SHARED_LOADER_PROTOCOL.md), actual
GPU training across close-mosaic and resume, combined dataset/mask integration,
resource measurements and supported-platform qualification. In particular,
eight-worker operation, GPU pinning and the declared 54-trial training grid are
not proven by these two-worker CPU reset tests. A subsequent
[full-image loader pilot](SHARED_LOADER_PILOT.md) now verifies non-augmented
CPU loading at workers 0/2/8; GPU pinning and actual lifecycle training remain open.

The subsequent [full repeated comparison](SHARED_LOADER_FULL_RESULTS.md) passed
all 60 measured processes, two fresh-reference verifications and independent
artifact qualification. Eight-worker speed gains were small; the 10% loader
improvement gate and actual GPU lifecycle grid remain open.

The subsequent [named-storage lifetime diagnostic](SHARED_TRANSPORT_LIFETIME.md)
found retained handles and parent exit hangs under Linux `file_system` transport.
A stock PyTorch control reproduced the exit hang without either project or
Ultralytics, while its `file_descriptor` control exited normally. A supervised
original-collator reset also aborted a worker. All failures are preserved;
Linux `file_system` and macOS named-storage lifetime remain unqualified. These
small instrumented observations do not quantify a candidate-specific leak.

The current source subsequently passed a fresh [M2 wheel qualification](M2_PACKET_VALIDATION.md):
366 installed tests, zero failures/skips and natural process exit on macOS
26.6.2 with its default `file_system` transport. This closes the bounded installed
functional check on that host. Named-storage lifetime, repeated process stress,
other platform/Python versions and full real-data training remain separate gates.
