# Persistent adapter Linux qualification: incomplete

The fresh candidate wheel builds and passes its isolated installation checks,
but the full installed suite **failed: 221 passed, 1 failed, 0 skipped** in
31.92 seconds. The failing test exercises a real spawned-worker reset. A focused
rerun failed before native opt-in (`native=False`), and an independent script
that never imports either new library reproduced reference worker failures.
This is a qualification failure, not a passing candidate or a training result.

## Build and isolated installation

Selected source commit: `d58158009bc52166f8a82c1912621e76f7ddd386`.
The 76-file source selection excludes earlier results and validation artifacts;
its complete member hashes and original tar are retained. C++ and runtime Python
bytes match this selection. The new diagnostic script was written afterward and
is separately included in the evidence; it is not part of the built wheel.

Linux x86_64, CPython 3.12.14, GNU 13.3.0, private OpenCV 4.13.0 and Release
configuration were used, with two compile jobs. Wheel SHA-256:
`aff43a9d54b1f73b83b19c199bfcc2ef624b5465b7404aa7902b8158d11835c8`.
Extension SHA-256:
`6727e6b1f1ac4c83b4f6fd90aba49f0da8e9a1fd85fb5a3b7aa4ca11d738c555`.
Source archive SHA-256:
`ac98321528b1cbd1d86d16bd6ecfefb5ca4db88740a90123d5a9f61f3eb4f453`.

A new venv disables system site packages. With NumPy 2.4.4 and the wheel installed,
OpenCV's Python package and Ultralytics were absent. Standalone overlap output,
wheel RECORD and installed-byte checks passed. All nine source notice files
matched the source distribution, wheel and installed files. This does not prove
the supported ABI/platform matrix or completeness of all license obligations.
See the [standalone report](validation/mask-persistent-linux-v1-standalone-wheel-validation.json).

The existing 81-entry Linux environment freeze was then installed with its old
mask wheel excluded. All 79 other non-pip dependency entries match after PEP 503
name normalization, including exact versions/direct URLs; pip was explicitly
pinned to 26.2.1. `pip check` passed. Installed package Python bytes match the
selected source. The prior environment and measured datasets were not modified.

## Executed test coverage

| Test module | Passed | Failed |
|---|---:|---:|
| Kernel parity | 120 | 0 |
| ROI and resize | 48 | 0 |
| Framework integration | 15 | 0 |
| Training tests | 4 | 0 |
| Wheel comparison helper | 10 | 0 |
| GPU-series artifact auditor | 12 | 0 |
| Persistent Format | 12 | 1 |

The training tests are the existing small test cases, not the declared full
5,000-image GPU lifecycle grid. The failing case is
`test_close_mosaic_retains_format_through_real_worker_reset[True-2]`.
It aborts inside the pinned `InfiniteDataLoader.reset → close →
_shutdown_workers` path. The focused rerun with local variables shows
`native=False`; the native adapter branch had not run yet.

The [reference-only reproducer](validation/reproduce-reference-reset-v1.py)
uses the same four-image fixture, augmentations and two spawned workers. It
asserts that neither new library is imported, including in each spawned worker.
One fresh process was run for each condition:

| Reset point | Observed result |
|---|---|
| After first batch | Reset returned, but both old workers exited with `-6` (SIGABRT); diagnostic failed |
| After full epoch | Reset raised a worker SIGABRT error; diagnostic failed |
| During setup before consumption | Reset and subsequent epoch passed once; old worker exit codes were zero |

These observations show that changing the test to consume a full epoch alone
does not resolve the failure. They do not identify the underlying C++ race or
prove all reference/native environments behave this way. The existing test's
dead-worker/PID checks also cannot certify a clean exit when reset returns after
an abort; future lifecycle qualification must check exit codes explicitly.

A [Lightning issue](https://github.com/Lightning-AI/pytorch-lightning/issues/21703)
reports a similar failure with explicit PyTorch worker shutdown and spawn. That
is investigation context, not proof of an identical root cause here. No upstream
code, reset behavior, worker mode or test expectations were changed to obtain a
pass, and no exception was suppressed in the qualification suite. The diagnostic
captures exceptions only to retain a failed report and returns a nonzero status.

## Evidence and remaining work

The [35-file archive](../bench/results/mask-persistent-linux-v1-qualification-evidence.tar.gz)
retains exact source/wheel/sdist, build and installation logs, environment pins,
original suite XML/log, focused failing rerun, independent diagnostic source and
all three outcomes. Its [manifest](validation/mask-persistent-linux-v1-qualification-preservation.json)
records every member. Archive size is 1,635,831 bytes; SHA-256:
`752017b7531333650cfebb0388c42a7ae20a2e3cca284caba7a16643a22e1c1a`.
All members were read back and independently hash-checked after download.
The [test command/result](validation/mask-persistent-linux-v1-tests.json) and
[full XML](validation/mask-persistent-linux-v1-tests.xml) are also directly available.

Resolve and verify worker shutdown before claiming persistent lifecycle support.
Then complete the installed suite, actual combined-library close-mosaic/resume
training, and broader platform qualification. Existing kernel test successes do
not substitute for those gates. M2's separate dataset-startup campaign continues;
this diagnosis does not modify or qualify its frozen runtime.
