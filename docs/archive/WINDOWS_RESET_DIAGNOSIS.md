# Windows worker-reset qualification remains open

The first hosted wheel matrix at `06a26d106ba8dd896061f987a6561521de68e6e1`
failed on Windows 2022 / CPython 3.13.15. The installed native/core checks passed
289 tests. Integration passed 66 tests and failed
`test_close_mosaic_retains_format_through_real_worker_reset[setup-True-2]`:
a previous worker had a nonzero exit code immediately after the unchanged
Ultralytics loader reset. This was the setup-before-first-batch, overlap case.
The original assertion did not record the backend or actual exit codes, so the
log alone cannot establish whether the reference packet or native path failed,
nor distinguish a timeout termination from a worker exception.

[Original run](https://github.com/developer0hye/ultrafast-maskops/actions/runs/34744592119)
and [preserved failure receipt](validation/mask-hosted-windows313-v1-failure.json)
retain this adverse result. The full matrix subsequently ended with **11 successful jobs and this one
failed job**. All 12 terminal job logs and API records have been preserved in
the [terminal matrix receipt](validation/mask-hosted-matrix-v1-terminal.json). The small evidence archive retains the complete failed job log,
job/artifact metadata, and exact original test/workflow sources. It does not
back up the original 116.5MB wheel/sdist artifact.

## Predeclared diagnosis

[Diagnostic run 34746107477](https://github.com/developer0hye/ultrafast-maskops/actions/runs/34746107477)
uses the **same original Windows wheel**, downloaded on its hosted runner after
checking the original artifact ID, run, source SHA and ZIP digest. Installed file
bytes are checked against its wheel RECORD. The current project source is used
only for the diagnostic harness and assertion messages; the extension is not
rebuilt. Both a copy of the wheel and original JUnit reports are retained.

`bench/windows_reset_probe.py` schedules 27 cases: stock collator, reference
packet transport, and native packet transport; setup, first batch, and full epoch
reset points; three repetitions each. All use two spawn workers and overlap masks.
It records old/new process IDs, liveness, exit codes, and calls to join/terminate.
The wrappers forward the original arguments and deadlines, with diagnostic
logging overhead. Trainer, reset, shutdown, and multiprocessing sharing policies
are unchanged. A nonzero exit remains a failure. All cases run without retries;
every failure is uploaded. This experiment covers Windows overlap reset diagnosis,
not general storage-lifetime qualification or performance.

The complete original 67-case integration suite is then run even if the diagnostic
probe fails, with the original exit-code assertion augmented by backend and PID
context. A passing repetition will not erase the failed first matrix or prove the
intermittent issue resolved. Root cause and any correction remain to be established.

The first diagnostic dispatch, run `34746032038` at `a1a2cee`, skipped every
job because its guard required both `windows_reset` and `!windows_reset`. No
test ran. This dispatch is retained as a workflow error, and the contradictory
condition is removed before the next dispatch.

The corrected dispatch at `2a22316` selects exactly one intended job for each
of the 12 supported push/pull_request/manual mode combinations checked directly
from the workflow expressions. It completed all 27 diagnostic cases and all 67 integration cases without
failures or skips. The [readback receipt](validation/mask-windows-reset-v2-preservation.json)
checks the exact original wheel, installed payload, all 27 records, and both
JUnit reports. All 108 recorded joins used the original 5-second timeout; the
maximum observed join was 2.481 seconds and none called terminate. The original
failed integration JUnit is retained alongside the passing repetition. These
observations do not identify the first failure or establish a correction.


## Slow-initialization control

[Run 34746468696](https://github.com/developer0hye/ultrafast-maskops/actions/runs/34746468696)
uses the same original failed wheel and schedules six additional cases: the three
collators with ordinary initialization or a deliberately delayed worker callback.
The delay is three times PyTorch's recorded per-worker join interval (15 seconds
with the pinned version), exceeding both workers' join windows. Only replacement
workers use the ordinary initializer. Reset, close, sharing policy and join
arguments are unchanged. Child initializer receipts record whether maskops was
imported; stock and reference-packet worker controls must not import it.

This checks whether the unchanged upstream shutdown can force nonzero worker
exits in all three paths under slow initialization. The delayed controls explicitly
expect that mechanism and retain `clean_zero_exit=false`; their success must not
be presented as a clean-exit pass, performance result, or proof that the original
intermittent failure had this cause. The natural-case exit assertions and original
failed matrix remain unchanged. The runtime's exact reset/close/shutdown method
source is retained with the control output.

All six controls completed. The [independent readback](validation/mask-windows-deadline-v1-preservation.json)
verified all 24 child receipts and six JUnit cases. In each ordinary case both
previous workers exited with code 0. In each delayed case both original 5-second
join calls expired and upstream called terminate twice; both previous workers
exited with **-15**, and the replacement generation completed and exited with 0.
Neither stock nor reference-packet workers imported maskops. Native workers did.
Thus this possible mechanism also exists without importing the extension. The
original failed job did not record terminate calls or actual exit codes, so its
cause remains unproven; no runtime fix or relaxed exit criterion is claimed.

The persistent-format integration test now records original join/terminate calls,
arguments, timings and exit codes on its previous worker objects, adding them to
assertion failure context. The exit-code-zero requirement, reset ordering and
paired output comparisons are retained. Parent-side logging adds diagnostic
overhead. The next full matrix also qualifies the compact source packaging on
all supported platforms; any success remains a repetition with richer evidence,
not erasure of the original failure.
