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
retain this adverse result. The full matrix was still running when this document
was created. The small evidence archive retains the complete failed job log,
job/artifact metadata, and exact original test/workflow sources. It does not
back up the original 116.5MB wheel/sdist artifact.

## Predeclared diagnosis

The diagnostic workflow uses the **same original Windows wheel**, downloaded on its hosted runner after
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
