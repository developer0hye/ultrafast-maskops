# Shared collator real-data loader protocol: full comparison audited

The CPU loader harness now accepts `--persistent-mask`. Both backends still use
the ordinary YOLODataset and the complete real 5,000-image segmentation fixture.
The reference retains its original formatter/collator. Native persistent mode
installs FastFormat and the shared packet collator. It records the actual loader
collator, and verifies installed Python runtime bytes against the frozen harness
source tree before execution. The full repeated protocol has completed; see the
[results and preserved evidence](SHARED_LOADER_FULL_RESULTS.md).

The [serial pilot controller](validation/launch-shared-loader-linux-v1-pilot.py)
completed on Linux from frozen harness commit `c25ef28`, using the installed
257-test packet wheel. Its [launch identity](validation/mask-shared-loader-linux-v1-pilot-launch-identity.json)
binds source, wheel, qualification receipts and actual controller PID. The pilot
uses all 5,000 images but only one pair per worker count and mask mode: 12 fresh
measured processes. After both timing modes terminated, both fresh-reference
cache verifications passed. The [completed pilot evidence](SHARED_LOADER_PILOT.md)
also passed independent audits and all 76 damaged-evidence rejection cases.
This is not a five-pair performance result.

The [full serial controller](validation/run-shared-loader-linux-full-v1.py) has
now completed the separately declared five-pair matrix: 60 fresh measured processes
across both modes and workers 0/2/8, followed by two fresh-reference verifications.
It reuses the exact frozen pilot harness/runtime and requires the sealed pilot
receipt before launching. Pilot measurements are excluded. The
[launch identity](validation/mask-shared-loader-linux-full-v1-launch-identity.json)
records the observed PID, process start ticks, host boot ID and source/qualification
hashes. Both fresh-reference checks and the independent complete audits passed.

Use both overlap modes, worker counts 0/2/8, five alternating fresh-process pairs,
batch size 8, image size 640, mask ratio 4, no augmentation, no image RAM cache,
and no pinning. The first epoch is timed with periodic process-family RSS
samples. A complete second pass hashes every batch field outside measurement.
Shutdown follows verification and must produce zero exits for every worker.
Original-reference shutdown failures remain failures; transport is not normalized
on the reference side of a performance comparison.

The report declares its entire round/worker plan before launching children and
marks completion only after all planned records and summaries exist. The final
fresh-reference verification checks plan coverage, actual collator identity,
worker count and exit codes before comparing outputs to a freshly rebuilt cache.
Run that destructive-to-generated-cache verification only after all timing jobs
using the cache finish. Image and annotation bytes remain unchanged.

This measures full CPU loader behavior, not close-mosaic transitions or training.
Shared pages are counted in each process RSS; the summed family peak is not
unique physical memory. The separate GPU lifecycle/resume protocol and direct
allocation evidence remain necessary. The completed experiment finds small
eight-worker gains, with no 10% loader improvement or training-speed claim.

The harness source can be frozen separately from its installed wheel, but all
runtime Python files and native extension identities must match the measured
wheel. Changes to benchmark scripts do not replace the installed wheel's test
receipt. Prior loader results belong to their original immutable harness and
candidate; they cannot qualify the packet revision.

## Independent artifact audit, executed for the pilot

The new [auditor](validation/audit-shared-loader-v1.py) uses the frozen source,
wheel and installed extension [identity](validation/mask-shared-loader-linux-v1-audit-identity.json),
derived separately from the result JSON. It checks exact execution order, full
counts, source/runtime identity, actual collator and clean worker exits. It
recomputes timing, throughput, batch-arrival quantiles and sampled family RSS
from raw records, and binds fresh-cache reference output to each complete mode.
A one-pair pilot produces raw pairs without inferential intervals. The five-pair
path remains a separate protocol and requires all 30 runs per mode.

The [rejection qualifier](validation/qualify-shared-loader-audit-v1.py) prepares
two valid controls and 76 damaged-evidence cases, including consistent edits to
raw/parent records and rebound fresh-report hashes. It exercises wrong collators,
worker aborts, duplicate/missing PIDs, type confusion, incomplete plans, incorrect
resource arithmetic and altered fresh-reference transport. Both controls passed
and all 76 damaged cases were rejected after timing and fresh-reference follow-up
finished. The five-pair path has now passed its separate qualification against
actual repeated records. Both archives passed server and downloaded-member readback.

## Five-pair audit qualification completed

The separate [full-series qualifier](validation/qualify-shared-loader-full-audit-v1.py)
requires the completed sixty-process controller and both successful fresh-reference
steps. It binds their recorded hashes, invokes the auditor with `rounds=5`, and
requires 30 complete records per mask mode. Its 92 damaged-evidence cases include
late-round omissions, duplicate rounds, reversed execution order, incorrect p95,
missing aggregate groups/metrics, and a fresh receipt covering only pilot counts.
Worker transport cases select a native row with two workers by its fields; the
pilot's fixed row index would select a zero-worker row in the five-pair matrix.

For all 30 real metric summaries and three synthetic mathematical controls, a
separate oracle enumerates all 3,125 ordered paired resamples. This checks the
auditor's weighted-combination implementation, including ties and percentile
interpolation. Five pairs remain a small sample: agreement between calculations
does not establish statistical coverage or eliminate shared-host interference.

The [full-series sealer](validation/seal-shared-loader-full-v1.py) requires those
controls, all 92 rejections, matching canonical audit contents, and complete
report/raw/fresh bindings. It preserves all samples, logs, memory arrays, exact
source and runtime qualification evidence, then reads every archive member back.
Downloaded archive/member verification also passed for all 285 members.

The full qualifier and sealer passed Ruff formatting/lint, with recorded hashes
and unchanged ASTs. After both the measurement and queued lifecycle qualification
terminated successfully, both real controls, all 92 rejections, both full audits
and archive readback passed. The full-result document links their distinct
receipts; the earlier pilot evidence was not substituted for this qualification.
