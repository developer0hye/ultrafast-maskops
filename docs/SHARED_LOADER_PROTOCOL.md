# Shared collator real-data loader protocol: pilot passed

The CPU loader harness now accepts `--persistent-mask`. Both backends still use
the ordinary YOLODataset and the complete real 5,000-image segmentation fixture.
The reference retains its original formatter/collator. Native persistent mode
installs FastFormat and the shared packet collator. It records the actual loader
collator, and verifies installed Python runtime bytes against the frozen harness
source tree before execution. The full repeated protocol is not complete.

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
now started the separately declared five-pair matrix: 60 fresh measured processes
across both modes and workers 0/2/8, followed by two fresh-reference verifications.
It reuses the exact frozen pilot harness/runtime and requires the sealed pilot
receipt before launching. Pilot measurements are excluded. The
[launch identity](validation/mask-shared-loader-linux-full-v1-launch-identity.json)
records the observed PID, process start ticks, host boot ID and source/qualification
hashes. This series is still running; no complete repeated aggregate is claimed.

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
allocation evidence remain necessary. A speed improvement is unproven until the
full repeated measurements and independent artifact audit complete.

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
finished. The five-pair path still requires actual complete repeated records and
its own audit. The pilot archive passed server and downloaded-member readback.
