# Shared packet: full-image functional pilot passed

All **12 fresh loader processes** completed: overlap/non-overlap, workers 0/2/8,
one original/native pair per condition. Each processed 5,000 images and verified
all fields of all 625 batches in a separate untimed pass. Every configured worker
exited with code zero. Both subsequent fresh original-cache verifications passed.
The two independent artifact audits passed, and all 76 damaged-evidence cases
were rejected. This is functional qualification on one Linux profile, not a
five-pair performance result, accuracy test or GPU training result.

The native wheel is the [257-test packet revision](SHARED_PACKET_VALIDATION.md),
source `5f6fc8e`. The separately frozen CPU harness is `c25ef28`; its runtime
Python files and C++ sources match that wheel's source, and installed bytes were
checked. Native persistent mode uses FastFormat and the shared packet collator;
the reference keeps its original formatter/collator. The corpus, loader settings,
single-thread settings and memory limitations are in the
[declared protocol](SHARED_LOADER_PROTOCOL.md).

## Raw observations, one pair per condition

| Overlap | Workers | Reference epoch, s | Native epoch, s | Reference family RSS, MiB | Native family RSS, MiB |
|---|---:|---:|---:|---:|---:|
| Yes | 0 | 23.6559 | 19.2666 | 666.68 | 662.14 |
| Yes | 2 | 15.9997 | 15.4287 | 1982.95 | 1966.17 |
| Yes | 8 | 15.9474 | 15.6717 | 5936.63 | 5879.12 |
| No | 0 | 37.4771 | 37.8431 | 667.75 | 665.91 |
| No | 2 | 24.0289 | 24.3145 | 1992.88 | 1982.13 |
| No | 8 | 20.1258 | 19.5075 | 5962.14 | 5966.98 |

No interval or general speedup estimate is justified by one pair. Some native
observations are slower, and non-overlap/workers=8 has slightly higher sampled
family RSS. The large overlap/workers=0 timing difference cannot demonstrate a
benefit from shared IPC: that path delegates collation to the original function.
All observations remain in the report. Repeated, counterbalanced measurements
are required to evaluate gains and regressions.

Family RSS is sampled approximately every 50 ms and sums the benchmark process
and its children, counting shared pages multiple times. It is not unique physical
memory or an allocation trace. First-epoch time includes worker startup and queue
fill; hashing, fresh verification and explicit shutdown occur outside that timer.
Full corpus checks pre-read file contents, so these are not cold-storage results.

The six overlap outputs and fresh original reference match
`81118723b01b4f3ca696dea1f7c3dd275b203a3e0dded57ac1c1667e22165481`.
The six non-overlap outputs and fresh original reference match
`21a55d19bc7c18aadb0630ae7c562f9f2619172fa687f73e26afe82daa8dedaf`.
Each run covers the same 36,335-instance population. This does not exercise the
trainer's close-mosaic/resume paths or GPU pinning.

## Independent checks and preserved evidence

The [overlap audit](validation/mask-shared-loader-linux-v1-pilot-yes-audit.json)
and [non-overlap audit](validation/mask-shared-loader-linux-v1-pilot-no-audit.json)
check raw/parent equality, full plan/order, exact collator, exit codes, identities,
image/batch counts, topology, output hashes, timing/throughput arithmetic, arrival
quantiles and actual memory sample arrays. Both are bound to complete fresh-cache
reference receipts. The [rejection results](validation/mask-shared-loader-linux-v1-audit-negative.json)
include raw/parent-consistent edits and rebound outer hashes; malformed data was
not rejected merely because an unrelated outer checksum changed.

The [93-file archive](../bench/results/mask-shared-loader-linux-v1-pilot-evidence.tar.gz)
contains every measured JSON/log/phase/memory file, both fresh-reference runs,
audit/qualification sources and results, frozen harness sources, upstream files,
launch state and the exact wheel qualification archive. The
[preservation receipt](validation/mask-shared-loader-linux-v1-pilot-preservation.json)
binds each member. All bytes were checked after download against the archive's
own manifest and the separate receipt. Archive size is 2,234,459 bytes, SHA-256
`2c55babb249ea1be02eead21a870557740f114be31d173c38e66a3ae5cac9535`.

The frozen input image/label corpus remains external. Full repeated performance,
GPU lifecycle/resume, combined-library training and broader platform gates remain.
