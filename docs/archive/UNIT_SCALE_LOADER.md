# Unit-scale candidate: completed real-image DataLoader comparison

The Linux comparison completed against the original Python/OpenCV Format
implementation on the same ordinary YOLODataset. It retains 60 complete
fresh-process runs: overlap/non-overlap × workers 0/2/8 × five counterbalanced
reference/native pairs. Every run processes all 5,000 COCO val2017 images at
batch 8, image size 640 and mask ratio 4. This tests actual loader throughput;
it is not a GPU training benchmark.

The installed candidate is the 209-test wheel from source `9c4ece5`.
[Launch identity](validation/mask-unit-scale-linux-loader-identity-v1.json)
binds its seven measured source files, extension, wheel, input manifest and
upstream sources. The [launch script](validation/run-unit-scale-linux-loader-v1.sh)
runs the two modes sequentially. Measured source/runtime bytes were retained unchanged through both timing modes
and the fresh-reference follow-up.

## Preserved three-worker checkpoint

The first reference/native pair and the following native worker at workers=0
passed the [partial audit](validation/mask-unit-scale-linux-loader-overlap-yes-checkpoint-3-audit-v1.json).
All three processed and verified 5,000 images / 625 batches, using identical
36,335-instance populations. Their full batch-output digest is
`81118723b01b4f3ca696dea1f7c3dd275b203a3e0dded57ac1c1667e22165481`.
The audit independently checks arrival/epoch/throughput calculations and the
external process-family memory samples against each recorded peak/count/gap.

This checkpoint does not establish a throughput gain: the first pair measured
22.362 s reference and 22.531 s native. The next unpaired native sample was
21.434 s. These values are retained diagnostics, not a five-pair aggregate or
proof that a regression generalizes. No condition or slower sample is removed.

The [25-file checkpoint archive](../bench/results/mask-unit-scale-linux-loader-checkpoint-3-v1.tar.gz)
contains the parent snapshot, all three workers' raw JSON/log/phase/memory
files, identity, audit/verification source, launch script, seven frozen sources
and hash manifest. SHA-256:
`6c6e8f29ccf52731b514e0d0832b8475f96de63f99cbb57868f41dea5c42cdcd`.
The downloaded archive was reopened and every member hash checked.

## Completion and verification requirements

The [new independent auditor](validation/audit-unit-scale-loader-v1.py) checks
the exact 30-worker order for each mode, raw/parent equality, actual memory
sample arrays, complete-output hashes, input/source identities, full counts,
arrival quantiles, throughput arithmetic and aggregate medians/p95 values.
Its final paired intervals enumerate all 3,125 five-pair resamples using
weighted multisets, rather than trusting the harness's random-bootstrap
intervals. Throughput ratios have the opposite preferred direction from
latency/RSS ratios; reports must label that distinction.

Both complete paths have now passed on the actual 30-worker reports and fresh
reference evidence. Separate valid controls passed and 44 damaged-evidence cases
were rejected, including consistent parent/raw edits and rebound fresh-report
hashes. The earlier three-worker checkpoint remains historical evidence.

After both timing matrices terminated, the frozen
`bench/verify_coco_loader.py` ran separately for each complete report. It deleted
and rebuilt only the fixture's generated label cache, then checked all benchmark
outputs against a fresh original YOLODataset pass. Do not rebuild that cache
while either timing mode runs. Keep each fresh pass's JSON, log, phase, cache
hashes and parent receipt; the auditor requires this full bound fresh evidence
before it will produce an aggregate.

Memory sampling sums the benchmark process and children every approximately
50 ms while timing is active. It counts shared pages more than once and can
miss brief peaks; maximum gaps are preserved. All batch-output hashing is in
a second untimed pass. Full input-content verification before/after each
process pre-reads file bytes, so this is not a cold-storage experiment. Current
shared-host load remains part of the evidence. Independent allocation and
synthetic-call results do not substitute for completion of this loader series.

## Queued post-timing verification

The [follow-up controller](validation/finish-unit-scale-linux-loader-v1.py)
finished after watching the exact timing shell PID 1482166
(start ticks 231510731). It checks that exact process identity, freezes its
verifier/source hashes, and waits without touching the corpus or generated
cache. After that process exits, it requires both 30-worker summaries and every
worker's terminal phase before launching either cache rebuild. An incomplete
measurement stops the follow-up; it never restarts the benchmark automatically.

The controller then runs fresh-cache reference verification and the complete
auditor for overlap and non-overlap sequentially, retaining commands, logs,
results and errors. All four post-timing steps completed successfully. The timing process and
follow-up both terminated before the 44 rejection cases ran. The final archive
was sealed and downloaded, then every member was checked again locally.

## Complete results: five paired repetitions per condition

| Overlap | Loader workers | Reference epoch, s | Native epoch, s | Reference/native | Exact paired 95% interval | Sampled family RSS reference / native, MiB |
|---|---:|---:|---:|---:|---:|---:|
| Yes | 0 | 21.8834 | 22.2351 | 0.9842 | 0.9842–1.0183 | 674.04 / 674.05 |
| Yes | 2 | 15.6464 | 14.9029 | 1.0499 | 1.0136–1.0998 | 2025.08 / 2016.30 |
| Yes | 8 | 15.9531 | 15.6377 | 1.0202 | 1.0122–1.0244 | 6065.07 / 6050.48 |
| No | 0 | 37.3647 | 37.3317 | 1.0009 | 0.9750–1.0637 | 685.21 / 682.26 |
| No | 2 | 24.3566 | 23.6054 | 1.0318 | 1.0209–1.0442 | 2042.83 / 2036.09 |
| No | 8 | 20.0601 | 19.9543 | 1.0053 | 0.9990–1.0106 | 6110.30 / 6109.80 |

The best epoch median ratio is 1.0499 at overlap/workers=2. The 10% full-loader
performance gate remains unmet. Overlap/workers=0 is slower at the median, and
three conditions' intervals include 1.0. Every result is retained. Sampled family
RSS differences are small: even the statistically separated workers=2 conditions
have median reductions below 0.5%. This does not reproduce the large isolated
mask-allocation reductions in whole-loader memory.

Family RSS sums the parent and children, including shared pages multiple times;
do not interpret the approximately 6 GiB workers=8 values as distinct physical
allocations or compare worker counts as an allocation model. Max sampling gaps,
load, available memory and host swap deltas remain in the raw reports. This is
a non-augmented CPU loader experiment, not a GPU training throughput claim.

All 30 overlap outputs and the fresh reference match hash
`81118723b01b4f3ca696dea1f7c3dd275b203a3e0dded57ac1c1667e22165481`.
All 30 non-overlap outputs and their fresh reference match
`21a55d19bc7c18aadb0630ae7c562f9f2619172fa687f73e26afe82daa8dedaf`.
Each worker covers 5,000 images, 625 batches and 36,335 instances. Verification
hashes every batch field in a separate untimed pass.

The [287-file complete archive](../bench/results/mask-unit-scale-linux-loader-complete-v1.tar.gz)
contains every measured JSON/log/phase/memory file, both fresh-reference runs,
complete audit statistics, the 44 rejection cases, exact verifier sources, seven
measured project sources, six upstream sources and the tested candidate wheel.
SHA-256: `fd15d178131add1913c2a1c5b563f4c3f4f0c54c738f969a9d50edc4106b3c31`.
It was reopened on the server and after download; every member matched its hash.
The 830 MB image/label corpus remains external and is bound by the input manifest.

See the complete [overlap audit](validation/mask-unit-scale-linux-loader-complete-yes-audit-v1.json),
[non-overlap audit](validation/mask-unit-scale-linux-loader-complete-no-audit-v1.json),
[rejection results](validation/mask-unit-scale-linux-loader-audit-negative-v1.json)
and [preservation receipt](validation/mask-unit-scale-linux-loader-complete-preservation-v1.json).
The result qualifies this frozen binary's loader parity and measured scope;
current-candidate GPU, sanitizer, supported-platform and release gates remain.
