# Unit-scale candidate: full DataLoader comparison in progress

The Linux comparison is running against the original Python/OpenCV Format
implementation on the same ordinary YOLODataset. It will retain 60 complete
fresh-process runs: overlap/non-overlap × workers 0/2/8 × five counterbalanced
reference/native pairs. Every run processes all 5,000 COCO val2017 images at
batch 8, image size 640 and mask ratio 4. This tests actual loader throughput;
it is not a GPU training benchmark.

The installed candidate is the 209-test wheel from source `9c4ece5`.
[Launch identity](validation/mask-unit-scale-linux-loader-identity-v1.json)
binds its seven measured source files, extension, wheel, input manifest and
upstream sources. The [launch script](validation/run-unit-scale-linux-loader-v1.sh)
runs the two modes sequentially. Source and runtime bytes remain frozen while
these processes are active.

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

The partial path has run on the actual three-worker checkpoint. **The complete
path and adversarial cases have not run yet.** Both benchmark hosts remain
occupied. Final validation must exercise the complete path and rejection tests,
then preserve the final receipts with raw samples and exact verifier bytes.
The checkpoint alone is not full-auditor qualification.

After both timing matrices terminate, run the frozen
`bench/verify_coco_loader.py` separately for each complete report. This deletes
and rebuilds only the fixture's generated label cache, then checks all benchmark
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
