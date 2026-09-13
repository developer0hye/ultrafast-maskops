# Full Linux packet-loader result: small gains at eight workers

All 60 fresh measured processes completed, followed by two successful fresh-cache
reference verifications. Each process loaded all 5,000 COCO val2017 segmentation
images (36,335 instances) and separately hashed every batch outside timing. All
observed DataLoader workers exited with code zero. The original reference
formatter/collator was retained; the candidate combines FastFormat with the
packet collator. This does not isolate the packet transport's incremental effect.

This is a CPU loader experiment on the i5-10400/32 GiB RTX 3070 host. The GPU was
not used for training. Each condition has five alternating reference/native pairs,
batch size 8, image size 640, mask ratio 4, no augmentation, no image RAM cache,
no pinning, and workers 0/2/8. Pilot measurements are excluded from these results.

| Overlap | Workers | Reference epoch median | Native epoch median | Reference / native | Paired bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Yes | 0 | 21.335 s | 22.483 s | 0.9490 | 0.9407–1.0501 |
| Yes | 2 | 15.777 s | 15.635 s | 1.0091 | 0.9962–1.0394 |
| Yes | 8 | 15.910 s | 15.646 s | 1.0168 | 1.0137–1.0185 |
| No | 0 | 37.739 s | 36.104 s | 1.0453 | 0.9923–1.0752 |
| No | 2 | 24.318 s | 24.202 s | 1.0048 | 0.9896–1.0119 |
| No | 8 | 20.150 s | 19.556 s | 1.0304 | 1.0246–1.0335 |

Eight-worker epoch medians are 1.65% lower for overlap and 2.95% lower for
non-overlap. Their reported speed-ratio intervals are above 1; the other four
intervals include 1. The zero-worker overlap median is 5.38% slower. Every sample,
including slow candidate observations, remains in the report and calculation.
These small gains do not meet the 10% loader-improvement gate and do not establish
an end-to-end training speedup. Five pairs on a shared host provide limited
evidence; the intervals are per-condition and are not adjusted for multiple
comparisons.

| Overlap | Workers | Reference sampled family peak RSS median | Native median | Median reduction |
| --- | ---: | ---: | ---: | ---: |
| Yes | 0 | 658.352 MiB | 658.387 MiB | −0.005% |
| Yes | 2 | 1975.266 MiB | 1954.953 MiB | 1.028% |
| Yes | 8 | 5919.281 MiB | 5852.387 MiB | 1.130% |
| No | 0 | 668.074 MiB | 667.063 MiB | 0.151% |
| No | 2 | 1989.957 MiB | 1972.879 MiB | 0.858% |
| No | 8 | 5938.762 MiB | 5913.836 MiB | 0.420% |

Only the overlap 2/8-worker RSS-ratio intervals exclude 1. These are sampled sums
of process RSS, which count shared pages in multiple processes and can miss peaks.
They are not unique physical RAM or isolated temporary allocation measurements.
The result does not support a broad claim of substantial loader-memory savings.

## Qualification and preservation

The [terminal controller receipt](validation/mask-shared-loader-linux-full-v1-state.json)
records both timing modes and both fresh-reference steps returning zero. The
[overlap audit](validation/mask-shared-loader-linux-full-v1-yes-audit.json) and
[non-overlap audit](validation/mask-shared-loader-linux-full-v1-no-audit.json)
check exact execution order, all raw artifacts, counts, actual collator, source
and runtime identity, output parity, worker exits, timing arithmetic, batch-arrival
statistics and memory samples. Fresh reference results bind to the complete
30-record report for their mode.

The [qualification](validation/mask-shared-loader-linux-full-v1-audit-negative.json)
accepted both real controls and rejected all 92 damaged-evidence cases. A separate
ordered-resampling oracle checked all 30 real metric summaries and three synthetic
mathematical controls against the auditor's weighted implementation. Synthetic
controls supply no performance measurements. The
[execution record](validation/mask-shared-loader-linux-full-v1-audit-commands.json)
and [lint receipt](validation/mask-shared-loader-linux-full-v1-audit-lint.json)
retain the actual commands, successful exits and source hashes.

Frozen harness commit: `c25ef287595af7cbfdd9b6fc0a8029dcadd6ac29`.
Installed runtime source: `5f6fc8e4c9da94e33cfd079f35fe3e3f8bec7d52`.
Wheel SHA-256:
`8b467c837c323812a0201b81210a6dcd0c180ef50ba0d16531f07ef2c741cd9a`.
The runtime's 257-test installed qualification and 18 reset-stress processes are
separate prior evidence, included through the sealed runtime archive.

The [285-member archive](../bench/results/mask-shared-loader-linux-full-v1-evidence.tar.gz)
is 3,108,407 bytes, SHA-256
`4ee4bc834fe99c3e9b151fedb3d6ddffe5788004439a906e5d72d24095a77b77`.
Its [preservation receipt](validation/mask-shared-loader-linux-full-v1-preservation.json)
and [download readback](validation/mask-shared-loader-linux-full-v1-download-readback.json)
verify every member's bytes and hash, including raw reports/logs/phase/memory
arrays, fresh-reference results, exact sources and runtime qualification.

The separately prepared GPU coordinator passed 63 synthetic tests after this
measurement terminated. That does not exercise real CUDA pinning, close-mosaic
training, checkpoint restoration or combined mask/dataset training. The actual
54-trial GPU grid, current macOS packet qualification, named-storage lifetime
diagnostic and release/platform gates remain open.
