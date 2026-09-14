# Repeated RTX 3070 training results — 2026-09-13

All 45 requested jobs completed, covering 90 full epochs. An independent audit
verified all raw/compact records, commands, runtime/source/input identities,
complete batch traces and memory accounting. All 15 worker/repetition groups
have identical batch loss vectors and final model hashes across reference,
mask replacement and combined mask/dataset replacements. All 12 epoch timing
summaries match independently recomputed paired bootstrap estimates.

The measured training benefit is modest: ratios of median epoch times range
from 1.002× to 1.021×. Several 95% intervals include 1. Process-family RSS does
not show a consistent reduction; medians increase for both replacements with
workers 2 and 8. CUDA allocator peak medians are unchanged. These results do
not establish the remaining CPU batch-preparation performance gate.

## Scope

The server has an i5-10400 (6 cores / 12 logical CPUs), 32 GB RAM and an RTX 3070
8 GB, driver 580.173.02. Runtime: Python 3.12.14, Torch 2.10.0+cu128, torchvision
0.25.0+cu128, NumPy 2.4.4, OpenCV 4.13.0 and pinned Ultralytics 8.4.149.
The hashed COCO fixture has 5,000 images and 36,335 instances. Random YOLO11n-seg,
FP32, SGD, batch 4, 640 pixels, overlap ratio 4, seed 912 and training augmentation
are fixed. Every job has two complete epochs of 1,250 batches each.

These measurements use the original alignment-corrected Linux wheels, before
the later masks-only and resize-ROI candidates. `mask` replaces Format mask
preparation; `both` also replaces dataset scanning/cache. They are shared-host
throughput observations, not accuracy measurements or a general speedup guarantee.
Five pairs give limited uncertainty resolution. See the
[protocol](GPU_BENCHMARK_PROTOCOL.md) for ordering, cache contracts and reproduction.

## Epoch wall time

Medians of five fresh jobs; epoch 1 and epoch 2 are reported separately. CUDA
synchronization brackets each epoch. Initialization, validation and checkpoint
saving are excluded. The first epoch includes GPU warmup; upstream may prefetch
during loader construction. The ratio is reference median / candidate median;
the 95% paired interval uses 10,000 resamples, seed 912.

| Workers | Replacement | Epoch | Reference (s) | Candidate (s) | Ratio | 95% paired interval |
|---:|---|---:|---:|---:|---:|---:|
| 0 | mask | 1 | 159.838 | 156.491 | 1.0214 | 1.0120–1.0261 |
| 0 | mask | 2 | 154.908 | 152.723 | 1.0143 | 1.0092–1.0196 |
| 0 | both | 1 | 159.838 | 156.819 | 1.0193 | 1.0077–1.0336 |
| 0 | both | 2 | 154.908 | 152.286 | 1.0172 | 1.0144–1.0292 |
| 2 | mask | 1 | 91.042 | 90.787 | 1.0028 | 1.0016–1.0101 |
| 2 | mask | 2 | 87.081 | 86.622 | 1.0053 | 0.9951–1.0166 |
| 2 | both | 1 | 91.042 | 90.860 | 1.0020 | 0.9966–1.0156 |
| 2 | both | 2 | 87.081 | 86.917 | 1.0019 | 0.9886–1.0172 |
| 8 | mask | 1 | 92.638 | 92.076 | 1.0061 | 0.9905–1.0114 |
| 8 | mask | 2 | 88.524 | 87.393 | 1.0129 | 0.9918–1.0218 |
| 8 | both | 1 | 92.638 | 91.961 | 1.0074 | 1.0030–1.0112 |
| 8 | both | 2 | 88.524 | 88.039 | 1.0055 | 0.9971–1.0154 |

## Memory

Whole-job sampled summed process-family RSS, in GiB. Each cell gives the median
and observed min–max across five jobs. The range is not a confidence interval.
RSS includes initialization and validation, double-counts shared pages across
processes and misses between-sample spikes. The largest observed sampling gap
was 0.667 seconds. It is not the mask kernel working allocation or an epoch-only
peak; earlier Memray reductions must not be substituted for this measurement.

| Workers | Reference | Mask | Both |
|---:|---:|---:|---:|
| 0 | 3.485 (3.389–3.519) | 3.421 (3.408–3.431) | 3.470 (3.427–3.540) |
| 2 | 7.254 (7.246–7.342) | 7.358 (7.307–7.444) | 7.411 (7.319–7.469) |
| 8 | 16.621 (16.552–16.715) | 16.906 (16.609–17.046) | 16.848 (16.770–17.094) |

CUDA allocated/reserved peak medians below are in GiB and are the same for all
three backends. These are PyTorch allocator peaks per epoch, not total GPU
device memory. All five raw values and observed ranges remain in the summary.

| Workers | Epoch | Allocated | Reserved |
|---:|---:|---:|---:|
| 0 | 1 | 1.5021 | 2.4570 |
| 0 | 2 | 1.4830 | 2.4570 |
| 2 | 1 | 1.5098 | 2.1094 |
| 2 | 2 | 1.4806 | 2.1094 |
| 8 | 1 | 1.5271 | 2.6875 |
| 8 | 2 | 1.5323 | 2.6875 |

## Preserved evidence

The [complete archive](../bench/results/cuda-series-complete-v1.tar.gz) contains
all 45 raw reports, the parent series and the exact parent/trial source files.
Its server and local SHA-256 is
`47494b76c4493254f21b648b54a2eca5025f9e4e77e4496f0e98c1cf94c78131`.
All 48 source/report hashes were checked after copying.

- [Independent final audit](validation/cuda-series-complete-audit-v1.json)
- [Timing and memory summary](validation/cuda-series-complete-summary-v1.json)
- [Preservation manifest](validation/cuda-series-complete-manifest-v1.json)
- [Exact auditor](validation/cuda-series-complete-auditor-v1.py)
- [Summary reproduction script](validation/summarize-cuda-series-v1.py)

Run the archived auditor with `--series`, `--runs-dir`, `--parent-script`,
`--trial-script` and a new `--out`, without `--allow-partial`. Then run the
summary script with `--series`, `--audit`, `--runs-dir` and a new `--out`.

The final audit checks the entire recorded series. Matching losses/models
supports equivalence on this workload; it does not replace direct target parity
or validate other datasets, non-overlap, higher resolution or newer binaries.
Exact release-candidate, portability and remaining performance gates stay open.
