# Unit-scale candidate: complete Linux paired comparisons

The unit-scale copy candidate passed 209 tests, but **does not yet establish a
regression fix**. Its 100-instance, 640×640 non-overlap call is 2.1% faster than
the frozen ROI wheel in this experiment (4.291 → 4.202 ms; speed ratio 1.021,
paired 95% interval 1.0003–1.0604). In a separate direct comparison against the
masks-only wheel, it remains 1.8% slower at the median (3.976 → 4.049 ms; ratio
0.982, interval 0.9510–1.0263). These are separate paired runs; do not multiply
ratios from different runs. Keep the candidate experimental.

Nonempty ratio-4 calls retain 1.47–3.57× gains over masks-only. Against ROI,
nonempty ratio-4 ratios range from 0.960 to 1.037. Three ROI comparisons have
intervals wholly below one: empty non-overlap (0.967), N=100 ratio-4 bounded
(0.998), and N=100 ratio-1 bounded (0.981). All conditions, including these
regressions, are retained below. Five paired repetitions on a shared host do
not establish that these small differences generalize or identify their cause.

Whole-process peak RSS differs by less than 1 MiB in every condition. The
candidate is 0.45–0.81 MiB higher than ROI and between 0.086 MiB lower and
0.105 MiB higher than masks-only. These measurements do not demonstrate reduced
working allocation. The separate [75-trace Memray series](UNIT_SCALE_ALLOCATIONS.md) is complete
and independently audited; its allocation results are not inferred from RSS.

## Method and evidence

Two independent complete matrices each contain 270 measured fresh processes,
two separate oracles, five counterbalanced pairs per condition, ten warmups and
30 latency samples per worker (16,200 latency samples total). Whole public calls
include packing, output allocation/freeing and a new Rasterizer for explicit
bounded mode. Outputs are checked before and after timing. The nine seeded
16-vertex synthetic fixtures and three operations match the prior fixed suite.
Input preparation and oracle execution are outside measured calls. RSS includes
imports, installed-wheel verification, fixture creation and warmups.

The harness is unchanged at SHA-256
`ea4d7fa18643f22698675e5d7c4b6350fca86f0f9b13019bd7dc609a2e7c1a9c`.
Both comparisons use Linux x86-64, Intel i5-10400 (6 physical / 12 logical cores),
32 GB system RAM, Python 3.12.14, NumPy 2.4.4, GNU 13.3.0 Release and one private
OpenCV thread. Recorded installed dependencies and build settings agree;
OpenCV build timestamp and temporary install prefix are excluded from comparison.
The host is shared and the OS cache is uncontrolled. These are synthetic mask
calls, not DataLoader or GPU throughput measurements.

[Installed validation](UNIT_SCALE_MASK.md) identifies the candidate source,
wheel, extension and seven test-source files. The new
[auditor](validation/audit-unit-scale-paired-linux-v1.py) accepts only the pinned
candidate and the two pinned baselines. It verifies archive contents, test and
wheel identities, all raw/compact records, command ordering, dtypes/shapes,
output hashes, samples and independently recomputes all timing/RSS statistics
using weighted multisets equivalent to all 3,125 ordered bootstrap resamples.
The unchanged [negative-case driver](validation/audit-paired-wheels-negative-cases-v1.py)
accepted each valid archive and rejected eight separately forged, rehashed
archives for each comparison. This checks the auditor, not new native semantics.

Archive creation is reproducible with the retained
[sealing script](validation/seal-unit-scale-paired-linux-v1.py). Each archive
contains 578 files: complete parent/raw results and logs, measured wheels,
benchmark/core sources, candidate test sources, build/test/freeze/notice evidence
and a manifest. Downloaded archives were reopened and all member hashes checked.

## Direct comparison with resize-roi

[Archive](../bench/results/mask-unit-vs-resize-roi-linux-evidence-v1.tar.gz) ·
[audit receipt](validation/mask-unit-vs-resize-roi-linux-audit-v1.json) ·
[negative-case receipt](validation/unit-vs-resize-roi-negative-tests-v1.json)

Archive SHA-256: `1b4f84d5b8abfcba270112f6bf6cd414eca7923401509455722baaf17efb089a`.

Times are medians of five process medians, in microseconds. Ratios above one
favor the candidate. RSS values are medians in MiB, baseline → candidate.

| N | Ratio | Call | Baseline µs | Candidate µs | Speed ratio [paired 95% interval] | Peak RSS MiB |
|---:|---:|---|---:|---:|---|---|
| 0 | 4 | overlap | 1.62 | 1.65 | 0.9803 [0.9717, 1.0164] | 73.63 → 74.37 |
| 0 | 4 | bounded | 29.15 | 29.90 | 0.9751 [0.9424, 1.0462] | 73.68 → 74.38 |
| 0 | 4 | masks | 0.38 | 0.40 | 0.9673 [0.9144, 0.9796] | 73.63 → 74.38 |
| 1 | 4 | overlap | 48.24 | 48.72 | 0.9902 [0.9568, 1.0563] | 73.62 → 74.43 |
| 1 | 4 | bounded | 56.24 | 55.67 | 1.0103 [0.9990, 1.0328] | 73.63 → 74.38 |
| 1 | 4 | masks | 38.64 | 38.12 | 1.0138 [0.9587, 1.0611] | 73.62 → 74.37 |
| 5 | 4 | overlap | 88.10 | 89.85 | 0.9806 [0.9663, 1.0297] | 73.84 → 74.42 |
| 5 | 4 | bounded | 132.90 | 133.88 | 0.9927 [0.9732, 1.0116] | 73.92 → 74.43 |
| 5 | 4 | masks | 73.25 | 73.72 | 0.9935 [0.9722, 1.0225] | 73.64 → 74.45 |
| 20 | 4 | overlap | 280.83 | 282.32 | 0.9947 [0.9740, 1.0100] | 73.85 → 74.46 |
| 20 | 4 | bounded | 463.53 | 482.63 | 0.9604 [0.9534, 1.0084] | 73.81 → 74.45 |
| 20 | 4 | masks | 241.94 | 247.09 | 0.9792 [0.9535, 1.0443] | 73.63 → 74.41 |
| 100 | 4 | overlap | 1088.38 | 1100.55 | 0.9889 [0.9703, 1.0248] | 74.17 → 74.84 |
| 100 | 4 | bounded | 1960.69 | 1965.34 | 0.9976 [0.9954, 0.9983] | 73.96 → 74.52 |
| 100 | 4 | masks | 952.92 | 952.27 | 1.0007 [0.9568, 1.0315] | 73.62 → 74.37 |
| 255 | 4 | overlap | 2697.77 | 2735.64 | 0.9862 [0.9617, 1.0160] | 79.85 → 80.45 |
| 255 | 4 | bounded | 4881.43 | 4803.61 | 1.0162 [0.9876, 1.0265] | 73.75 → 74.44 |
| 255 | 4 | masks | 2348.45 | 2431.27 | 0.9659 [0.9163, 1.0160] | 79.28 → 79.79 |
| 256 | 4 | overlap | 3086.82 | 2977.99 | 1.0365 [0.9151, 1.0546] | 79.88 → 80.45 |
| 256 | 4 | bounded | 5544.85 | 5560.99 | 0.9971 [0.9855, 1.0293] | 73.64 → 74.39 |
| 256 | 4 | masks | 2595.82 | 2695.84 | 0.9629 [0.9564, 1.0251] | 79.30 → 79.85 |
| 500 | 4 | overlap | 5607.11 | 5841.81 | 0.9598 [0.9521, 1.0032] | 85.96 → 86.58 |
| 500 | 4 | bounded | 9741.68 | 10059.84 | 0.9684 [0.9554, 1.0058] | 73.88 → 74.55 |
| 500 | 4 | masks | 4896.18 | 4910.79 | 0.9970 [0.9802, 1.0258] | 85.24 → 85.75 |
| 100 | 1 | overlap | 5418.72 | 5406.16 | 1.0023 [0.9996, 1.0649] | 112.76 → 113.35 |
| 100 | 1 | bounded | 4160.79 | 4242.96 | 0.9806 [0.9675, 1.0000] | 73.77 → 74.44 |
| 100 | 1 | masks | 4290.52 | 4202.31 | 1.0210 [1.0003, 1.0604] | 112.00 → 112.45 |

## Direct comparison with masks-only

[Archive](../bench/results/mask-unit-vs-masks-only-linux-evidence-v1.tar.gz) ·
[audit receipt](validation/mask-unit-vs-masks-only-linux-audit-v1.json) ·
[negative-case receipt](validation/unit-vs-masks-only-negative-tests-v1.json)

Archive SHA-256: `23970b7afc08210f462368a75c4a44164a5843e59c5762483fdb046cf9e6dae1`.

Times are medians of five process medians, in microseconds. Ratios above one
favor the candidate. RSS values are medians in MiB, baseline → candidate.

| N | Ratio | Call | Baseline µs | Candidate µs | Speed ratio [paired 95% interval] | Peak RSS MiB |
|---:|---:|---|---:|---:|---|---|
| 0 | 4 | overlap | 1.63 | 1.61 | 1.0077 [0.9647, 1.0144] | 74.40 → 74.34 |
| 0 | 4 | bounded | 28.46 | 28.65 | 0.9932 [0.9527, 1.0243] | 74.43 → 74.48 |
| 0 | 4 | masks | 0.40 | 0.40 | 0.9875 [0.9227, 1.0278] | 74.35 → 74.44 |
| 1 | 4 | overlap | 73.56 | 49.89 | 1.4745 [1.4527, 1.5042] | 74.34 → 74.39 |
| 1 | 4 | bounded | 107.39 | 55.13 | 1.9479 [1.8597, 2.0108] | 74.42 → 74.35 |
| 1 | 4 | masks | 63.13 | 38.53 | 1.6385 [1.5345, 1.7517] | 74.43 → 74.37 |
| 5 | 4 | overlap | 214.34 | 88.46 | 2.4231 [2.2657, 2.5082] | 74.49 → 74.42 |
| 5 | 4 | bounded | 387.08 | 131.97 | 2.9330 [2.6913, 3.0141] | 74.46 → 74.46 |
| 5 | 4 | masks | 198.16 | 72.35 | 2.7387 [2.6378, 2.7731] | 74.41 → 74.43 |
| 20 | 4 | overlap | 787.56 | 277.71 | 2.8359 [2.4813, 2.9321] | 74.50 → 74.41 |
| 20 | 4 | bounded | 1468.35 | 474.04 | 3.0975 [2.7832, 3.2619] | 74.40 → 74.45 |
| 20 | 4 | masks | 730.63 | 242.12 | 3.0176 [2.8083, 3.1261] | 74.39 → 74.41 |
| 100 | 4 | overlap | 3601.88 | 1082.34 | 3.3279 [3.0354, 3.3360] | 74.81 → 74.75 |
| 100 | 4 | bounded | 6997.61 | 1962.28 | 3.5661 [3.2919, 3.6446] | 74.54 → 74.50 |
| 100 | 4 | masks | 3123.87 | 968.99 | 3.2238 [3.1956, 3.5507] | 74.43 → 74.39 |
| 255 | 4 | overlap | 9083.88 | 2710.23 | 3.3517 [3.1168, 3.4416] | 80.47 → 80.46 |
| 255 | 4 | bounded | 15990.13 | 4790.59 | 3.3378 [3.2632, 3.3517] | 74.39 → 74.46 |
| 255 | 4 | masks | 7919.26 | 2407.18 | 3.2899 [3.2674, 3.8217] | 79.86 → 79.77 |
| 256 | 4 | overlap | 8424.90 | 2973.94 | 2.8329 [2.7389, 2.8550] | 80.55 → 80.54 |
| 256 | 4 | bounded | 16393.50 | 5550.52 | 2.9535 [2.9166, 3.4598] | 74.42 → 74.36 |
| 256 | 4 | masks | 8912.81 | 2564.29 | 3.4757 [2.9918, 3.4903] | 79.83 → 79.84 |
| 500 | 4 | overlap | 16447.77 | 5633.85 | 2.9195 [2.8199, 3.1916] | 86.59 → 86.63 |
| 500 | 4 | bounded | 31748.04 | 10102.04 | 3.1427 [3.1280, 3.1736] | 74.54 → 74.52 |
| 500 | 4 | masks | 15506.66 | 4933.78 | 3.1430 [3.0832, 3.2104] | 85.73 → 85.83 |
| 100 | 1 | overlap | 6021.49 | 5311.54 | 1.1337 [1.1093, 1.2139] | 113.32 → 113.35 |
| 100 | 1 | bounded | 4649.64 | 4075.57 | 1.1409 [1.1032, 1.1470] | 74.43 → 74.38 |
| 100 | 1 | masks | 3976.46 | 4049.47 | 0.9820 [0.9510, 1.0263] | 112.46 → 112.54 |

## Remaining gates

M2 execution, sanitizer validation, non-overlap traced allocation, full real-data
validation for this exact binary, DataLoader/GPU throughput, and platform/Python
wheel CI remain separate requirements. Neither the earlier ROI corpus checks
nor the earlier GPU series automatically validates this new candidate.
