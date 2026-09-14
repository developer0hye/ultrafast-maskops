# Linux paired-wheel ROI results

The corrected resize-ROI wheel completed the frozen 270-process comparison
against the tested masks-only wheel. All output checks passed. For nonempty
ratio-4 cases, public-call median speedups range from **1.47× to 3.77×**.
The ratio-1 non-overlap case regressed: **3.960 → 4.107 ms**, about **3.7% slower**
(ratio 0.964, paired interval 0.923–0.983). This condition is retained below and
needs investigation; the candidate is not yet ready to merge or release.

Process RSS medians differ by less than 1 MiB in every condition, including the
empty cases. These numbers do not establish a meaningful reduction in working
allocation. Full real-loader timing, GPU throughput, sanitizer checks and M2
performance remain separate requirements. Do not multiply these ratios by the
earlier Ultralytics comparisons or describe them as training speedups.

## Exact comparison and method

This is CPU mask preparation on the i5-10400 / RTX 3070 server (12 logical CPUs,
32 GB nominal RAM); the GPU is not used. Both environments use Python 3.12.14,
NumPy 2.4.4, OpenCV 4.13.0, GNU 13.3.0 and CMake 3.28.3. The wrappers, golden
reference and fixture code match. All installed packages except the selected
maskops wheel match. Private OpenCV build information is identical except for
the build timestamp and temporary wheel install prefix. Both use one native
thread. This remains a shared host with an uncontrolled OS cache.

- Baseline: masks-only binding SHA `99e10b49…e9a10`, source commit `ae7cfd7`;
  wheel `b5257f36505db01e1d1c0e46db2a059c3b64fc25c4f1b9281095bbc550383026`.
  A clean installation passed standalone/RECORD/notice checks and all 151 tests
  (20.34 s), with the expected compiled source/profile hashes.
- Candidate: corrected ROI binding SHA `c63940b6…1813`;
  wheel `1b08974c11b1bc00e7e3f85f68fb6f8505e9e2edc5ca55b941ca6ef09b789fad`.
  Its 201-test run and all 18 full augmented COCO output checks are recorded in
  [RESIZE_ROI.md](RESIZE_ROI.md). Those real-data checks measure correctness only.

Nine unchanged cases use 640×640 images and 16-vertex polygons: ratio 4 at
N=0/1/5/20/100/255/256/500, plus N=100 at ratio 1. Each operation/version/repetition
runs in a fresh process. Five paired repetitions alternate which version goes
first. Each process checks full outputs, warms ten calls, retains 30 latency
samples, and checks outputs again. The 270 workers retain 8,100 latency samples;
two separate oracle processes generate expected output hashes before timing.

`overlap` is the public automatic-scratch wrapper. `bounded` includes packing
and constructing a new Rasterizer. `masks` is the public non-overlap wrapper.
All include output allocation/freeing. Empty non-overlap outputs retain the
upstream `(0,)` / `float64` behavior. The 255/256-instance overlap dtype boundary
is explicitly checked. RSS is the process high-water mark after timing and
before the final output check; imports, wheel verification, inputs and warmups
are included, so it is not isolated mask working allocation or process-family RSS.

## Every measured condition

Times are medians of five process medians. Intervals are paired percentile
bootstrap intervals over all 3,125 ordered five-pair resamples, with linear
interpolation. Values above 1 mean the candidate is faster. RSS columns are
separate medians of whole-process peaks; the audit retains every paired value.

| N | Ratio | Operation | Baseline ms | ROI ms | Speed ratio | 95% interval | Baseline RSS MiB | ROI RSS MiB |
|---:|---:|---|---:|---:|---:|---|---:|---:|
| 0 | 4 | overlap | 0.001682 | 0.001613 | 1.042 | 1.014–1.071 | 74.40 | 73.63 |
| 0 | 4 | bounded | 0.028903 | 0.029141 | 0.992 | 0.934–1.041 | 74.42 | 73.70 |
| 0 | 4 | masks | 0.000400 | 0.000378 | 1.058 | 1.027–1.073 | 74.38 | 73.63 |
| 1 | 4 | overlap | 0.071387 | 0.048444 | 1.474 | 1.434–1.532 | 74.38 | 73.63 |
| 1 | 4 | bounded | 0.104968 | 0.055658 | 1.886 | 1.826–1.947 | 74.39 | 73.65 |
| 1 | 4 | masks | 0.062625 | 0.037535 | 1.668 | 1.603–1.717 | 74.36 | 73.65 |
| 5 | 4 | overlap | 0.213752 | 0.089058 | 2.400 | 2.250–2.524 | 74.41 | 73.88 |
| 5 | 4 | bounded | 0.387163 | 0.133376 | 2.903 | 2.657–3.015 | 74.43 | 73.86 |
| 5 | 4 | masks | 0.198264 | 0.072818 | 2.723 | 2.401–2.839 | 74.36 | 73.67 |
| 20 | 4 | overlap | 0.769271 | 0.276074 | 2.786 | 2.461–2.969 | 74.46 | 73.80 |
| 20 | 4 | bounded | 1.461296 | 0.476458 | 3.067 | 2.675–3.097 | 74.48 | 73.87 |
| 20 | 4 | masks | 0.694075 | 0.240491 | 2.886 | 2.678–3.121 | 74.37 | 73.62 |
| 100 | 4 | overlap | 3.576145 | 1.066825 | 3.352 | 2.940–3.392 | 74.88 | 74.18 |
| 100 | 4 | bounded | 6.943354 | 1.926583 | 3.604 | 3.298–3.604 | 74.53 | 73.91 |
| 100 | 4 | masks | 3.439874 | 0.962102 | 3.575 | 3.260–3.657 | 74.43 | 73.63 |
| 255 | 4 | overlap | 8.235039 | 2.723425 | 3.024 | 2.953–3.337 | 80.46 | 79.85 |
| 255 | 4 | bounded | 16.031732 | 4.765991 | 3.364 | 3.357–3.719 | 74.36 | 73.73 |
| 255 | 4 | masks | 8.815231 | 2.335233 | 3.775 | 3.391–3.789 | 79.85 | 79.23 |
| 256 | 4 | overlap | 8.489423 | 2.973371 | 2.855 | 2.719–3.173 | 80.50 | 79.93 |
| 256 | 4 | bounded | 16.429530 | 5.335409 | 3.079 | 2.923–3.090 | 74.38 | 73.77 |
| 256 | 4 | masks | 8.017020 | 2.552227 | 3.141 | 3.122–3.539 | 79.86 | 79.28 |
| 500 | 4 | overlap | 16.333188 | 5.583475 | 2.925 | 2.899–3.241 | 86.57 | 85.91 |
| 500 | 4 | bounded | 31.703237 | 9.726409 | 3.260 | 3.140–3.276 | 74.50 | 73.89 |
| 500 | 4 | masks | 15.570059 | 4.869247 | 3.198 | 3.090–3.553 | 85.79 | 85.20 |
| 100 | 1 | overlap | 6.015813 | 5.304682 | 1.134 | 1.082–1.209 | 113.33 | 112.75 |
| 100 | 1 | bounded | 4.658372 | 4.137900 | 1.126 | 1.095–1.174 | 74.45 | 73.75 |
| 100 | 1 | masks | 3.959815 | 4.106896 | 0.964 | 0.923–0.983 | 112.51 | 111.95 |

## Evidence and independent verification

The [589-file archive](../bench/results/paired-wheels-linux-v1.tar.gz) contains
all measured and oracle raw reports/logs, the complete parent, exact measurement
and reference sources, both wheels and core sources, build/test logs, environment
freezes, standalone wheel checks and the earlier qualification run.
SHA-256: `c4b71c47793703ed14a2bdfa45402d4c3e3afc59b262a62bb23c9907dc3e8028`.

The [independent audit](validation/paired-wheels-linux-audit-v1.json) verifies
all archive hashes, 270 commands in their declared order, raw/parent records,
all 8,100 samples, wheel/source/environment identity and the fixed output shapes,
dtypes and hashes. It independently recomputes all 54 timing/RSS summaries using
126 weighted multisets equivalent to the parent's 3,125 ordered selections.
The [auditor](validation/audit-paired-wheels-linux-v1.py) takes the archive path
as its single argument and does not import a backend or rerun native code.

A valid complete archive passed and eight independently modified/rehashed
archives were rejected, including truncated confidence intervals, forged
statistics, reordered/missing workers, changed output/sample data, orphan files
and incorrect empty-output dtype. The [test receipt](validation/paired-wheels-linux-negative-tests-v1.json)
and [reproduction driver](validation/audit-paired-wheels-negative-cases-v1.py)
bind the exact audit/test source. The driver takes the auditor, original archive
and a new output directory. The initial shape review incorrectly assumed empty
masks used uint8; the retained golden source explicitly uses `np.array([])`.
The auditor was corrected to that existing behavior; no measurement was changed.

The comparison harness first passed two full oracle processes, six ten-sample
qualification workers covering the three operations, and constant/weighted
multiset aggregate checks. Those qualification timings are not included above.
