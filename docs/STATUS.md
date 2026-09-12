# Implementation status — 2026-09-13

Active development; original PRD goals remain unchanged. Not release-ready.

- C++ three-function core, packed API, retained/bounded overlap and explicit Format adapter implemented.
- macOS: 113 tests passed, including 10,000 seeded differential cases and actual YOLODataset first batches at workers 0/2.
- Linux: the same 113 tests passed, including actual framework/worker integration.
- Dirty scratch clearing and conservative composition bounds implemented without changing OpenCV fill/resize semantics. Final boundary regressions passed 98 core tests on both hosts.
- Nine synthetic cases: wrapper geometric-mean speedup 2.05× on M2 and 2.41× on the i5-10400/3070 server (CPU operation). The initial slower M2 result is retained.
- Memray: 75 independent allocation runs on each host; automatic mode reduced working allocation 61–91% in the five measured N≥100 cases. Process RSS is reported separately.
- Project-code ASan/UBSan: 98 tests passed on macOS, leak detection disabled, private OpenCV uninstrumented.
- Raw measurements and scope limitations are in [BENCHMARKS.md](BENCHMARKS.md). Invalid inherited Linux ru_maxrss data is explicitly excluded; corrected VmHWM reruns are retained.
- Full COCO CPU DataLoader harness added with frozen input/upstream hashes, workers 0/2/8, separate full output verification, externally sampled process-family RSS, and fresh-reference cache rebuilding. A 64-image smoke passed complete batch parity at workers 0/2; this is not a representative speedup result. Full two-host repeated measurements and diagnostic profiling are in progress; see [COCO_LOADER.md](COCO_LOADER.md).
- Full COCO non-augmented CPU DataLoader: 60 fresh measured processes across two hosts and workers 0/2/8; all full outputs match, including an independent original-scanner cache rebuild on each host. The 10% full-loader performance gate remains unmet; family RSS is approximately unchanged. See [COCO_LOADER.md](COCO_LOADER.md) and the complete results in [BENCHMARKS.md](BENCHMARKS.md).
- Direct stage probes passed exact call-count and full-output checks. Native-side cProfile call accounting was invalid and is explicitly excluded from component timing conclusions; the rejected evidence is retained.
- A separate worktree is evaluating cached polygon bounds and consecutive duplicate integer vertex removal. It is not part of this measured baseline.
- Remaining release gates: representative workload performance, augmented/non-overlap and higher-resolution coverage, full GPU epochs, Windows and supported Python matrix, full dependency license inventory, wheels/CI, exact release-candidate validation.
