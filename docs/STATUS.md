# Implementation status — 2026-09-13

Active development; original PRD goals remain unchanged. Not release-ready.

- C++ three-function core, packed API, retained/bounded overlap and explicit Format adapter implemented.
- Current corrected source: 139 tests passed on each host, including 10,000 seeded differential cases, actual framework/worker integration, and four deterministic CPU YOLO11n-seg loss/gradient/update checks.
- Dirty scratch clearing, cached contour extents, conservative composition bounds and consecutive duplicate integer vertex removal are implemented without changing the OpenCV fill/resize sequence. Untyped byte snapshots correct the sanitizer failure on unaligned NumPy input.
- Earlier scratch-reuse snapshot, nine synthetic cases: wrapper geometric-mean speedup 2.05× on M2 and 2.41× on the i5-10400/3070 server (CPU operation). The initial slower M2 result is retained. These are not exact-current-source measurements.
- Earlier snapshot, Memray: 75 independent allocation runs on each host; automatic mode reduced working allocation 61–91% in the five measured N≥100 cases. Process RSS is reported separately; the current source needs exact-source reruns.
- Current project-code ASan/UBSan: 120 core tests passed on macOS, leak detection disabled, private OpenCV uninstrumented. Both the original alignment failure and the separate subprocess runtime-injection failure are retained.
- Raw measurements and scope limitations are in [BENCHMARKS.md](BENCHMARKS.md). Invalid inherited Linux ru_maxrss data is explicitly excluded; corrected VmHWM reruns are retained.
- Full COCO CPU DataLoader harness uses frozen input/upstream hashes, workers 0/2/8, separate full output verification, externally sampled process-family RSS, and fresh-reference cache rebuilding; see [COCO_LOADER.md](COCO_LOADER.md).
- Full COCO non-augmented CPU DataLoader: 60 fresh measured processes across two hosts and workers 0/2/8; all full outputs match, including an independent original-scanner cache rebuild on each host. The 10% full-loader performance gate remains unmet; family RSS is approximately unchanged. See [COCO_LOADER.md](COCO_LOADER.md) and the complete results in [BENCHMARKS.md](BENCHMARKS.md).
- Direct stage probes passed exact call-count and full-output checks. Native-side cProfile call accounting was invalid and is explicitly excluded from component timing conclusions; the rejected evidence is retained.
- The contour-bounds candidate and subsequent alignment correction are now merged. The candidate's separate 60-process full-loader experiment still misses the 10% gate and shows approximately unchanged family RSS. Its exact measured pre-correction source archive and all raw results are retained in [COCO_BOUNDS.md](COCO_BOUNDS.md). They must not be attributed to the subsequent alignment-corrected binary.
- Full real-data augmentation verification with both libraries is running on the corrected source; completion requires independent reference replay and native agreement at every requested worker count. GPU and wheel validation have their own evidence and must not be inferred from CPU parity.
- Remaining release gates: representative workload performance, augmented/non-overlap and higher-resolution coverage, full GPU epochs, Windows and supported Python matrix, full dependency license inventory, wheels/CI, exact release-candidate validation.
