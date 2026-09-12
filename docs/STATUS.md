# Implementation status — 2026-09-12

Active development; original PRD goals remain unchanged. Not release-ready.

- C++ three-function core, packed API, retained/bounded overlap and explicit Format adapter implemented.
- macOS: 113 tests passed, including 10,000 seeded differential cases and actual YOLODataset first batches at workers 0/2.
- Linux: the same 113 tests passed, including actual framework/worker integration.
- Dirty scratch clearing and conservative composition bounds implemented without changing OpenCV fill/resize semantics. Final boundary regressions passed 98 core tests on both hosts.
- Nine synthetic cases: wrapper geometric-mean speedup 2.05× on M2 and 2.41× on the i5-10400/3070 server (CPU operation). The initial slower M2 result is retained.
- Memray: 75 independent allocation runs on each host; automatic mode reduced working allocation 61–91% in the five measured N≥100 cases. Process RSS is reported separately.
- Project-code ASan/UBSan: 98 tests passed on macOS, leak detection disabled, private OpenCV uninstrumented.
- Raw measurements and scope limitations are in [BENCHMARKS.md](BENCHMARKS.md). Invalid inherited Linux ru_maxrss data is explicitly excluded; corrected VmHWM reruns are retained.
- Remaining release gates: real COCO/resolution/vertex coverage, CPU DataLoader throughput, full GPU epochs, Windows and supported Python matrix, full dependency license inventory, wheels/CI, exact release-candidate validation. These synthetic results do not establish the full representative-workload gate.
