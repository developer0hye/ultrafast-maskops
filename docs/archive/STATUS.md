# Implementation status — 2026-09-13

Active development; original PRD goals remain unchanged. Not release-ready.

The `feat/trainer-lifecycle` branch adds a persistent Format factory and a
bytes-only shared-batch packet. Its current Linux packet wheel passed 257 installed
tests and 18 fresh-process reset stress cases; earlier factory/shared-storage
failures remain preserved. The complete 60-process CPU loader experiment and
both fresh-reference checks passed independent audits. Eight-worker ratios are
1.0168× for overlap and 1.0304× for non-overlap, with small sampled family-RSS
changes; the 10% loader gate remains unmet. See
[SHARED_LOADER_FULL_RESULTS.md](SHARED_LOADER_FULL_RESULTS.md).

The GPU coordinator passed 63 synthetic tests and local workflow lint. A fresh
combined Linux runtime then passed 166 dataset and 257 mask tests with both wheel
payloads verified. The full 54-trial/126-epoch close-mosaic/resume GPU grid has
launched; the coordinator reports identical loss vectors and initial/final model
hashes across the first fresh reference/mask/combined cohort. Five trials exited
zero; combined resume-at-boundary is active in the
[retained checkpoint](validation/mask-lifecycle-gpu-linux-v1-m2-qualification-checkpoint.json). The
[independent checker](GPU_LIFECYCLE_AUDIT.md) passed 46 synthetic tests and audited
the first reference/mask trials, but the complete grid and final audit remain pending.
The current M2 wheel passed **366 installed tests, zero failures/skips and natural
process exit**, using its default `file_system` transport. All 573 selected
compiled dependency files match notice provenance. Named-storage lifetime,
repeated stress and broader release/platform qualification remain open. See
[M2_PACKET_VALIDATION.md](M2_PACKET_VALIDATION.md) and
[LIFECYCLE_GPU.md](LIFECYCLE_GPU.md).

Linux `file_system` diagnostics now expose retained storage and parent exit hangs
in both collators. A stock PyTorch full-epoch control reproduced the hang without
project imports; changing only to `file_descriptor` exited normally. One original
collator reset also aborted a worker. Linux `file_system` transport remains
unqualified; this is adverse evidence, not a new performance or release pass.
See [SHARED_TRANSPORT_LIFETIME.md](SHARED_TRANSPORT_LIFETIME.md).

The earlier `perf/unit-scale-mask` worktree added a unit-scale mask-only copy candidate
that passed all 209 installed-wheel Linux tests; see [UNIT_SCALE_MASK.md](UNIT_SCALE_MASK.md). The ROI
evidence below applies to its frozen `fec5cab` parent, not this new kernel.
The new candidate completed two independently audited 270-process comparisons:
2.1% faster than ROI for ratio-1 non-overlap, but still 1.8% slower at the median
than masks-only, with the latter interval including one. The regression fix
remains unproven; see [UNIT_SCALE_RESULTS.md](UNIT_SCALE_RESULTS.md).
The new binary also completed 75 audited Memray traces: public overlap peak
tracked allocation is 61–91% lower than the upstream Python/OpenCV reference;
explicit bounded mode reduces it 94–99.6%. These are allocation, not RSS or
training results. See [UNIT_SCALE_ALLOCATIONS.md](UNIT_SCALE_ALLOCATIONS.md).
The full 60-process real-data loader series and both rebuilt-cache reference
checks completed. All outputs match. Epoch ratios range from 0.9842 to 1.0499,
with small family-RSS differences; the 10% loader gate remains unmet. Both full
audits passed, and 44 damaged-evidence cases were rejected.
See [UNIT_SCALE_LOADER.md](UNIT_SCALE_LOADER.md).

The earlier experimental `perf/resize-roi` branch added sampling-aligned resize crops.
The initial crop candidate failed parity at Arm dispatch boundaries; its corrected
wheel passed 191 M2 tests, including 40 crop cases. Failure artifacts are retained.
The corrected Linux wheel passed 201 tests and all 18 full augmented overlap and
non-overlap runs (5,000 images per run; reference/reference/native at workers 0/2/8), with an
independent audit of the complete raw evidence. Its 375 selected dependency files
match the bundled notice inventory. The audited 270-process Linux comparison
shows 1.47–3.77× public-call gains in nonempty ratio-4 cases, a 3.7% ratio-1
non-overlap regression and less than 1 MiB RSS differences. See
[PAIRED_WHEEL_RESULTS.md](PAIRED_WHEEL_RESULTS.md). Sanitizer, M2, working-allocation
and end-to-end validation remain open before merging; see [RESIZE_ROI.md](RESIZE_ROI.md).
Historical performance numbers below belong to parent implementations and do not
measure this candidate.

The 12-job platform/Python [wheel CI workflow](WHEEL_CI.md) is prepared and
locally linted. Hosted execution awaits the repository visibility choice; this
is not Windows or broader Python compatibility evidence. The main branch now
explicitly aligns OpenCV's MSVC CRT with the extension; existing benchmark
binaries predate that build-configuration change.

- C++ three-function core, packed API, retained/bounded overlap and explicit Format adapter implemented.
- The alignment-corrected source passed 139 tests on each host, including 10,000 seeded differential cases, actual framework/worker integration, and four deterministic CPU YOLO11n-seg loss/gradient/update checks.
- The subsequent non-overlap masks-only path skips unused area reduction and passed all 139 tests in a fresh M2 wheel installation. Five-process isolated measurements show about 1.08× at ratio 4 with 100/500 instances and 2.65× at ratio 1 with 100 instances; these are prepacked mask-stage results, not full API, RSS or training claims. Full real-data non-overlap verification of this new binary completed all nine 5,000-image processes at workers 0/2/8 with exact reference/replay/native equality. See [MASKS_ONLY.md](MASKS_ONLY.md).
- Dirty scratch clearing, cached contour extents, conservative composition bounds and consecutive duplicate integer vertex removal are implemented without changing the OpenCV fill/resize sequence. Untyped byte snapshots correct the sanitizer failure on unaligned NumPy input.
- Earlier scratch-reuse snapshot, nine synthetic cases: wrapper geometric-mean speedup 2.05× on M2 and 2.41× on the i5-10400/3070 server (CPU operation). The initial slower M2 result is retained. These are not exact-current-source measurements.
- Earlier snapshot, Memray: 75 independent allocation runs on each host; automatic mode reduced working allocation 61–91% in the five measured N≥100 cases. Process RSS is reported separately; the current source needs exact-source reruns.
- Earlier project-code ASan/UBSan: 120 core tests passed on macOS, leak detection disabled, private OpenCV uninstrumented. Both the original alignment failure and the separate subprocess runtime-injection failure are retained.
- Raw measurements and scope limitations are in [BENCHMARKS.md](BENCHMARKS.md). Invalid inherited Linux ru_maxrss data is explicitly excluded; corrected VmHWM reruns are retained.
- Full COCO CPU DataLoader harness uses frozen input/upstream hashes, workers 0/2/8, separate full output verification, externally sampled process-family RSS, and fresh-reference cache rebuilding; see [COCO_LOADER.md](COCO_LOADER.md).
- Full COCO non-augmented CPU DataLoader: 60 fresh measured processes across two hosts and workers 0/2/8; all full outputs match, including an independent original-scanner cache rebuild on each host. The 10% full-loader performance gate remains unmet; family RSS is approximately unchanged. See [COCO_LOADER.md](COCO_LOADER.md) and the complete results in [BENCHMARKS.md](BENCHMARKS.md).
- Direct stage probes passed exact call-count and full-output checks. Native-side cProfile call accounting was invalid and is explicitly excluded from component timing conclusions; the rejected evidence is retained.
- The contour-bounds candidate and subsequent alignment correction are now merged. The candidate's separate 60-process full-loader experiment still misses the 10% gate and shows approximately unchanged family RSS. Its exact measured pre-correction source archive and all raw results are retained in [COCO_BOUNDS.md](COCO_BOUNDS.md). They must not be attributed to the subsequent alignment-corrected binary.
- Full overlap and non-overlap augmentation verification with both libraries completed on the alignment-corrected build: all 18 processes at workers 0/2/8 have 5,000-image reference/replay/native equality. These results predate the later masks-only kernel; see [COCO_BOUNDS.md](COCO_BOUNDS.md).
- Installed Linux CUDA-environment mask wheel passed 139 tests; wheel source/native bytes and RECORD are verified. Reference and combined-library GPU pilots each completed a real 5,000-image YOLO11n-seg epoch with identical initial/final model hashes and all 1,250 batch loss vectors. This is integration evidence, not a repeated speedup claim; see [CUDA_WHEEL_VALIDATION.md](CUDA_WHEEL_VALIDATION.md).
- The full 45-job / 90-epoch RTX 3070 series completed and passed the independent artifact/statistics audit: all 15 groups have exact loss/model equality. Epoch median ratios are 1.002–1.021×, with several intervals including 1. Sampled family RSS has no consistent reduction and increases at workers 2/8; CUDA allocator peak medians are unchanged. These are frozen-original-wheel overlap results, not later masks-only/ROI evidence. See [GPU_RESULTS.md](GPU_RESULTS.md).
- The earlier eight-trial checkpoint and its rejected initial auditor are retained. The auditor's 12 synthetic corruption/completion tests passed on M2; the complete real-series summary audit separately passed, as recorded in GPU_RESULTS.md.
- Native notice collection now covers 584 observed OpenCV/KleidiCV/pybind11 source/header files. A rebuilt M2 wheel and sdist preserve the collected files exactly; the installed wheel passed 120 core tests and archive/installed-byte checks. The wheel is currently macOS 26 arm64 only; remaining platform and redistribution review is explicit in [NATIVE_DEPENDENCIES.md](NATIVE_DEPENDENCIES.md).
- Remaining release gates: representative workload performance, augmented/non-overlap and higher-resolution coverage, exact-current-candidate GPU validation, Windows and supported Python matrix, full dependency license inventory, wheels/CI, exact release-candidate validation.
