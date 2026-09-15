# Measurement index

- `m2-initial.json`: initial slower implementation, retained without selection.
- `m2-support.json`: optimized synthetic wrapper latency; macOS process RSS.
- `server-vmhwm.json`: optimized synthetic latency and corrected Linux process VmHWM.
- `server-support.json`: historical latency snapshot; **all process memory fields are invalid because ru_maxrss inherited the parent high-water mark**. Do not use this file for memory comparisons.
- `m2-allocations.json`, `server-allocations.json`: independent Memray heap-allocation experiments, no latency claims. Referenced binary trace files are retained locally/on the server and are not included in this repository.
- `coco-source-manifest.json`, `coco-segment-manifest.json`: frozen official COCO images/annotations, converter and complete converted Segmentation fixture provenance.
- `coco-loader-pilot-m2.json`: 64-image execution/parity smoke at workers 0/2. One run per backend/configuration; its timing and degenerate bootstrap intervals are not representative performance evidence.
- `mask-stage-sampled-m2-v1.json`, `.log`: overlap-mask stage (`bench/mask_stage.py`) on 2,000 captured COCO Format calls at commit `490eed2`. Reference, the full-resolution native path and the sampled native path, five alternating rounds; every output is compared with the unmodified function first. The log records the commit and the load average around the run.
- `geometry-loader-sampled-m2-v1.json`: augmented COCO val2017 segmentation `__getitem__` (`bench/geometry_loader.py`) on the M2 at the same commit, 1,000 samples and five alternating rounds per backend, with identical output digests.
- `mask-stage-sampled-linux-v1.json`: the same mask-stage measurement on an i5-10400 (Linux) at commit `54fc9f6`; `sampled-linux-v1-env.txt` records the host, compiler and commit.
- `geometry-loader-sampled-linux-v1.json`: augmented COCO val2017 segmentation `__getitem__` (`bench/geometry_loader.py`) on the same host and commit, 1,000 samples and five alternating rounds per backend, with identical output digests.
- `coco-epoch-hit-linux-v1.json`, `coco-epoch-miss-linux-v1.json`: one augmented training epoch (`bench/coco_epoch.py`) over a 118,287-image COCO-scale segmentation corpus on the same host. Maskops is at `54fc9f6` and ultrafast-yolo-dataset at `ad1d738`. The label cache is hit (three rounds) or missed (two rounds), and the first 64 batches have identical verification digests.
- `coco-epoch-hit-maskops-linux-v1.json`: the same hit-mode epoch with only the maskops calls on the unmodified YOLODataset (`--backends reference maskops`), three alternating rounds, identical verification digests.
- `mask-stage-minimal-linux-v1.json`: the mask stage on the i5-10400 with the restructured, OpenCV-free kernel (one vector pass per contour, crossing buckets, row bit sets; SSSE3 pass), five alternating rounds.
- `mask-stage-sampled-windows-v1.json`: the mask stage on an Intel i5-12600 (Windows 11, MSVC 19.44 build) at commit `3c04fd6`, five alternating rounds; `sampled-windows-v1-env.txt` records the host, toolchain and commit. The 2,000-call input was regenerated with `bench/capture_format_inputs.py` from the val2017 fixture with the same seeds and holds the same 27,497 instances.
- `geometry-loader-sampled-windows-v1.json`: augmented COCO val2017 segmentation `__getitem__` (`bench/geometry_loader.py`) on the same host and commit, 1,000 samples and five alternating rounds per backend, with identical output digests.
- `gpu-training-b16-fp32-windows-v1.json`, `gpu-training-b16-amp-windows-v1.json`: end-to-end YOLO11n-seg training (`bench/gpu_training.py`) on the same host with a TITAN RTX, batch 16, two epochs, fresh process per job, workers 0/2/8 in FP32 (three repeats) and 2/8 with AMP (two repeats). Per-epoch CUDA-synchronized wall times, whole-job times and the hashes of every job's per-batch losses and final weights (identical between the reference and accelerated trainers in every configuration).

Full methodology, unfavorable cases and incomplete release gates: [report](../../docs/archive/BENCHMARKS.md).
