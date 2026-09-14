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

Full methodology, unfavorable cases and incomplete release gates: [report](../../docs/BENCHMARKS.md).
