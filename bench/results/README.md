# Measurement index

- `m2-initial.json`: initial slower implementation, retained without selection.
- `m2-support.json`: optimized synthetic wrapper latency; macOS process RSS.
- `server-vmhwm.json`: optimized synthetic latency and corrected Linux process VmHWM.
- `server-support.json`: historical latency snapshot; **all process memory fields are invalid because ru_maxrss inherited the parent high-water mark**. Do not use this file for memory comparisons.
- `m2-allocations.json`, `server-allocations.json`: independent Memray heap-allocation experiments, no latency claims. Referenced binary trace files are retained locally/on the server and are not included in this repository.
- `coco-source-manifest.json`, `coco-segment-manifest.json`: frozen official COCO images/annotations, converter and complete converted Segmentation fixture provenance.
- `coco-loader-pilot-m2.json`: 64-image execution/parity smoke at workers 0/2. One run per backend/configuration; its timing and degenerate bootstrap intervals are not representative performance evidence.

Full methodology, unfavorable cases and incomplete release gates: [report](../../docs/BENCHMARKS.md).
