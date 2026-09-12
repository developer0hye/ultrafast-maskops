# Measurement index

- `m2-initial.json`: initial slower implementation, retained without selection.
- `m2-support.json`: optimized synthetic wrapper latency; macOS process RSS.
- `server-vmhwm.json`: optimized synthetic latency and corrected Linux process VmHWM.
- `server-support.json`: historical latency snapshot; **all process memory fields are invalid because ru_maxrss inherited the parent high-water mark**. Do not use this file for memory comparisons.
- `m2-allocations.json`, `server-allocations.json`: independent Memray heap-allocation experiments, no latency claims. Referenced binary trace files are retained locally/on the server and are not included in this repository.

Full methodology, unfavorable cases and incomplete release gates: [report](../../docs/BENCHMARKS.md).
