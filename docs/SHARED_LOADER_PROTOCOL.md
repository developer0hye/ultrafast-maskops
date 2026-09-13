# Shared collator real-data loader protocol: not executed

The CPU loader harness now accepts `--persistent-mask`. Both backends still use
the ordinary YOLODataset and the complete real 5,000-image segmentation fixture.
The reference retains its original formatter/collator. Native persistent mode
installs FastFormat and the shared packet collator. It records the actual loader
collator, and verifies installed Python runtime bytes against the frozen harness
source tree before execution. This protocol has not yet produced measurements.

Use both overlap modes, worker counts 0/2/8, five alternating fresh-process pairs,
batch size 8, image size 640, mask ratio 4, no augmentation, no image RAM cache,
and no pinning. The first epoch is timed with periodic process-family RSS
samples. A complete second pass hashes every batch field outside measurement.
Shutdown follows verification and must produce zero exits for every worker.
Original-reference shutdown failures remain failures; transport is not normalized
on the reference side of a performance comparison.

The report declares its entire round/worker plan before launching children and
marks completion only after all planned records and summaries exist. The final
fresh-reference verification checks plan coverage, actual collator identity,
worker count and exit codes before comparing outputs to a freshly rebuilt cache.
Run that destructive-to-generated-cache verification only after all timing jobs
using the cache finish. Image and annotation bytes remain unchanged.

This measures full CPU loader behavior, not close-mosaic transitions or training.
Shared pages are counted in each process RSS; the summed family peak is not
unique physical memory. The separate GPU lifecycle/resume protocol and direct
allocation evidence remain necessary. A speed improvement is unproven until the
full repeated measurements and independent artifact audit complete.

The harness source can be frozen separately from its installed wheel, but all
runtime Python files and native extension identities must match the measured
wheel. Changes to benchmark scripts do not replace the installed wheel's test
receipt. Prior loader results belong to their original immutable harness and
candidate; they cannot qualify the packet revision.
