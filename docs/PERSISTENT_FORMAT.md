# Persistent Format candidate

This branch adds an **unvalidated** optional adapter feature on top of the
unit-scale mask candidate at `25853ef`. The new source parses, but a fresh wheel,
installed tests and training/lifecycle checks have not run. Both hosts remain
reserved by the existing full dataset-startup measurements. The earlier 209-test
Linux result and loader/GPU evidence predate this Python adapter change.

## Behavior

The existing `accelerate_dataset(dataset)` changes the current Compose list.
Ultralytics `YOLODataset.close_mosaic` later calls `build_transforms`, replacing
that list. The pinned builder instantiates its final formatter through
`self.format_class`, whose default is the reference `Format`.

The candidate adds an explicit keyword option:

```python
from ultrafast_maskops.ultralytics import accelerate_dataset

# In the dataset factory, before DataLoader workers are created:
replaced = accelerate_dataset(segmentation_dataset, persistent=True)
```

This replaces current base mask formatters and sets only that dataset instance's
`format_class` to the module-level `FastFormat`. Future base `build_transforms`
calls therefore create native formatters with the new reference-selected mask
ratio, overlap setting and all inherited Format options. No trainer override,
global class replacement, model/loss change, or DataLoader-reset override is
introduced. The default `persistent=False` keeps the existing one-shot behavior.

The return value remains the number of current formatters replaced. Repeated
persistent opt-in, or upgrading an already accelerated current list to persistent
mode, returns zero. The instance factory is pickleable, and FastFormat already
discards its per-process native engine when pickled or crossing a PID boundary;
the new tests must verify those properties through a rebuilt dataset too.

Persistent mode requires the pinned base segmentation `build_transforms` method
and a Format/FastFormat factory. Its method source is checked alongside the
existing mask-function profile. Custom factories, overridden builders, custom
Format subclasses, and non-default FastFormat mode/budget require their own
integration and are rejected before changing the instance/list. Rebuilding uses
the default FastFormat auto mode and 64 MiB scratch budget. The scratch budget
does not cap total process memory.

This option does not enable native annotation scanning or change cache policy.
The dataset extension's FastYOLODataset inherits the base transform builder, but
combined-library execution remains an explicit validation gate. Constructing an
ordinary reference dataset is still the documented way to opt out.

## Validation prepared, not executed

[New tests](../tests/test_persistent_format.py) cover instance/global isolation,
unchanged one-shot behavior, repeat/upgrade calls, pickle and direct rebuilds,
unknown/custom builder/factory rejection without mutation, preservation of
non-default settings by rejecting unsupported persistence, and strict flag type.
They also call the actual pinned trainer close-mosaic method, reset real
InfiniteDataLoader workers at counts 0 and 2, verify prior workers terminate,
and compare every batch field for both overlap and non-overlap masks. Other random
augmentations are disabled for this focused equality test; mosaic/mixup/copy-paste/
cutmix are enabled before the transition and disabled by the reference method.

[Reviewed source identities](persistent-format-source-profile-v1.json) locate
the builder, close-mosaic, epoch/resume callers and loader reset implementation.
These were extracted from source without importing the active benchmark runtime.
Only the base transform builder is an additional runtime profile guard; the other
entries document the reviewed context. Runtime inspect-source agreement still
needs to pass in the installed environment.

Before merging, build a fresh wheel/sdist and run the entire installed suite on
both hosts, then verify combined-library datasets and actual CPU/GPU training
through a close-mosaic transition and resumed training. The focused loader tests
do not execute a training loop or establish loss/model parity after that transition.
Any throughput claim needs a separately declared repeated benchmark; the earlier
GPU protocol used `close_mosaic=0` and cannot prove this lifecycle behavior.

No build, test or benchmark is automatically queued by this candidate. Start its
qualification only after the selected host's current reservation ends.
