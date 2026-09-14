# Independent GPU lifecycle artifact audit

The [new auditor](../bench/audit_lifecycle_gpu.py) checks the complete frozen
54-trial/126-epoch artifact graph without importing the coordinator, Torch or
pickle. Its checker qualification passed **46 synthetic tests**, and its
per-trial checks passed on the first completed real reference and mask-only
trials. The full GPU grid is still running; it has **not** passed this final audit.

## What is checked

The full audit requires all six worker/mask conditions, all three backends and
all three lifecycle stages in the declared order. It rejects incomplete or
failed runs, duplicate/omitted conditions, altered child commands and misleading
completion flags. Raw report/log hashes, frozen harness files, the original
reference cache bytes and its fresh-verification receipt must match.

For every epoch, it checks 5,000 images, 1,250 batches, every finite loss value,
positive foreground segmentation loss, CUDA peak ordering and timing arithmetic.
It recomputes every loss digest and the parent's complete compacted report from
the raw arrays. Sampled family-RSS peaks must equal the maximum raw observation.

The lifecycle checks cover formatter/factory/collator retention, pinned images,
actual multiprocessing context and prefetch setting, expected transform rebuilds,
worker replacement history, disjoint replacement pools and clean final train and
validation shutdown. Validation worker counts are independently derived from the
pinned builder's `min(2 * workers, cpu_count)` rule for this one-GPU, full-dataset
protocol, rather than merely accepting the raw report's own declared count.

Every retained checkpoint must have the expected path, actual byte count and
SHA-256. Resumed trials must refer to the same original reference checkpoint and
receipt, with the expected checkpoint epoch and positive recorded optimizer/EMA
restoration counts. All loss values and initial/final model digests must match
within each three-backend cohort. Parent cohort/runtime summaries are rebuilt
and compared. The total must be exactly 54 trials, 126 epochs and 18 cohorts.

The auditor does not deserialize checkpoint tensors or independently establish
held-out accuracy. Resumed cohorts are compared internally; they are not required
to equal uninterrupted training. Resource samples do not prove unsampled peaks,
and summed RSS counts shared pages more than once. This is not a repeated
throughput experiment.

## Checker qualification and real controls

The [46 tests](../tests/test_lifecycle_gpu_audit.py) include two complete/relocated
synthetic graph controls, 40 damaged-evidence cases, three invalid-JSON cases and
a CLI failure/overwrite-preservation check. Damaged cases update checksums and
parent compaction where appropriate, forcing semantic checks to reject them.
The generated fixtures are explicitly synthetic; they are not benchmark results.
The production auditor does not reuse the coordinator's validators. Existing
coordinator fixtures are used only to generate positive test artifacts.

All tests passed on the local M2 Python environment, with no failures or skips;
Ruff and actionlint 1.7.12 also passed. The framework-free test module is selected
by every prepared wheel workflow core job. Hosted Linux/Windows/macOS matrix
execution is still pending; local checker tests are not matrix evidence.

The actual `w2-yes-fresh-reference` trial completed four full epochs, retained
both checkpoint files, observed the close-mosaic worker replacement and exited
cleanly. Its raw/log hashes, all loss vectors, complete parent compaction,
checkpoint bytes and cache/source bindings passed the separate checker. The
actual incomplete grid was explicitly rejected by the full-audit entry point.

The subsequent `w2-yes-fresh-mask` trial also passed per-trial checks. Direct
canonical comparison found **every reported loss value and both model digests
identical** to the reference over all four epochs. This proves the recorded
first worker/mask condition. Separate audits of combined-backend and resumed
trials remain pending. The coordinator subsequently completed the first fresh
three-backend cohort; its full-grid independent audit is still pending. The two
separately audited executions are not repeated performance evidence.

Evidence is preserved in:

- [Checker and reference inputs](../bench/results/mask-lifecycle-auditor-m2-v1-evidence.tar.gz):
  24 members, 30,512,564 bytes, SHA-256
  `3bd68335d231cf64f9021a1b9d9a873fbde8f9cb550bfb79d9df38754106ab50`.
  Includes exact auditor/test sources, logs/JUnit, workflow/tool identities,
  original reference cache, raw loss/resource data and both reference checkpoints.
- [Mask-only trial inputs](../bench/results/mask-lifecycle-real-mask-v1-evidence.tar.gz):
  five members, 21,615,766 bytes, SHA-256
  `ea34e2c089e5ab4b83bbe2bc2be0218c10f3ad85c9967681858c6b3761c833f2`.
  Includes its raw report/log, both checkpoints and a contemporaneous incomplete
  grid snapshot.

All archive members were read back and hashed. See the
[checker qualification](validation/mask-lifecycle-auditor-m2-v1-qualification.json),
[reference audit](validation/mask-lifecycle-real-reference-audit-v1.json),
[additional file bindings](validation/mask-lifecycle-real-reference-bindings-v1.json),
[mask audit](validation/mask-lifecycle-real-mask-audit-v1.json), and preservation
manifests for the [checker/reference](validation/mask-lifecycle-auditor-m2-v1-preservation.json)
and [mask trial](validation/mask-lifecycle-real-mask-v1-preservation.json).
The earlier auditor source attached to the first reference receipt is preserved
exactly; its trial-check function is unchanged in the final qualified source.

## Run after the complete grid has terminated

Use the qualified auditor with the frozen trial sources, not the latest edited
training source. On the current server the inputs are:

```sh
python /home/yonghye/ultrafast-vision-build/audit-lifecycle-gpu-linux-v1.py \
  --grid /home/yonghye/ultrafast-vision-build/mask-lifecycle-gpu-linux-v1-grid.json \
  --runs /home/yonghye/ultrafast-vision-build/mask-lifecycle-gpu-linux-v1-grid.runs \
  --source /home/yonghye/ultrafast-vision-build/mask-lifecycle-gpu-linux-v1-source \
  --fresh-check /home/yonghye/ultrafast-vision-build/mask-lifecycle-gpu-linux-v1-fresh-reference.json \
  --reference-cache /home/yonghye/ultrafast-vision-build/mask-lifecycle-gpu-linux-v1-original-reference.cache \
  --out /home/yonghye/ultrafast-vision-build/mask-lifecycle-gpu-linux-v1-final-audit.json
```

The qualified script has been staged at that server path with SHA-256
`e52082761e35ff9cf0a7ec8d469a1f3cfdbcc690afb83004a03fe8c1c643cc2e`.
The full audit has not been run successfully yet. Output paths are exclusive:
failures produce a failed receipt, and another invocation cannot overwrite it.
The current training campaign, remaining outcomes and final preservation must
still be checked before any full lifecycle qualification claim.
