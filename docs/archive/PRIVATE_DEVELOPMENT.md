# Private development snapshot — 2026-09-13

The owner approved uploading this project to `developer0hye/ultrafast-maskops`
as a private GitHub repository. This preserves development history and evidence;
it is not a release or a claim that all replacement gates have passed.

All five existing branches are retained. The default `feat/trainer-lifecycle`
branch contains the current persistent formatter/shared-batch implementation,
installed wheel qualifications and independent lifecycle auditor.
`feat/native-core`, `perf/coco-bounds`, `perf/resize-roi` and
`perf/unit-scale-mask` preserve the earlier implementation and optimization
checkpoints. Branches are not implicitly merged by this upload.

The 54-trial/126-epoch RTX 3070 lifecycle campaign is still running; its complete
result and independent final audit are pending. The full CPU loader comparison
has not met the 10% throughput target. Linux `file_system` transport remains
unqualified; the observed failures and stock-PyTorch controls are retained.
Current M2 functional tests passed, but broader lifetime/platform qualification
remains open. See [STATUS.md](STATUS.md) for scoped evidence and limitations.

Hosted wheel CI has not yet qualified this snapshot. Automatic Actions are
disabled for the initial multi-branch upload to avoid launching a separate
12-job wheel matrix for every historical development branch. Enable Actions
and select the intended source revision for the subsequent platform campaign.

Tracked raw reports, manifests and evidence archives are included. Large local
datasets, build environments and live experiment output remain on their hosts;
completed final evidence will be added after the running campaigns terminate.
