# Training validation in progress

The repeated non-augmented COCO DataLoader experiment remains separate from
training-target and model-update correctness. The following new checks must run
sequentially after that host's benchmark has stopped; they are not performance
samples and must not overlap a timed experiment.

## Augmented real-data targets

`bench/verify_augmented_coco.py` requires a complete full-loader benchmark and
its fresh original-scanner verification. It binds the loaded extension, compiled
source profile, source files, fixture fingerprint and derived cache to those
artifacts before each process. It does not rebuild or replace the label cache.

For a newly corrected candidate, `--reference-only-baseline` uses the old report
solely to prove the original cache's provenance. It still requires the current
extension's compiled hashes to match the current source files and records both
identities explicitly. The old performance numbers do not apply to that new
candidate. This permits correctness verification before another timed experiment.

For each requested worker count, it starts independent reference, reference
replay and native processes with the same seed. Reference replay must agree
before native is accepted. It hashes every image, mask, semantic target, class,
box, batch index and metadata field, with per-batch hashes for mismatch location.
Different worker counts have different RNG streams and are compared separately.

The fixed stress configuration enables Mosaic, MixUp and flip Copy-Paste at
probability 1, plus affine/perspective, HSV and horizontal/vertical flips. This
is a correctness stress workload, not the default training recipe. Non-overlap
and overlap masks are separate runs. A pilot restricts primary output samples;
mixing may still read any of the 5,000 real images.

`--candidate both` additionally substitutes FastYOLODataset with its native
content-validated cache, then applies FastFormat. References still use the
ordinary YOLODataset and Format. The dataset package's runtime Python/profile
files and extension hash are recorded and checked between processes. Native
cache files live only in the new report's `.runs/native-cache` directory. This
checks the composition of the two libraries as well as mask-only replacement.

```sh
python bench/verify_augmented_coco.py \
  --corpus /path/to/coco/segment --benchmark /path/to/full-loader.json \
  --fresh-check /path/to/fresh-reference.json \
  --out /path/to/new-augmented-overlap.json --workers 0 2 8
python bench/verify_augmented_coco.py \
  --corpus /path/to/coco/segment --benchmark /path/to/full-loader.json \
  --fresh-check /path/to/fresh-reference.json \
  --out /path/to/new-augmented-nonoverlap.json --workers 0 2 8 --overlap no
```

Start with `--limit 64 --workers 0 2` to verify the harness. A pilot does not
establish full-corpus correctness. Only a report with `complete=true` establishes
that all its requested comparisons finished successfully.

## Deterministic CPU model updates

`tests/test_training.py` uses the installed YOLO11n-seg architecture, random
weights, two-image 128×128 batches and two SGD steps. It compares exact loss
components, all parameter gradients and final model parameters/buffers between
Format and FastFormat. Foreground cases must produce positive segmentation loss;
background-only cases check empty-target backward behavior. Both overlap modes
are included. CPU threads and deterministic algorithms are controlled and
restored afterward. This supports target compatibility, not training accuracy
or GPU throughput.

## GPU environment selection

The CPU benchmark environment remains separate from a newly installed CPython
3.12 Linux environment using PyTorch 2.10.0+cu128 and torchvision 0.25.0+cu128.
Exact filenames, official index URLs and SHA-256 values
are recorded in `cuda-wheel-selection.json`, checked against the
[PyTorch wheel index](https://download.pytorch.org/whl/cu128/torch/) and
[torchvision wheel index](https://download.pytorch.org/whl/cu128/torchvision/).
The new environment recognized the RTX 3070 (sm_86) and passed a finite GPU
matrix-operation forward/backward smoke. Requirements, installed versions and
that limited smoke report are retained under `docs/validation`. It does not
establish library integration or training throughput. Full GPU epochs, model
finite-loss checks and independently repeated throughput/memory measurements
remain open.
