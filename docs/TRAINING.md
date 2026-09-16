# Does the faster loader shorten training?

Short answer, on the machine measured here: **it depends on the worker
count.** With the default worker counts both pipelines are GPU-bound on an
RTX 3070, so the end-to-end training time changes by 0–2%. Where the loader is
the limit — no workers, one worker, or a host with few cores per GPU — the
training itself gets **1.09× to 1.54× faster**. The numbers below say exactly
where that boundary lies.

## Method

One fresh process per measurement, arms alternating, two rounds, the best of
each reported. Throughput is measured after a warmup with CUDA synchronized at
both marks; validation, plots and checkpoints never run.

- **RF-DETR**: `RFDETRSegSmall`/`RFDETRSegNano`'s own `train()` — PyTorch
  Lightning, EMA, bf16 AMP, batch 4, the default training configuration —
  through `bench/rfdetr_training.py`, which swaps only the training dataset
  (`build_dataset`) and adds a timing callback. 160 steps, 40 of them warmup.
- **Ultralytics**: `YOLO("yolo11n-seg.yaml")` with the default training
  configuration (batch 16, AMP, imgsz 640, mosaic and every default
  augmentation) through `bench/ultralytics_training.py`, reference
  `SegmentationTrainer` against `FastSegmentationTrainer`. One epoch of 313
  batches, 30 of them warmup.
- **Host**: i5-10400 (6 cores, 12 threads), 32 GB, RTX 3070 8 GB, driver
  580.173.02, torch 2.10.0+cu128, COCO val2017 (5,000 segmentation images).
  Load average was below 0.1 before each sweep.
- **Same training**: the first-step losses are identical between the arms
  (RF-DETR at 0 workers, where the sampling is reproducible, and Ultralytics
  in every run), and peak CUDA memory is unchanged.

## RF-DETR

Images per second of the whole training step, including the loader:

| Model | workers | RF-DETR loader | ultrafast-maskops | no loader cost (cached) | Speedup |
|---|---:|---:|---:|---:|---:|
| Seg-Small (384 px) | 0 | 13.3 | **14.6** | 16.9 | 1.09× |
| Seg-Small | 2 (default) | 16.8 | 16.9 | 17.1 | 1.01× |
| Seg-Small | 8 | 16.8 | 17.0 | 17.0 | 1.02× |
| Seg-Nano (312 px) | 0 | 15.7 | **17.1** | 19.8 | 1.09× |
| Seg-Nano | 2 (default) | 19.8 | 20.0 | 20.0 | 1.01× |
| Seg-Nano | 8 | 19.6 | 19.9 | 20.2 | 1.02× |

The "no loader cost" column serves 64 pre-built samples in a cycle, so it is
the ceiling the GPU and the training loop impose. At two workers RF-DETR's own
loader already reaches that ceiling: the GPU consumes 17–20 images per second
while the loader can produce 67 (reference) or 118 (ours), so the loader was
never the limit. Only at zero workers, where the loading runs in the training
process itself, does the faster loader show: 1.09×.

## Ultralytics

| workers | Ultralytics reference | with ultrafast-maskops | Speedup |
|---:|---:|---:|---:|
| 0 | 43.4 | **57.4** | **1.32×** |
| 1 | 65.6 | **100.9** | **1.54×** |
| 2 | 122.4 | 122.1 | 1.00× |
| 4 | 122.4 | 121.7 | 0.99× |
| 8 | 122.6 | 121.3 | 0.99× |

The training step caps at about 122 images per second. With at least one
worker the observed rate is the smaller of the loader's rate and that cap;
with no workers the two costs add, because the loading then runs inside the
training process. At one worker the reference loader delivers 68 images per
second and ours 105, so both sides are the limit and training is 1.54× faster;
from two workers on, the reference loader already passes 122 and the
difference disappears into run-to-run noise. All eight low-worker runs
alternate arms across two rounds and their first-step losses match.

## Where the loader does decide training time

Put the two measurements side by side. Training gets faster only while the
loader produces fewer images per second than the training step consumes:

| | Training step (3070) | Reference loader | ultrafast-maskops loader |
|---|---:|---:|---:|
| RF-DETR Seg-Small, batch 4 | 17.1 img/s | 41 (0 w) · 67 (2 w) · 122 (8 w) | 76 (0 w) · 118 (2 w) · 211 (8 w) |
| Ultralytics YOLO11n-seg, batch 16 | 122 img/s | 69 (0 w) · 128 (2 w) · 333 (8 w) | 112 (0 w) · 196 (2 w) · 452 (8 w) |

So the gain appears when:

- **the worker count is small** — measured above: 1.32× with no workers and
  1.54× with one for Ultralytics, 1.09× with no workers for RF-DETR. This is
  also a way to spend fewer workers: ours at one worker (101 img/s) beats the
  reference at one worker by half and gets most of the way to the GPU's 122,
  and each Ultralytics worker costs memory — the archived 3070 series measured
  3.5 GB of process-family RSS at zero workers, 7.3 GB at two and 16.6 GB at
  eight.
- **the host has few cores per GPU.** A node with eight GPUs and 32 cores
  gives four cores per GPU; the loader columns above are then the budget, and
  ours delivers 1.6–1.8× more images per core.
- **the GPU is faster than this one.** An H100 runs these models several times
  faster than a 3070, which moves the training step above the reference
  loader's rate on the same CPU.

## What it does not change

Peak CUDA memory is the same (RF-DETR 5.0–5.3 GB, Ultralytics 3.3 GB in both
arms), and the batches are identical, so accuracy and the training curve are
unchanged by construction.

Raw reports: [bench/results/training-gpu-linux-v1](../bench/results/training-gpu-linux-v1).
