"""One full, local-only Ultralytics GPU training trial; repeat in fresh processes.

This is a performance fixture, not an accuracy experiment: train and validation
both use the fixed 5,000-image COCO val2017 conversion. No pretrained downloads.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--fresh-check", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("reference", "mask", "both"), required=True)
    parser.add_argument("--workers", type=int, choices=(0, 2, 8), default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--overlap", choices=("yes", "no"), default="yes")
    return parser.parse_args()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    args = arguments()
    assert args.epochs >= 1 and args.batch >= 1 and args.imgsz >= 32
    args.out = args.out.resolve()
    run_dir = args.out.with_suffix(".run")
    assert not args.out.exists() and not run_dir.exists(), "retain prior trials; use a new output"
    run_dir.mkdir(parents=True)
    os.environ["YOLO_CONFIG_DIR"] = str(run_dir / "config")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_VERBOSE"] = "false"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import cv2
    import numpy as np
    import psutil
    import torch
    import ultrafast_maskops as maskops
    import ultrafast_yolo_dataset as datasetops
    import ultralytics
    from coco_loader import COCO_FINGERPRINT, fingerprint, validate_profile
    from ultrafast_maskops.ultralytics import accelerate_dataset
    from ultrafast_yolo_dataset.ultralytics import build_yolo_dataset, check_profile
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    from ultralytics.utils.callbacks import get_default_callbacks
    from ultralytics.utils.torch_utils import unwrap_model

    validate_profile()
    check_profile()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    for package in (maskops, datasetops):
        assert Path(package.__file__).is_relative_to(sys.prefix), "use installed wheels, not editable packages"
    root = args.corpus.resolve()
    assert fingerprint(root) == COCO_FINGERPRINT
    fresh = json.loads(args.fresh_check.read_text())
    cache = root / "labels/val2017.cache"
    # The fresh verifier binds the original scanner's cache to the full corpus.
    cache_sha = file_sha(cache)
    assert cache_sha == fresh["fresh_cache_sha256"]
    assert fresh["fresh_reference"]["images"] == 5000
    fixture_yaml = run_dir / "fixture.yaml"
    fixture_yaml.write_text(
        json.dumps(
            {
                "path": str(root),
                "train": "images/val2017",
                "val": "images/val2017",
                "names": list(map(str, range(80))),
            }
        )
        + "\n"
    )
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    upstream = Path(ultralytics.__file__).parent
    source_names = (
        "engine/trainer.py",
        "engine/validator.py",
        "models/yolo/detect/train.py",
        "models/yolo/segment/train.py",
        "models/yolo/segment/val.py",
        "nn/tasks.py",
        "utils/loss.py",
        "utils/torch_utils.py",
        "data/build.py",
        "cfg/models/11/yolo11-seg.yaml",
    )
    source_hashes = {name: file_sha(upstream / name) for name in source_names}
    packages = {}
    for package in (maskops, datasetops):
        package_root = Path(package.__file__).parent
        packages[package.__name__] = {
            "path": str(package_root),
            "files": {
                p.name: file_sha(p) for p in sorted(package_root.iterdir()) if p.suffix in (".py", ".json", ".so")
            },
        }

    def state_sha(model):
        digest = hashlib.sha256()
        for name, value in model.state_dict().items():
            array = value.detach().cpu().contiguous().numpy()
            digest.update(f"{name}:{array.dtype}:{array.shape}:".encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    class LocalSegmentationRun(SegmentationTrainer):
        # Dispatch only this benchmark's callbacks, including during base init.
        # No module globals or installed framework functions are patched.
        def run_callbacks(self, event):
            callback = getattr(self, "measure_" + event, None)
            if callback:
                callback()

        def get_dataset(self):
            return {
                "path": root,
                "train": str(root / "images/val2017"),
                "val": str(root / "images/val2017"),
                "nc": 80,
                "channels": 3,
                "names": dict(enumerate(map(str, range(80)))),
            }

        def build_dataset(self, img_path, mode="train", batch=None):
            started = time.perf_counter()
            if args.backend == "both":
                stride = max(int(unwrap_model(self.model).stride.max()), 32)
                dataset = build_yolo_dataset(
                    self.args,
                    img_path,
                    batch,
                    self.data,
                    mode=mode,
                    rect=mode == "val",
                    stride=stride,
                    annotation_cache="native",
                    cache_dir=run_dir / "native-cache",
                )
            else:
                dataset = super().build_dataset(img_path, mode, batch)
            replaced = accelerate_dataset(dataset) if args.backend != "reference" else 0
            assert len(dataset) == 5000 and replaced == int(args.backend != "reference")
            self.dataset_records.append(
                {
                    "mode": mode,
                    "seconds": time.perf_counter() - started,
                    "images": len(dataset),
                    "replaced_formats": replaced,
                    "native_cache_hit": getattr(dataset, "annotation_cache_hit", None),
                }
            )
            return dataset

        def get_validator(self):
            validator = super().get_validator()
            # Validation retains its normal algorithm; callbacks stay local.
            validator.callbacks = {name: [] for name in get_default_callbacks()}
            return validator

        def measure_on_pretrain_routine_end(self):
            self.initial_state_sha = state_sha(unwrap_model(self.model))
            assert self.train_loader.num_workers == args.workers
            assert self.batch_size == args.batch and not self.amp

        def measure_on_train_epoch_start(self):
            assert self.batch_size == args.batch and not getattr(self, "_oom_retries", 0)
            self.seen_images = 0
            self.batch_losses = []
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.epoch_started = time.perf_counter()

        def preprocess_batch(self, batch):
            self.seen_images += len(batch["img"])
            return super().preprocess_batch(batch)

        def measure_on_train_batch_end(self):
            # Retain small detached GPU loss vectors; inspect outside epoch timing.
            self.batch_losses.append(torch.stack([v.detach() for v in self.loss_items.values()]))

        def measure_on_train_epoch_end(self):
            torch.cuda.synchronize()
            seconds = time.perf_counter() - self.epoch_started
            losses = torch.stack(self.batch_losses).cpu()
            assert self.seen_images == 5000
            assert len(losses) == math.ceil(5000 / args.batch)
            assert torch.isfinite(losses).all() and not self._oom_retries
            names = list(self.loss_items)
            assert losses[:, names.index("seg_loss")].mean() > 0
            self.epoch_records.append(
                {
                    "epoch": self.epoch,
                    "seconds": seconds,
                    "images": self.seen_images,
                    "batches": len(losses),
                    "images_per_second": self.seen_images / seconds,
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "loss_names": names,
                    "all_batch_losses": losses.tolist(),
                }
            )

    overrides = {
        "model": "yolo11n-seg.yaml",
        "data": str(fixture_yaml),
        "pretrained": False,
        "device": "0",
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "seed": 912,
        "deterministic": True,
        "amp": False,
        "optimizer": "SGD",
        "lr0": 0.001,
        "cache": False,
        "rect": False,
        "fraction": 1.0,
        "mask_ratio": 4,
        "overlap_mask": args.overlap == "yes",
        "close_mosaic": 0,
        "multi_scale": 0.0,
        "compile": False,
        "plots": False,
        "val": False,
        "save": False,
        "project": str(run_dir),
        "name": "training",
        "exist_ok": False,
        "verbose": False,
    }
    report = {
        "complete": False,
        "scope": "full real-data GPU training trial; not accuracy or repeated speedup evidence",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "script_sha256": file_sha(__file__),
        "upstream_files": source_hashes,
        "packages": packages,
        "mask_build": maskops.backend_info(),
        "fixture": COCO_FINGERPRINT,
        "original_cache_sha256": cache_sha,
        "fresh_check_sha256": file_sha(args.fresh_check),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "cpu_count": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "load_before": os.getloadavg(),
        "timing": "epoch start/end callbacks with CUDA synchronization; excludes setup, validation, checkpoint saving; first epoch includes GPU warmup; upstream may prefetch during loader construction",
        "memory": "CUDA allocator peaks per epoch; sampled summed process-family RSS double-counts shared pages",
        "callbacks": "benchmark callbacks only; external logger and analytics callbacks disabled",
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    samples, stop = [], threading.Event()

    def sample_resources():
        parent = psutil.Process()
        while not stop.is_set():
            processes = [parent, *parent.children(recursive=True)]
            rss = 0
            for process in processes:
                try:
                    rss += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            samples.append({"time": time.time(), "family_rss_bytes": rss, "load": os.getloadavg()})
            stop.wait(0.2)

    report["nvidia_smi_before"] = subprocess.check_output(["nvidia-smi"], text=True)
    monitor = threading.Thread(target=sample_resources, daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        trainer = LocalSegmentationRun(overrides=overrides)
        trainer.dataset_records, trainer.epoch_records = [], []
        trainer.train()
        whole_job_s = time.perf_counter() - started
        assert len(trainer.epoch_records) == args.epochs
        final_sha = state_sha(unwrap_model(trainer.model))
        assert final_sha != trainer.initial_state_sha, "model must actually update"
        assert file_sha(cache) == cache_sha, "original cache changed"
        assert source_hashes == {name: file_sha(upstream / name) for name in source_names}
        report.update(
            complete=True,
            whole_job_seconds=whole_job_s,
            resolved_config=vars(trainer.args),
            datasets=trainer.dataset_records,
            epochs=trainer.epoch_records,
            initial_state_sha256=trainer.initial_state_sha,
            final_state_sha256=final_sha,
            train_workers=trainer.train_loader.num_workers,
            val_workers=trainer.test_loader.num_workers,
            nvidia_smi_after=subprocess.check_output(["nvidia-smi"], text=True),
        )
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        stop.set()
        monitor.join()
        report["resource_samples"] = samples
        report["sampled_peak_family_rss_bytes"] = max((s["family_rss_bytes"] for s in samples), default=0)
        args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                "complete": report["complete"],
                "epochs": [{k: v for k, v in e.items() if k != "all_batch_losses"} for e in report["epochs"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
