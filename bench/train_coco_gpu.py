"""One full, local-only Ultralytics GPU training trial; repeat in fresh processes.

This is a performance fixture, not an accuracy experiment: train and validation
both use the fixed 5,000-image COCO val2017 conversion. No pretrained downloads.
"""

import argparse
import hashlib
import json
import math
import multiprocessing
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
    parser.add_argument("--close-mosaic", type=int, default=0)
    parser.add_argument("--persistent-mask", action="store_true")
    parser.add_argument("--checkpoint-epochs", type=int, nargs="+", default=[])
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--resume-receipt", type=Path)
    return parser.parse_args()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    args = arguments()
    harness_sources = {
        name: file_sha(Path(__file__).with_name(name)) for name in ("train_coco_gpu.py", "coco_loader.py")
    }
    assert args.epochs >= 1 and args.batch >= 1 and args.imgsz >= 32
    assert 0 <= args.close_mosaic < args.epochs
    assert not args.close_mosaic or args.backend == "reference" or args.persistent_mask
    assert len(set(args.checkpoint_epochs)) == len(args.checkpoint_epochs)
    assert all(0 <= epoch < args.epochs - 1 for epoch in args.checkpoint_epochs)
    assert bool(args.resume_from) == bool(args.resume_receipt)
    lifecycle = bool(args.close_mosaic or args.persistent_mask or args.resume_from or args.checkpoint_epochs)
    resume_receipt, resume_sha, resume_receipt_sha, expected_start = None, None, None, 0
    resume_state = None
    if args.resume_from:
        assert args.close_mosaic and not args.checkpoint_epochs
        args.resume_from = args.resume_from.resolve()
        args.resume_receipt = args.resume_receipt.resolve()
        receipt_bytes = args.resume_receipt.read_bytes()
        resume_receipt = json.loads(receipt_bytes)
        resume_receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
        del receipt_bytes
        assert resume_receipt["complete"] and resume_receipt["args"]["backend"] == "reference"
        assert resume_receipt["script_sha256"] == file_sha(__file__)
        assert resume_receipt["harness_sources"] == harness_sources
        assert resume_receipt["corpus_root"] == str(args.corpus.resolve()), "resume corpus path differs"
        for key in ("workers", "epochs", "batch", "imgsz", "overlap", "close_mosaic", "persistent_mask"):
            assert resume_receipt["args"][key] == getattr(args, key), f"resume protocol differs: {key}"
        matches = [c for c in resume_receipt["checkpoints"] if Path(c["path"]).resolve() == args.resume_from]
        assert len(matches) == 1, "resume only a retained checkpoint from the supplied completed reference trial"
        resume_sha = file_sha(args.resume_from)
        assert resume_sha == matches[0]["sha256"]
        expected_start = matches[0]["epoch"] + 1
        assert 0 < expected_start < args.epochs
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
    from ultrafast_maskops.ultralytics import FastFormat, accelerate_dataset
    from ultrafast_yolo_dataset.ultralytics import build_yolo_dataset, check_profile
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    from ultralytics.data.augment import Format
    from ultralytics.data.dataset import YOLODataset
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
        "cfg/__init__.py",
    )
    source_hashes = {name: file_sha(upstream / name) for name in source_names}
    if lifecycle:
        assert source_hashes["engine/trainer.py"] == "2eda410e0bfb8b65cb46eb975c465c5b62b0f3a3dad5a54967d6e9e1fcabecc7"
        assert source_hashes["cfg/__init__.py"] == "4f4deef636360243c08c32f2134bee31f497df7394c3b22c60c58f5452bcee85"
    packages = {}
    for package in (maskops, datasetops):
        package_root = Path(package.__file__).parent
        packages[package.__name__] = {
            "path": str(package_root),
            "files": {
                p.name: file_sha(p) for p in sorted(package_root.iterdir()) if p.suffix in (".py", ".json", ".so")
            },
        }
    if resume_receipt:
        assert resume_receipt["packages"] == packages and resume_receipt["upstream_files"] == source_hashes
        assert resume_receipt["fixture"] == COCO_FINGERPRINT
        assert resume_receipt["fresh_check_sha256"] == file_sha(args.fresh_check)
        assert resume_receipt["original_cache_sha256"] == cache_sha
        # Only deserialize the exact locally generated checkpoint bound above
        # to a completed reference receipt, matching script/runtime and inputs.
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] + 1 == expected_start
        assert checkpoint["optimizer"] and checkpoint["optimizer"]["state"]
        assert checkpoint["ema"] is not None and checkpoint["updates"] > 0
        resume_state = {
            "checkpoint_epoch": checkpoint["epoch"],
            "optimizer_states": len(checkpoint["optimizer"]["state"]),
            "ema_updates": checkpoint["updates"],
        }
        del checkpoint

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
            replaced = 0
            if args.backend != "reference":
                replaced = (
                    accelerate_dataset(dataset, persistent=True)
                    if args.persistent_mask
                    else accelerate_dataset(dataset)
                )
            assert len(dataset) == 5000 and replaced == int(args.backend != "reference")
            if mode == "train":
                self.initial_transforms_id = id(dataset.transforms)
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

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
            if lifecycle and mode == "train":
                # Resume may reset inside setup, before any epoch-start callback.
                # Retain only parent Process handles from the unmodified factory.
                self.setup_workers = list(getattr(loader.iterator, "_workers", ()))
            return loader

        def measure_on_pretrain_routine_end(self):
            self.initial_state_sha = state_sha(unwrap_model(self.model))
            assert self.train_loader.num_workers == args.workers
            assert self.batch_size == args.batch and not self.amp
            assert self.epochs == args.epochs and self.start_epoch == expected_start
            assert self.save_dir.resolve() == (run_dir / "training").resolve()
            if resume_state:
                assert len(self.optimizer.state) == resume_state["optimizer_states"]
                assert self.ema.updates == resume_state["ema_updates"]
            if lifecycle:
                closed_on_resume = bool(args.resume_from) and expected_start > args.epochs - args.close_mosaic
                self.observe_format("prepared", closed_on_resume)
                if closed_on_resume:
                    self.observe_replacement(self.setup_workers)
                self.setup_workers = []

        def observe_replacement(self, previous):
            current = list(getattr(self.train_loader.iterator, "_workers", ()))
            observation = self.lifecycle_records[-1]
            observation.update(
                replaced_worker_pids=[p.pid for p in previous],
                replaced_worker_exitcodes=[p.exitcode for p in previous],
                replaced_workers_alive=[p.is_alive() for p in previous],
            )
            assert len(previous) == len(current) == args.workers
            assert not any(observation["replaced_workers_alive"]), observation
            assert all(code == 0 for code in observation["replaced_worker_exitcodes"]), observation
            assert not ({p.pid for p in previous} & {p.pid for p in current}), observation

        def observe_format(self, stage, expected_closed):
            dataset = self.train_loader.dataset
            formats = [t for t in dataset.transforms.transforms if isinstance(t, Format) and t.return_mask]
            wanted = FastFormat if args.backend != "reference" else Format
            assert len(formats) == 1 and type(formats[0]) is wanted, "mask acceleration was lost during a rebuild"
            wanted_factory = wanted if args.persistent_mask else Format
            assert dataset.format_class is wanted_factory
            collator = self.train_loader.collate_fn
            if args.persistent_mask and args.backend != "reference":
                from ultrafast_maskops._shared_collate import shared_collate_fn

                assert collator is dataset.collate_fn is shared_collate_fn
            else:
                assert collator is dataset.collate_fn is YOLODataset.collate_fn
            rebuilt = id(dataset.transforms) != self.initial_transforms_id
            assert rebuilt == expected_closed, "expected real upstream transform rebuild was not observed"
            context = self.train_loader.multiprocessing_context
            self.lifecycle_records.append(
                {
                    "stage": stage,
                    "epoch": getattr(self, "epoch", None),
                    "start_epoch": self.start_epoch,
                    "formatter": type(formats[0]).__name__,
                    "factory": dataset.format_class.__name__,
                    "collator": f"{collator.__module__}.{collator.__qualname__}",
                    "rebuilt_from_initial": rebuilt,
                    "worker_pids": [p.pid for p in getattr(self.train_loader.iterator, "_workers", ())],
                    "worker_start_method": (
                        (context.get_start_method() if context else multiprocessing.get_start_method())
                        if self.train_loader.num_workers
                        else None
                    ),
                    "pin_memory": self.train_loader.pin_memory,
                    "prefetch_factor": self.train_loader.prefetch_factor,
                }
            )

        def measure_on_model_save(self):
            if self.epoch not in args.checkpoint_epochs:
                return
            # Preserve the original serialized checkpoint before final stripping
            # or a later epoch overwrites last.pt. Do not rewrite its arguments.
            destination = run_dir / f"resume-epoch-{self.epoch}.pt"
            data = self.last.read_bytes()
            with destination.open("xb") as stream:
                stream.write(data)
            digest = hashlib.sha256(data).hexdigest()
            assert file_sha(destination) == digest
            self.checkpoint_records.append(
                {
                    "epoch": self.epoch,
                    "path": str(destination),
                    "sha256": digest,
                    "bytes": len(data),
                }
            )

        def measure_on_train_epoch_start(self):
            assert self.batch_size == args.batch and not getattr(self, "_oom_retries", 0)
            self.seen_images = 0
            self.batch_losses = []
            if lifecycle:
                # Upstream closes mosaic AFTER this callback. Check at the first
                # batch below, after its unmodified close/reset calls have run.
                self.previous_workers = list(getattr(self.train_loader.iterator, "_workers", ()))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.epoch_started = time.perf_counter()

        def preprocess_batch(self, batch):
            if lifecycle and self.seen_images == 0:
                boundary = args.epochs - args.close_mosaic
                self.observe_format("first-batch", bool(args.close_mosaic) and self.epoch >= boundary)
                pinned_image = batch["img"].is_pinned()
                assert not self.train_loader.pin_memory or pinned_image, "image did not pass through pinning"
                self.lifecycle_records[-1]["pinned_image"] = pinned_image
                if args.close_mosaic and self.epoch == boundary:
                    self.observe_replacement(self.previous_workers)
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
        "close_mosaic": args.close_mosaic,
        "multi_scale": 0.0,
        "compile": False,
        "plots": False,
        "val": False,
        "save": bool(args.checkpoint_epochs),
        "project": str(run_dir),
        "name": "training",
        "exist_ok": False,
        "verbose": False,
    }
    if args.resume_from:
        overrides.update(resume=str(args.resume_from), save_dir=str(run_dir / "training"))
    report = {
        "complete": False,
        "scope": "full real-data GPU training trial; not accuracy or repeated speedup evidence",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "script_sha256": file_sha(__file__),
        "harness_sources": harness_sources,
        "upstream_files": source_hashes,
        "packages": packages,
        "mask_build": maskops.backend_info(),
        "fixture": COCO_FINGERPRINT,
        "corpus_root": str(root),
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
        "lifecycle_requested": lifecycle,
        "resume_input": None
        if not args.resume_from
        else {
            "path": str(args.resume_from),
            "sha256": resume_sha,
            "receipt_sha256": resume_receipt_sha,
            "expected_start_epoch": expected_start,
            "restored_state_expectation": resume_state,
        },
    }
    if lifecycle:
        report["scope"] = "full real-data GPU lifecycle trial; not accuracy or repeated speedup evidence"
        report["lifecycle_scope"] = (
            "Observe the parent dataset formatter/factory and worker reset at the first batch after upstream closure; "
            "no close/reset override or worker instrumentation. Compare resumed backends from the same retained "
            "reference checkpoint; do not assume upstream resume reproduces uninterrupted RNG/EMA state. "
            "Epoch timing includes closure/reset and the small observation callback; checkpoints are outside it."
        )
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
    trainer = None
    try:
        trainer = LocalSegmentationRun(overrides=overrides)
        trainer.dataset_records, trainer.epoch_records = [], []
        trainer.lifecycle_records, trainer.checkpoint_records = [], []
        trainer.train()
        whole_job_s = time.perf_counter() - started
        # The unmodified trainer closes both loaders before returning. Observe
        # that result instead of overriding close/reset or treating return as
        # proof of clean worker teardown.
        shutdown = {}
        for name, loader in (("train", trainer.train_loader), ("val", trainer.test_loader)):
            workers = list(getattr(loader.iterator, "_workers", ()))
            shutdown[name] = {
                "worker_count": loader.num_workers,
                "worker_pids": [p.pid for p in workers],
                "worker_exitcodes": [p.exitcode for p in workers],
                "workers_alive": [p.is_alive() for p in workers],
            }
        report["worker_shutdown"] = shutdown
        for observation in shutdown.values():
            assert len(observation["worker_pids"]) == observation["worker_count"]
            assert all(code == 0 for code in observation["worker_exitcodes"]), observation
            assert not any(observation["workers_alive"]), observation
        assert [r["epoch"] for r in trainer.epoch_records] == list(range(expected_start, args.epochs))
        assert sorted(r["epoch"] for r in trainer.checkpoint_records) == sorted(args.checkpoint_epochs)
        if lifecycle:
            assert len(trainer.lifecycle_records) == args.epochs - expected_start + 1
        for checkpoint in trainer.checkpoint_records:
            assert file_sha(checkpoint["path"]) == checkpoint["sha256"]
        if args.resume_from:
            assert file_sha(args.resume_from) == resume_sha, "input checkpoint changed"
            assert file_sha(args.resume_receipt) == resume_receipt_sha, "input checkpoint receipt changed"
        final_sha = state_sha(unwrap_model(trainer.model))
        assert final_sha != trainer.initial_state_sha, "model must actually update"
        assert file_sha(cache) == cache_sha, "original cache changed"
        assert harness_sources == {name: file_sha(Path(__file__).with_name(name)) for name in harness_sources}
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
        if trainer is not None:
            report.update(
                datasets=getattr(trainer, "dataset_records", []),
                epochs=getattr(trainer, "epoch_records", []),
                lifecycle=getattr(trainer, "lifecycle_records", []),
                checkpoints=getattr(trainer, "checkpoint_records", []),
            )
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
