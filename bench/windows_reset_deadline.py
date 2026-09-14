"""Mechanism control for reset during slow worker initialization on Windows.

Six predeclared cases: three collators x ordinary/delayed worker initialization.
The delayed cases deliberately exceed both unchanged per-worker join deadlines.
A mechanism-control pass is NOT a clean-exit qualification or a diagnosis of the
original intermittent failure. Stock workers import no ultrafast-maskops module.
"""

import copy
import functools
import inspect
import json
import os
import pickle
import sys
import time
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.segment import SegmentationTrainer


class ReferencePacket:
    __slots__ = ("payload",)

    def __init__(self, batch):
        self.payload = bytes(ForkingPickler.dumps(batch))

    def __reduce__(self):
        return pickle.loads, (self.payload,)


def reference_transport_collate(batch):
    result = YOLODataset.collate_fn(batch)
    for value in result.values():
        if isinstance(value, torch.Tensor):
            value.share_memory_()
    return ReferencePacket(result)


def initialize_worker(worker_id, *, delay, directory):
    record = {
        "worker_id": worker_id,
        "pid": os.getpid(),
        "delay_seconds": delay,
        "maskops_imported": any(n.startswith("ultrafast_maskops") for n in sys.modules),
        "started_at": time.time(),
    }
    with (Path(directory) / f"worker-{os.getpid()}.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    time.sleep(delay)


def state(workers):
    return [{"pid": p.pid, "alive": p.is_alive(), "exitcode": p.exitcode} for p in workers]


def instrument(workers, events):
    for process in workers:
        for name in ("join", "terminate"):
            method = getattr(process, name)

            def wrapped(*args, _method=method, _name=name, _process=process, **kwargs):
                event = {"method": _name, "pid": _process.pid, "args": args, "kwargs": kwargs}
                events.append(event)
                start = time.perf_counter()
                try:
                    return _method(*args, **kwargs)
                finally:
                    event.update(seconds=time.perf_counter() - start, state=state([_process])[0])

            setattr(process, name, wrapped)


def make_dataset(folder):
    images, labels = folder / "images", folder / "labels"
    images.mkdir()
    labels.mkdir()
    rng = np.random.default_rng(1943)
    for index in range(4):
        Image.fromarray(rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)).save(images / f"{index}.png")
        (labels / f"{index}.txt").write_text("0 .1 .1 .8 .1 .8 .8 .1 .8\n1 .3 .3 .5 .3 .4 .5\n", encoding="utf-8")
    hyp = copy.deepcopy(DEFAULT_CFG)
    hyp.overlap_mask, hyp.mask_ratio, hyp.augmentations = True, 4, []
    for key in ("mosaic", "copy_paste", "mixup", "cutmix"):
        setattr(hyp, key, 1.0)
    for key in (
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "bgr",
    ):
        setattr(hyp, key, 0.0)
    dataset = YOLODataset(
        img_path=str(images),
        imgsz=64,
        batch_size=2,
        augment=True,
        hyp=hyp,
        task="segment",
        data={"names": {0: "a", 1: "b"}, "nc": 2},
    )
    return dataset, hyp


@pytest.mark.parametrize("delayed", [False, True])
@pytest.mark.parametrize("backend", ["stock", "reference-packet", "native-packet"])
def test_deadline_control(tmp_path, backend, delayed):
    output = Path(os.environ["RESET_DEADLINE_OUT"]) / f"{backend}-{int(delayed)}"
    output.mkdir()
    interval = torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL
    delay = 3 * interval if delayed else 0
    record = {"backend": backend, "delayed": delayed, "interval": interval, "worker_delay": delay}
    loader = None
    try:
        dataset, hyp = make_dataset(tmp_path)
        if backend == "native-packet":
            from ultrafast_maskops.ultralytics import accelerate_dataset

            assert accelerate_dataset(dataset, persistent=True) == 1
        collate = reference_transport_collate if backend == "reference-packet" else dataset.collate_fn
        loader = InfiniteDataLoader(
            dataset,
            batch_size=2,
            num_workers=2,
            collate_fn=collate,
            multiprocessing_context="spawn",
            worker_init_fn=functools.partial(initialize_worker, delay=delay, directory=str(output)),
        )
        previous = list(loader.iterator._workers)
        events = record["events"] = []
        instrument(previous, events)
        record["before_reset"] = state(previous)
        # Only the *replacement* generation uses the ordinary initializer.
        # Existing worker copies retain the original (possibly slow) initializer.
        loader.worker_init_fn = functools.partial(initialize_worker, delay=0, directory=str(output))
        SegmentationTrainer._close_dataloader_mosaic(SimpleNamespace(args=hyp, train_loader=loader))
        loader.reset()
        record["after_reset_old"] = state(previous)
        restarted = list(loader.iterator._workers)
        record["after_reset_new"] = state(restarted)
        record["clean_zero_exit"] = all(not p.is_alive() and p.exitcode == 0 for p in previous)
        terminated = [e["pid"] for e in events if e["method"] == "terminate"]
        assert all(not p.is_alive() for p in previous), record
        assert not ({p.pid for p in previous} & {p.pid for p in restarted})
        joins = [e for e in events if e["method"] == "join"]
        assert len(joins) == 2 and all(e["kwargs"] == {"timeout": interval} for e in joins)
        if delayed:
            assert terminated and not record["clean_zero_exit"], record
            assert all(p.exitcode != 0 for p in previous if p.pid in terminated), record
        else:
            assert not terminated and record["clean_zero_exit"], record
        assert len(list(loader)) == 2
        loader.close()
        record["after_close"] = state(restarted)
        assert all(not p.is_alive() and p.exitcode == 0 for p in restarted), record
        child_records = [json.loads(p.read_text()) for p in output.glob("worker-*.json")]
        assert len(child_records) == 4, child_records
        assert all(r["maskops_imported"] == (backend == "native-packet") for r in child_records)
        record["worker_initializers"] = child_records
        record["mechanism_confirmed"] = True
    except BaseException as error:
        record.update(mechanism_confirmed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            if loader is not None:
                loader.close()
        finally:
            with (output / "result.json").open("x", encoding="utf-8") as stream:
                json.dump(record, stream, indent=2)
                stream.write("\n")


if __name__ == "__main__":
    output = Path("reset-evidence/deadline").resolve()
    output.mkdir(parents=True)
    os.environ["RESET_DEADLINE_OUT"] = str(output)
    sources = {
        "InfiniteDataLoader-reset.py": InfiniteDataLoader.reset,
        "InfiniteDataLoader-close.py": InfiniteDataLoader.close,
        "Torch-shutdown-workers.py": torch.utils.data.dataloader._MultiProcessingDataLoaderIter._shutdown_workers,
    }
    for name, obj in sources.items():
        (output / name).write_text(inspect.getsource(obj), encoding="utf-8")
    (output / "runtime.json").write_text(json.dumps({"python": sys.version, "torch": torch.__version__}, indent=2))
    raise SystemExit(pytest.main([__file__, "-q", "-s", f"--junitxml={output / 'tests.xml'}"]))
