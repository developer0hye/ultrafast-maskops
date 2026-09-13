"""Predeclared Windows reset diagnosis: 3 collators x 3 reset points x 3 repetitions.

Uses the original failed wheel. No reset/join deadlines or sharing policy changes.
Join/terminate wrappers record parent-side events and add diagnostic overhead.
All 27 cases run even after failures; none substitutes for the failed full matrix.
"""

import copy
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import pytest
from test_persistent_format import (
    FastFormat,
    Format,
    InfiniteDataLoader,
    SegmentationTrainer,
    YOLODataset,
    accelerate_dataset,
    corpus,  # noqa: F401 -- pytest fixture
    formatter,
    options,
    reference_transport_collate,
)


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


@pytest.mark.parametrize("repetition", range(3))
@pytest.mark.parametrize("phase", ["setup", "first-batch", "full-epoch"])
@pytest.mark.parametrize("backend", ["stock", "reference-packet", "native-packet"])
def test_reset(corpus, backend, phase, repetition):  # noqa: F811 -- injected imported pytest fixture
    out = Path(os.environ["RESET_PROBE_OUT"])
    record = {"backend": backend, "phase": phase, "repetition": repetition, "platform": platform.platform()}
    path = out / f"{backend}-{phase}-{repetition}.json"
    loader = None
    try:
        kwargs = options(corpus, True, mosaic=1.0)
        dataset = YOLODataset(**copy.deepcopy(kwargs))
        if backend == "native-packet":
            assert accelerate_dataset(dataset, persistent=True) == 1
        collate = reference_transport_collate if backend == "reference-packet" else dataset.collate_fn
        loader = InfiniteDataLoader(
            dataset, batch_size=2, num_workers=2, collate_fn=collate, multiprocessing_context="spawn"
        )
        owner = SimpleNamespace(args=kwargs["hyp"], train_loader=loader)
        if phase == "first-batch":
            next(iter(loader))
        elif phase == "full-epoch":
            assert len(list(loader)) == 2
        previous = list(loader.iterator._workers)
        events = record["events"] = []
        instrument(previous, events)
        record["before_reset"] = state(previous)
        previous_transforms = dataset.transforms
        SegmentationTrainer._close_dataloader_mosaic(owner)
        assert dataset.transforms is not previous_transforms
        assert type(formatter(dataset)) is (FastFormat if backend == "native-packet" else Format)
        start = time.perf_counter()
        loader.reset()
        record["reset_seconds"] = time.perf_counter() - start
        record["after_reset_old"] = state(previous)
        restarted = list(loader.iterator._workers)
        record["after_reset_new"] = state(restarted)
        assert len(restarted) == 2
        assert all(not p.is_alive() and p.exitcode == 0 for p in previous), record
        assert not ({p.pid for p in previous} & {p.pid for p in restarted})
        assert len(list(loader)) == 2
        instrument(restarted, events)
        loader.close()
        record["after_close"] = state(restarted)
        assert all(not p.is_alive() and p.exitcode == 0 for p in restarted), record
        record["passed"] = True
    except BaseException as error:
        record.update(passed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            if loader is not None:
                loader.close()
        finally:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(record, stream, indent=2)
                stream.write("\n")


if __name__ == "__main__":
    output = Path("reset-evidence/probe").resolve()
    output.mkdir(parents=True)
    os.environ["RESET_PROBE_OUT"] = str(output)
    raise SystemExit(pytest.main([__file__, "-q", "-s", f"--junitxml={output / 'tests.xml'}"]))
