"""Observe named shared-memory lifetime across real InfiniteDataLoader resets.

Diagnostic only: synthetic tensors, instrumented reducers, explicit file_system
transport. Run each backend/reset point in a fresh process on an idle host.
Never unlinks shared memory, changes shutdown methods, or normalizes reference
transport. Surviving handles are observations, not proof of a native regression.
"""

import argparse
import ctypes
import errno
import gc
import hashlib
import inspect
import json
import os
import platform
import sys
import time
import traceback
from multiprocessing.reduction import ForkingPickler
from pathlib import Path

import torch
from torch.multiprocessing import reductions
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.data.dataset import YOLODataset

from ultrafast_maskops import _shared_collate as shared


def sample(index):
    count = index % 3 + 1
    return {
        "img": torch.full((3, 128, 128), index, dtype=torch.uint8),
        "masks": torch.full((count, 32, 32), index, dtype=torch.uint8),
        "bboxes": torch.full((count, 4), float(index)),
        "cls": torch.full((count, 1), float(index % 2)),
        "batch_idx": torch.zeros(count),
        "im_file": str(index),
    }


class Fixture(torch.utils.data.Dataset):
    def __init__(self, trace_dir):
        self.trace_dir = str(trace_dir)

    def __len__(self):
        return 16

    def __getitem__(self, index):
        return sample(index)


def traced_reduce_storage(storage):
    result = reductions.reduce_storage(storage)
    if result[0] is reductions.rebuild_storage_filename:
        _, manager, handle, size, *dtype = result[1]
        assert not dtype, "registered for untyped CPU storage only"
        trace_dir = Path(torch.utils.data.get_worker_info().dataset.trace_dir)
        payload = (json.dumps({
            "pid": os.getpid(), "manager": os.fsdecode(manager),
            "handle": os.fsdecode(handle), "storage_bytes": size,
            "at_ns": time.time_ns(),
        }) + "\n").encode()
        descriptor = os.open(trace_dir / f"{os.getpid()}.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            assert os.write(descriptor, payload) == len(payload)
        finally:
            os.close(descriptor)
    return result


def initialize_worker(_worker_id):
    torch.multiprocessing.set_sharing_strategy("file_system")
    ForkingPickler.register(torch.UntypedStorage, traced_reduce_storage)


def require_equal(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.stride() == expected.stride() and torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key in expected:
            require_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            require_equal(a, b)
    else:
        assert actual == expected


def probe_handles(trace_dir, worker_pids):
    """Open only observed names read-only; never enumerate or unlink host SHM."""
    library = ctypes.CDLL(None, use_errno=True)
    open_shm = library.shm_open
    open_shm.argtypes = (ctypes.c_char_p, ctypes.c_int, ctypes.c_uint)
    open_shm.restype = ctypes.c_int
    names = {}
    trace_hashes = {}
    for pid in worker_pids:
        path = trace_dir / f"{pid}.jsonl"
        if not path.exists():
            continue  # setup-only reset may end before any batch is produced
        raw = path.read_bytes()
        trace_hashes[path.name] = hashlib.sha256(raw).hexdigest()
        for line in raw.splitlines():
            record = json.loads(line)
            assert record["pid"] == pid
            handle = record["handle"]
            assert handle.startswith(f"/torch_{pid}_") and handle.count("/") == 1
            assert type(record["storage_bytes"]) is int and record["storage_bytes"] >= 0
            previous = names.get(handle)
            assert previous is None or previous["storage_bytes"] == record["storage_bytes"]
            names[handle] = record
    present = []
    for handle, record in names.items():
        descriptor = open_shm(os.fsencode(handle), os.O_RDONLY, 0)
        if descriptor >= 0:
            os.close(descriptor)
            present.append({"handle": handle, "storage_bytes": record["storage_bytes"]})
        elif ctypes.get_errno() != errno.ENOENT:
            raise OSError(ctypes.get_errno(), f"shm_open failed for observed handle {handle}")
    return {
        "worker_pids": worker_pids, "observed_handles": len(names),
        "trace_sha256": trace_hashes, "present": present,
        "present_storage_bytes": sum(record["storage_bytes"] for record in present),
        "scope": "Named storage payload sizes, not physical RSS or allocator overhead; parent still alive.",
    }


def workers_state(workers):
    return [{"pid": worker.pid, "alive": worker.is_alive(), "exitcode": worker.exitcode} for worker in workers]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("reference", "native"), required=True)
    parser.add_argument("--reset-point", choices=("setup", "first-batch", "full-epoch"), required=True)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert args.cycles > 0 and not args.out.exists()
    traces = args.out.with_suffix(".traces")
    traces.mkdir()  # no reuse of an earlier attempt
    state = {
        "complete": False, "protocol_passed": False, "args": vars(args) | {"out": str(args.out)},
        "pid": os.getpid(), "python": sys.version, "platform": platform.platform(),
        "torch_version": torch.__version__, "sharing_strategy": "file_system",
        "context": "spawn", "workers": 2, "prefetch_factor": 2, "batch_size": 4,
        "instrumented_reducers": True, "performance_measurement": False,
        "cycles": [], "started_at_ns": time.time_ns(),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in map(Path, (
                __file__, shared.__file__, reductions.__file__,
                inspect.getfile(InfiniteDataLoader), inspect.getfile(YOLODataset),
            ))
        },
    }
    loader = None
    retired_pids = []

    def save():
        temporary = args.out.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        os.replace(temporary, args.out)

    try:
        shared._check_reference_profile()
        assert "file_system" in torch.multiprocessing.get_all_sharing_strategies()
        torch.multiprocessing.set_sharing_strategy("file_system")
        save()
        loader = InfiniteDataLoader(
            Fixture(traces), batch_size=4, shuffle=False, num_workers=2,
            multiprocessing_context="spawn", prefetch_factor=2, pin_memory=False,
            worker_init_fn=initialize_worker,
            collate_fn=YOLODataset.collate_fn if args.backend == "reference" else shared.shared_collate_fn,
        )
        for index in range(args.cycles):
            record = {"index": index, "verified_batches": 0}
            state["cycles"].append(record)
            save()
            if args.reset_point == "setup":
                # Exercise actual unconsumed prefetch instead of shutting down
                # while workers are still importing, before any storage exists.
                started = time.monotonic()
                workers = tuple(loader.iterator._workers)
                while True:
                    paths = [traces / f"{worker.pid}.jsonl" for worker in workers]
                    if all(path.exists() and path.stat().st_size > 0 for path in paths):
                        break
                    assert all(worker.is_alive() for worker in workers), "worker exited before prefetch"
                    assert time.monotonic() - started < 30, "prefetch trace timeout"
                    time.sleep(0.05)
                record["setup_prefetch_observed_wait_s"] = time.monotonic() - started
                save()
            if args.reset_point != "setup":
                for batch_index, batch in enumerate(loader):
                    expected = YOLODataset.collate_fn([sample(i) for i in range(batch_index * 4, (batch_index + 1) * 4)])
                    require_equal(batch, expected)
                    record["verified_batches"] += 1
                    del batch, expected
                    if args.reset_point == "first-batch":
                        break
            assert record["verified_batches"] == {"setup": 0, "first-batch": 1, "full-epoch": 4}[args.reset_point]
            # Retain Process handles, not the old iterator or any Tensor batch.
            old_workers = tuple(loader.iterator._workers)
            old_pids = [w.pid for w in old_workers]
            retired_pids.extend(old_pids)
            record["old_worker_pids"] = old_pids
            save()
            try:
                loader.reset()
            finally:
                record["old_workers"] = workers_state(old_workers)
                save()
            record["new_worker_pids"] = [worker.pid for worker in loader.iterator._workers]
            save()
            assert len(old_workers) == 2 and all(not w.is_alive() and w.exitcode == 0 for w in old_workers)
            assert not set(old_pids) & set(record["new_worker_pids"])
            gc.collect()
            record["retired_handles_after_reset"] = probe_handles(traces, retired_pids)
            save()
        state["protocol_passed"] = True
    except BaseException:
        state["error"] = traceback.format_exc()
        raise
    finally:
        try:
            if loader is not None:
                final_workers = tuple(loader.iterator._workers)
                loader.close()
                state["final_workers"] = workers_state(final_workers)
                retired_pids.extend(w.pid for w in final_workers if w.pid not in retired_pids)
                del loader
                gc.collect()
                state["handles_after_close"] = probe_handles(traces, retired_pids)
                # The manager may reclaim orphaned names asynchronously. Keep
                # both observations; never relabel a delayed cleanup as immediate.
                time.sleep(3)
                state["handles_after_close_plus_3s"] = probe_handles(traces, retired_pids)
                assert all(not w.is_alive() and w.exitcode == 0 for w in final_workers)
        except BaseException:
            state["protocol_passed"] = False
            state["cleanup_error"] = traceback.format_exc()
            raise
        finally:
            state.update(complete=True, finished_at_ns=time.time_ns())
            save()
    print(json.dumps({"complete": state["complete"], "protocol_passed": state["protocol_passed"],
                      "surviving_handles": len(state["handles_after_close_plus_3s"]["present"])}))


if __name__ == "__main__":
    main()
