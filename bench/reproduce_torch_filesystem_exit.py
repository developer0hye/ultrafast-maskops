"""Stock PyTorch full-epoch exit control; no Ultralytics, maskops or reducer hooks.

Uses the observer CLI contract, restricted to reference/full-epoch/one cycle.
There is no loader reset and no private worker shutdown call.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import torch


class Fixture(torch.utils.data.Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        return torch.full((3, 128, 128), index, dtype=torch.uint8)


def worker_init(_index):
    torch.multiprocessing.set_sharing_strategy("file_system")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("reference",), required=True)
    parser.add_argument("--reset-point", choices=("full-epoch",), required=True)
    parser.add_argument("--cycles", type=int, required=True)
    args = parser.parse_args()
    assert args.cycles == 1 and not args.out.exists()
    assert torch.__version__.split("+")[0] == "2.10.0"
    state = {
        "complete": False,
        "protocol_passed": False,
        "pid": os.getpid(),
        "python": sys.version,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_at_ns": time.time_ns(),
        "verified_batches": 0,
        "scope": "Stock PyTorch DataLoader, one full epoch, file_system, spawn, two workers, no resets.",
    }
    args.out.write_text(json.dumps(state, indent=2) + "\n")
    torch.multiprocessing.set_sharing_strategy("file_system")
    loader = torch.utils.data.DataLoader(
        Fixture(),
        batch_size=4,
        num_workers=2,
        multiprocessing_context="spawn",
        prefetch_factor=2,
        pin_memory=False,
        worker_init_fn=worker_init,
    )
    iterator = iter(loader)
    workers = tuple(iterator._workers)  # Observe only; full exhaustion does the shutdown.
    for batch_index, batch in enumerate(iterator):
        expected = torch.stack([Fixture()[i] for i in range(batch_index * 4, (batch_index + 1) * 4)])
        assert type(batch) is torch.Tensor and batch.dtype == expected.dtype
        assert batch.shape == expected.shape and batch.stride() == expected.stride()
        assert torch.equal(batch, expected)
        state["verified_batches"] += 1
        del batch, expected
    state["workers"] = [{"pid": w.pid, "exitcode": w.exitcode, "alive": w.is_alive()} for w in workers]
    assert state["verified_batches"] == 4
    assert all(w.exitcode == 0 and not w.is_alive() for w in workers)
    del iterator, loader, workers
    gc.collect()
    assert not any(
        name.startswith(("ultralytics", "ultrafast_maskops", "ultrafast_yolo_dataset")) for name in sys.modules
    )
    state.update(complete=True, protocol_passed=True, finished_at_ns=time.time_ns(), no_project_imports=True)
    args.out.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
