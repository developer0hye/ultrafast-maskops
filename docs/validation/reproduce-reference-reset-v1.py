"""Diagnose reference-only worker shutdown without importing either new library."""

import argparse
import copy
import hashlib
import json
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.segment import SegmentationTrainer


def assert_reference_only():
    assert not any(n.startswith(("ultrafast_maskops", "ultrafast_yolo_dataset")) for n in sys.modules)


assert_reference_only()  # Also runs in each spawned worker.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-reset", choices=("first-batch", "full-epoch", "setup"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    record = {
        "complete": False,
        "before_reset": args.before_reset,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "events": [],
    }
    loader = None
    with tempfile.TemporaryDirectory(prefix="reference-reset-") as temporary:
        root = Path(temporary)
        images, labels = root / "images", root / "labels"
        images.mkdir()
        labels.mkdir()
        rng = np.random.default_rng(1943)
        for index in range(4):
            Image.fromarray(rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)).save(images / f"{index}.png")
            (labels / f"{index}.txt").write_text("0 .1 .1 .8 .1 .8 .8 .1 .8\n1 .3 .3 .5 .3 .4 .5\n")
        hyp = copy.deepcopy(DEFAULT_CFG)
        hyp.overlap_mask, hyp.mask_ratio, hyp.augmentations = True, 4, []
        for name in ("mosaic", "copy_paste", "mixup", "cutmix"):
            setattr(hyp, name, 1.0)
        for name in (
            "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
            "flipud", "fliplr", "bgr",
        ):
            setattr(hyp, name, 0.0)
        try:
            dataset = YOLODataset(
                img_path=str(images), imgsz=64, batch_size=2, augment=True, hyp=hyp,
                task="segment", data={"names": {0: "a", 1: "b"}, "nc": 2},
            )
            loader = InfiniteDataLoader(
                dataset, batch_size=2, num_workers=2, collate_fn=YOLODataset.collate_fn,
                multiprocessing_context="spawn",
            )
            record["events"].append("constructed")
            if args.before_reset == "first-batch":
                next(iter(loader))
            elif args.before_reset == "full-epoch":
                assert len(list(loader)) == 2
            record["events"].append("consumed-before-reset")
            previous = list(loader.iterator._workers)
            record["old_worker_pids"] = [w.pid for w in previous]
            SegmentationTrainer._close_dataloader_mosaic(SimpleNamespace(args=hyp, train_loader=loader))
            record["events"].append("closed-mosaic")
            loader.reset()
            record["events"].append("reset")
            record["old_worker_exitcodes"] = [w.exitcode for w in previous]
            assert all(not w.is_alive() and w.exitcode == 0 for w in previous)
            assert len(list(loader)) == 2
            record["events"].append("consumed-after-reset")
            assert_reference_only()
            record["complete"] = True
        except Exception:
            record["error"] = traceback.format_exc()
        finally:
            if loader is not None:
                try:
                    loader.close()
                except Exception:
                    record["cleanup_error"] = traceback.format_exc()
                    record["complete"] = False
            record["imported_new_libraries"] = [
                n for n in sys.modules if n.startswith(("ultrafast_maskops", "ultrafast_yolo_dataset"))
            ]
            args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0 if record["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
