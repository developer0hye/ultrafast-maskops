"""FastSegmentationTrainer feeds the model the same batches as the unmodified trainer.

Each training runs in its own process: a second training in the same process
does not reproduce the first (Ultralytics keeps state between runs), whereas
fresh processes with the same seed do. The comparison is made where this
package's guarantee lies: on every batch the trainer receives, in both epochs,
across the close_mosaic rebuild.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("MASKOPS_SKIP_TRAINING") == "1", reason="training test disabled")


def digest(batch):
    h = hashlib.sha256()
    for key in sorted(batch):
        value = batch[key]
        h.update(key.encode() + b"\0")
        if isinstance(value, torch.Tensor):
            h.update(f"{value.dtype}:{tuple(value.shape)}".encode() + value.contiguous().numpy().tobytes())
        elif isinstance(value, np.ndarray):
            h.update(f"{value.dtype}:{value.shape}".encode() + np.ascontiguousarray(value).tobytes())
        else:
            h.update(repr(value).encode())
    return h.hexdigest()


def train(kind, root):
    """Train for two epochs; return the batch digests and the final transform classes."""
    from ultralytics import YOLO
    from ultralytics.models.yolo.segment import SegmentationTrainer
    from ultralytics.utils import YAML

    from test_geometry import perspectives

    if kind == "fast":
        from ultrafast_maskops.training import FastSegmentationTrainer as base
    else:
        base = SegmentationTrainer
    digests = []

    class Recording(base):
        def preprocess_batch(self, batch):
            digests.append(digest(batch))  # before the trainer's own preprocessing
            return super().preprocess_batch(batch)

    data = root / "data.yaml"
    YAML.save(data, {"path": str(root), "train": "images", "val": "images", "names": {0: "a", 1: "b", 2: "c"}})
    torch.set_num_threads(1)
    model = YOLO("yolo11n-seg.yaml")
    # Two epochs with close_mosaic=1 rebuild the transforms between them, the
    # point where a non-persistent acceleration would be lost.
    model.train(
        trainer=Recording,
        data=str(data),
        epochs=2,
        close_mosaic=1,
        imgsz=128,
        batch=4,
        workers=0,
        device="cpu",
        seed=3,
        deterministic=True,
        amp=False,
        plots=False,
        val=False,
        pretrained=False,
        exist_ok=True,
        project=str(root / "runs"),
        name=kind,
        verbose=False,
    )
    compose = model.trainer.train_loader.dataset.transforms
    classes = sorted({type(t).__name__ for t in compose.transforms} | {type(t).__name__ for t in perspectives(compose)})
    return {"digests": digests, "transforms": classes}


def run(kind, root):
    output = root / f"{kind}.json"
    env = {**os.environ, "YOLO_OFFLINE": "true"}
    subprocess.run([sys.executable, __file__, kind, str(root), str(output)], check=True, env=env)
    return json.loads(output.read_text())


def test_fast_trainer_receives_identical_batches(tmp_path):
    from test_geometry import corpus

    corpus(tmp_path)
    expected = run("reference", tmp_path)
    actual = run("fast", tmp_path)
    assert len(expected["digests"]) == 2 * 6  # 24 images, batch 4, two epochs
    assert actual["digests"] == expected["digests"]
    # The transforms were rebuilt by close_mosaic; the acceleration persisted.
    assert "FastFormat" in actual["transforms"] and "Format" not in actual["transforms"]
    assert "FastRandomPerspective" in actual["transforms"] and "RandomPerspective" not in actual["transforms"]


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    Path(sys.argv[3]).write_text(json.dumps(train(sys.argv[1], Path(sys.argv[2]))))
