import copy
import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from ultrafast_maskops.ultralytics import FastFormat
from ultralytics.data.augment import Format
from ultralytics.utils.instance import Instances


def inputs(n=5):
    rng = np.random.default_rng(143)
    segments = rng.uniform(0.05, 0.95, (n, 16, 2)).astype(np.float32)
    boxes = np.tile([0.5, 0.5, 0.8, 0.8], (n, 1)).astype(np.float32)
    return {
        "img": rng.integers(0, 256, (64, 80, 3), dtype=np.uint8),
        "cls": np.arange(n, dtype=np.float32).reshape(-1, 1) % 3,
        "instances": Instances(boxes, segments, bbox_format="xywh", normalized=True),
    }


def equal_dict(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert type(a[key]) is type(b[key]), key
        if isinstance(a[key], torch.Tensor):
            assert a[key].dtype == b[key].dtype, key
            assert torch.equal(a[key], b[key]), key
        else:
            assert a[key] == b[key], key


@pytest.mark.parametrize("n", [0, 1, 5, 129, 256])
@pytest.mark.parametrize("overlap", [True, False])
def test_full_format(n, overlap):
    original = Format(return_mask=True, mask_overlap=overlap, mask_ratio=4, bgr=1.0)
    replacement = FastFormat.from_reference(original)
    data = inputs(n)
    equal_dict(original(copy.deepcopy(data)), replacement(copy.deepcopy(data)))
    # A used formatter must also survive DataLoader spawn serialization.
    restored = pickle.loads(pickle.dumps(replacement))
    equal_dict(original(copy.deepcopy(data)), restored(copy.deepcopy(data)))


class FormatDataset(Dataset):
    def __init__(self, native):
        self.formatter = (FastFormat if native else Format)(return_mask=True, bgr=1.0)

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return self.formatter(inputs(5))


@pytest.mark.parametrize("workers", [0, 2])
def test_dataloader_spawn(workers):
    kwargs = {"multiprocessing_context": "spawn"} if workers else {}
    original = DataLoader(FormatDataset(False), batch_size=2, num_workers=workers, **kwargs)
    replacement = DataLoader(FormatDataset(True), batch_size=2, num_workers=workers, **kwargs)
    for a, b in zip(original, replacement):
        equal_dict(a, b)


def test_unknown_profile_rejected(monkeypatch):
    import ultrafast_maskops.ultralytics as integration

    monkeypatch.setitem(integration._SOURCE_HASHES, "Format", "unrecognized")
    with pytest.raises(RuntimeError, match="unsupported Ultralytics"):
        FastFormat()


@pytest.mark.parametrize("workers", [0, 2])
def test_actual_yolo_dataset_first_batches(tmp_path, workers):
    from PIL import Image
    from ultrafast_maskops.ultralytics import accelerate_dataset
    from ultralytics.cfg import DEFAULT_CFG
    from ultralytics.data.dataset import YOLODataset

    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for i in range(8):
        Image.fromarray(inputs()["img"]).save(images / f"{i}.png")
        (labels / f"{i}.txt").write_text("0 .1 .1 .8 .1 .8 .8 .1 .8\n1 .3 .3 .5 .3 .4 .5\n")
    kwargs = {
        "img_path": str(images),
        "imgsz": 64,
        "batch_size": 2,
        "augment": False,
        "hyp": copy.deepcopy(DEFAULT_CFG),
        "data": {"names": {0: "a", 1: "b"}, "nc": 2},
        "task": "segment",
    }
    original = YOLODataset(**kwargs)
    replacement = YOLODataset(**kwargs)
    assert accelerate_dataset(replacement) == 1
    loader_kwargs = {"batch_size": 2, "num_workers": workers, "collate_fn": YOLODataset.collate_fn}
    if workers:
        loader_kwargs["multiprocessing_context"] = "spawn"
    for a, b in zip(DataLoader(original, **loader_kwargs), DataLoader(replacement, **loader_kwargs)):
        equal_dict(a, b)
