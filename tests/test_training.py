"""Supporting deterministic CPU training checks, not GPU throughput evidence."""

import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import ultralytics
from ultrafast_maskops.ultralytics import FastFormat
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data.augment import Format
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import YAML
from ultralytics.utils.instance import Instances


def training_batch(formatter, count):
    rng = np.random.default_rng(195)
    values = []
    for _ in range(2):
        # Closed rectangles with different overlaps and nonempty interiors make
        # the foreground loss exercise real mask values rather than empty masks.
        segments = np.array(
            [[[0.1 + i * 0.03, 0.1], [0.8, 0.1], [0.8, 0.8], [0.1 + i * 0.03, 0.8]] for i in range(count)],
            dtype=np.float32,
        ).reshape(count, 4, 2)
        boxes = np.array(
            [[0.45 + i * 0.015, 0.45, 0.7 - i * 0.03, 0.7] for i in range(count)], dtype=np.float32
        ).reshape(count, 4)
        sample = {
            "img": rng.integers(0, 256, (128, 128, 3), dtype=np.uint8),
            "cls": (np.arange(count, dtype=np.float32) % 3).reshape(-1, 1),
            "instances": Instances(
                boxes,
                segments,
                bbox_format="xywh",
                normalized=True,
            ),
        }
        values.append(formatter(sample))
    batch = YOLODataset.collate_fn(values)
    batch["img"] = batch["img"].float() / 255
    return batch


@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize("count", [0, 5])
def test_yolo11n_cpu_losses_gradients_and_sgd_updates(overlap, count):
    config = YAML.load(Path(ultralytics.__file__).parent / "cfg/models/11/yolo11-seg.yaml")
    config["scale"] = "n"
    previous_threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    random_state = random.getstate()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        expected = None
        with torch.random.fork_rng(devices=[]):
            for formatter_type in (Format, FastFormat):
                torch.manual_seed(912)
                random.seed(912)
                model = SegmentationModel(copy.deepcopy(config), nc=3, verbose=False).cpu().train()
                model.args = copy.deepcopy(DEFAULT_CFG)
                model.args.overlap_mask = overlap
                formatter = formatter_type(return_mask=True, mask_overlap=overlap, mask_ratio=4, bgr=1.0)
                batch = training_batch(formatter, count)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
                steps = []
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    loss, components = model.loss(batch)
                    assert torch.isfinite(loss).all()
                    if count:
                        assert components["seg_loss"] > 0, "foreground case must exercise mask loss"
                    loss.sum().backward()
                    gradients = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
                    assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
                    optimizer.step()
                    steps.append({"loss": loss.detach().clone(), "components": components, "gradients": gradients})
                final = {n: p.detach().clone() for n, p in model.state_dict().items()}
                if expected is None:
                    expected = (steps, final)
                else:
                    for original, candidate in zip(expected[0], steps):
                        assert torch.equal(original["loss"], candidate["loss"])
                        for group in ("components", "gradients"):
                            assert original[group].keys() == candidate[group].keys()
                            for name in original[group]:
                                assert torch.equal(original[group][name], candidate[group][name]), (group, name)
                    assert expected[1].keys() == final.keys()
                    for name, value in final.items():
                        assert torch.equal(expected[1][name], value), name
    finally:
        random.setstate(random_state)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(previous_threads)
