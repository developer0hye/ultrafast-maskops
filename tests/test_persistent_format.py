"""Exercise the explicit instance factory across real rebuilds and worker resets."""

import copy
import pickle
from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from ultrafast_maskops.ultralytics import FastFormat, accelerate_dataset
from ultrafast_maskops._shared_collate import shared_collate_fn
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data.augment import Format
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.segment import SegmentationTrainer


@pytest.fixture
def corpus(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    rng = np.random.default_rng(1943)
    for index in range(4):
        Image.fromarray(rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)).save(images / f"{index}.png")
        (labels / f"{index}.txt").write_text("0 .1 .1 .8 .1 .8 .8 .1 .8\n1 .3 .3 .5 .3 .4 .5\n")
    return images


def options(images, overlap=True, mosaic=0.0):
    hyp = copy.deepcopy(DEFAULT_CFG)
    hyp.overlap_mask = overlap
    hyp.mask_ratio = 4
    hyp.augmentations = []
    for key in ("mosaic", "copy_paste", "mixup", "cutmix"):
        setattr(hyp, key, mosaic)
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
    return {
        "img_path": str(images),
        "imgsz": 64,
        "batch_size": 2,
        "augment": True,
        "hyp": hyp,
        "task": "segment",
        "data": {"names": {0: "a", 1: "b"}, "nc": 2},
    }


def formatter(dataset):
    values = [t for t in dataset.transforms.transforms if isinstance(t, Format)]
    assert len(values) == 1
    return values[0]


def equal_batch(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert type(left[key]) is type(right[key]), key
        if isinstance(left[key], torch.Tensor):
            assert left[key].dtype == right[key].dtype, key
            assert torch.equal(left[key], right[key]), key
        else:
            assert left[key] == right[key], key


class ReferencePacket:
    __slots__ = ("payload",)

    def __init__(self, batch):
        self.payload = bytes(ForkingPickler.dumps(batch))

    def __reduce__(self):
        return pickle.loads, (self.payload,)


def reference_transport_collate(batch):
    # The original reference spawn teardown failure is preserved separately.
    # Keep its collation computation independent; prepare only IPC storage here.
    result = YOLODataset.collate_fn(batch)
    if torch.utils.data.get_worker_info() is None:
        return result
    for value in result.values():
        if isinstance(value, torch.Tensor):
            value.share_memory_()
    return ReferencePacket(result)


@pytest.mark.parametrize("overlap", [True, False])
def test_persistent_factory_survives_rebuild_pickle_and_repeated_opt_in(corpus, overlap):
    kwargs = options(corpus, overlap)
    original, candidate = YOLODataset(**copy.deepcopy(kwargs)), YOLODataset(**copy.deepcopy(kwargs))
    assert accelerate_dataset(candidate, persistent=True) == 1
    assert candidate.format_class is FastFormat
    assert candidate.collate_fn is shared_collate_fn
    assert original.format_class is YOLODataset.format_class is Format
    assert original.collate_fn is YOLODataset.collate_fn
    assert accelerate_dataset(candidate, persistent=True) == 0
    # Use the formatter before pickling, so the native engine must be discarded.
    expected = YOLODataset.collate_fn([original[i] for i in range(2)])
    equal_batch(expected, YOLODataset.collate_fn([candidate[i] for i in range(2)]))
    restored = pickle.loads(pickle.dumps(candidate))
    assert restored.format_class is FastFormat
    assert restored.collate_fn is shared_collate_fn
    restored.transforms = restored.build_transforms(copy.deepcopy(kwargs["hyp"]))
    assert type(formatter(restored)) is FastFormat
    equal_batch(expected, YOLODataset.collate_fn([restored[i] for i in range(2)]))


def test_default_remains_one_shot_and_can_be_upgraded(corpus):
    kwargs = options(corpus)
    candidate = YOLODataset(**copy.deepcopy(kwargs))
    assert accelerate_dataset(candidate) == 1
    assert candidate.format_class is Format
    assert candidate.collate_fn is YOLODataset.collate_fn
    candidate.transforms = candidate.build_transforms(copy.deepcopy(kwargs["hyp"]))
    assert type(formatter(candidate)) is Format
    assert accelerate_dataset(candidate) == 1
    assert accelerate_dataset(candidate, persistent=True) == 0
    assert candidate.format_class is FastFormat


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize("before_reset", ["first-batch", "full-epoch", "setup"])
def test_close_mosaic_retains_format_through_real_worker_reset(corpus, workers, overlap, before_reset):
    owners = []
    loader_options = {"multiprocessing_context": "spawn"} if workers else {}
    try:
        for native in (False, True):
            kwargs = options(corpus, overlap, mosaic=1.0)
            dataset = YOLODataset(**copy.deepcopy(kwargs))
            if native:
                assert accelerate_dataset(dataset, persistent=True) == 1
            loader = InfiniteDataLoader(
                dataset,
                batch_size=2,
                num_workers=workers,
                collate_fn=dataset.collate_fn if native else reference_transport_collate,
                **loader_options,
            )
            owner = SimpleNamespace(args=kwargs["hyp"], train_loader=loader)
            owners.append(owner)
            if before_reset == "first-batch":
                next(iter(loader))
            elif before_reset == "full-epoch":
                assert len(list(loader)) == 2
            previous_transforms = dataset.transforms
            previous_workers = list(getattr(loader.iterator, "_workers", ()))
            # This is the actual unmodified trainer method, also used on resume.
            SegmentationTrainer._close_dataloader_mosaic(owner)
            assert dataset.transforms is not previous_transforms
            assert type(formatter(dataset)) is (FastFormat if native else Format)
            assert all(getattr(owner.args, key) == 1.0 for key in ("mosaic", "copy_paste", "mixup", "cutmix"))
            loader.reset()  # The pinned epoch/resume caller performs this next.
            if workers:
                restarted = list(loader.iterator._workers)
                assert len(restarted) == workers
                assert all(not worker.is_alive() for worker in previous_workers)
                assert all(worker.exitcode == 0 for worker in previous_workers)
                assert not ({worker.pid for worker in previous_workers} & {worker.pid for worker in restarted})
        left, right = (list(owner.train_loader) for owner in owners)
        assert len(left) == len(right) == 2
        for expected, actual in zip(left, right):
            equal_batch(expected, actual)
        for owner in owners:
            remaining = list(getattr(owner.train_loader.iterator, "_workers", ()))
            owner.train_loader.close()
            assert all(not worker.is_alive() and worker.exitcode == 0 for worker in remaining)
    finally:
        for owner in owners:
            owner.train_loader.close()


def test_unknown_builder_profile_rejected_without_mutation(corpus, monkeypatch):
    import ultrafast_maskops.ultralytics as integration

    dataset = YOLODataset(**options(corpus))
    previous = formatter(dataset)
    monkeypatch.setattr(integration, "_BUILD_TRANSFORMS_SHA256", "unrecognized")
    with pytest.raises(RuntimeError, match="build_transforms source"):
        accelerate_dataset(dataset, persistent=True)
    assert formatter(dataset) is previous and dataset.format_class is Format


def test_custom_builder_rejected_without_mutation(corpus):
    class CustomDataset(YOLODataset):
        def build_transforms(self, hyp=None):
            return super().build_transforms(hyp)

    dataset = CustomDataset(**options(corpus))
    previous = formatter(dataset)
    with pytest.raises(TypeError, match="base segmentation build_transforms"):
        accelerate_dataset(dataset, persistent=True)
    assert formatter(dataset) is previous and dataset.format_class is Format


def test_custom_factory_rejected_without_mutation(corpus):
    dataset = YOLODataset(**options(corpus))
    previous = formatter(dataset)

    def factory(*args, **kwargs):
        return Format(*args, **kwargs)

    dataset.format_class = factory
    with pytest.raises(TypeError, match="custom format factories"):
        accelerate_dataset(dataset, persistent=True)
    assert formatter(dataset) is previous and dataset.format_class is factory


def test_custom_fast_format_settings_are_not_silently_lost(corpus):
    dataset = YOLODataset(**options(corpus))
    accelerate_dataset(dataset)
    previous = formatter(dataset)
    previous._maskops_mode = "bounded"
    with pytest.raises(TypeError, match="default FastFormat mode"):
        accelerate_dataset(dataset, persistent=True)
    assert formatter(dataset) is previous and dataset.format_class is Format


def test_custom_format_subclass_is_not_replaced_on_future_rebuild(corpus):
    class CustomFormat(Format):
        pass

    dataset = YOLODataset(**options(corpus))
    previous = formatter(dataset)
    custom = CustomFormat(return_mask=True)
    index = dataset.transforms.transforms.index(previous)
    dataset.transforms.transforms[index] = custom
    with pytest.raises(TypeError, match="Format subclasses"):
        accelerate_dataset(dataset, persistent=True)
    assert formatter(dataset) is custom and dataset.format_class is Format


def test_persistent_flag_requires_bool(corpus):
    dataset = YOLODataset(**options(corpus))
    previous = formatter(dataset)
    with pytest.raises(TypeError, match="persistent must be a bool"):
        accelerate_dataset(dataset, persistent=1)
    assert formatter(dataset) is previous and dataset.format_class is Format


def test_custom_collator_rejected_before_persistent_mutation(corpus):
    dataset = YOLODataset(**options(corpus))
    previous = formatter(dataset)

    def custom(batch):
        return batch

    dataset.collate_fn = custom
    with pytest.raises(TypeError, match="custom collators"):
        accelerate_dataset(dataset, persistent=True)
    assert formatter(dataset) is previous and dataset.format_class is Format
    assert dataset.collate_fn is custom
