"""Explicit Format adapter; never patches installed module globals."""

import hashlib
import inspect
import os

import numpy as np
from ultralytics.data import utils
from ultralytics.data.augment import Format
from ultralytics.data.dataset import YOLODataset

from . import Rasterizer
from ._shared_collate import check_shared_collate_profile, shared_collate_fn
from ._shared_collate import share_dataset_batches as share_dataset_batches

_SOURCE_HASHES = {
    "Format": "5141a276c0d58af8175d53ef165e270e984e549b575408ce54154b0c5648cfdc",
    "polygon2mask": "7c26a4710e1dc89fb4bb80e7cd6abf216b7c56170786e3631e5e733980934f27",
    "polygons2masks": "0b06719c1864120ec1f1f39b6158971bf455ebc8a1b095f33057872cd1a48528",
    "polygons2masks_overlap": "95743aa94524665769288d311d1443acab92a59f2f9116c67d1616c98bf1e89f",
}
_BUILD_TRANSFORMS_SHA256 = "85dc9e28f59b2f7a54ced5190cede13c4a918348f5fa5a9b411db461d08f0c72"


def check_profile():
    """Reject Ultralytics sources whose mask code differs from the replicated one.

    The installed cv2 itself is not version-checked: the native kernel is
    verified against it per image size at first use, and any size it cannot
    reproduce runs the reference code with that cv2 (see backend_info()).
    """
    if int(np.__version__.split(".")[0]) < 2:
        raise RuntimeError("unsupported profile: the adapter requires NumPy 2")
    for name, expected in _SOURCE_HASHES.items():
        value = Format if name == "Format" else getattr(utils, name)
        actual = hashlib.sha256(inspect.getsource(value).encode()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"unsupported Ultralytics source for {name}; retain the original Format")


class FastFormat(Format):
    """Opt-in replacement for the validated Format source profile.

    Only rasterization changes. Format's image, semantic-mask, class, box, and
    tensor handling remain inherited. Per-process engines are created lazily.
    """

    def __init__(self, *args, mode="auto", scratch_limit_bytes=64 * 1024**2, **kwargs):
        check_profile()
        super().__init__(*args, **kwargs)
        self._maskops_mode = mode
        self._maskops_budget = scratch_limit_bytes
        self._maskops_engine = None
        self._maskops_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_maskops_engine"] = None
        state["_maskops_pid"] = None
        return state

    @classmethod
    def from_reference(cls, formatter):
        if type(formatter) is not Format:
            raise TypeError("custom Format subclasses require their own integration")
        result = cls()
        result.__dict__.update(formatter.__dict__)
        return result

    def _format_segments(self, instances, cls, w, h):
        pid = os.getpid()
        if self._maskops_engine is None or self._maskops_pid != pid:
            self._maskops_engine = Rasterizer(scratch_limit_bytes=self._maskops_budget)
            self._maskops_pid = pid
        engine, segments = self._maskops_engine, instances.segments
        if self.mask_overlap:
            masks, order = engine.overlap_segments((h, w), segments, self.mask_ratio, mode=self._maskops_mode)
            return masks[None], instances[order], cls[order]
        return engine.masks_segments((h, w), segments, 1, self.mask_ratio), instances, cls


def _check_persistent_profile(dataset):
    builder = getattr(getattr(dataset, "build_transforms", None), "__func__", None)
    if builder is not YOLODataset.build_transforms or not getattr(dataset, "use_segments", False):
        raise TypeError("persistent acceleration requires the base segmentation build_transforms hook")
    try:
        source_hash = hashlib.sha256(inspect.getsource(builder).encode()).hexdigest()
    except (OSError, TypeError):
        source_hash = None
    if source_hash != _BUILD_TRANSFORMS_SHA256:
        raise RuntimeError("unsupported Ultralytics build_transforms source; retain the original format factory")
    if getattr(dataset, "format_class", None) not in (Format, FastFormat):
        raise TypeError("custom format factories require their own integration")
    check_shared_collate_profile(dataset)


def accelerate_dataset(dataset, *, persistent=False):
    """Replace this dataset's base Format transforms and return their count.

    Call after construction in an explicit custom trainer/dataset factory.
    This does not change other datasets or globally imported functions.
    persistent=True also sets this instance's base format_class and shared CPU
    collator hooks. Rebuilds retain acceleration; batch tensors are prepared
    before IPC to avoid deferred storage sharing during worker teardown.
    Repeated opt-in returns zero if the current formatter is already native.
    """
    check_profile()
    if type(persistent) is not bool:
        raise TypeError("persistent must be a bool")
    transforms = getattr(getattr(dataset, "transforms", None), "transforms", None)
    if transforms is None:
        raise TypeError("expected an Ultralytics dataset with Compose.transforms")
    if persistent:
        _check_persistent_profile(dataset)
        if type(transforms) is not list or any(
            isinstance(t, Format) and type(t) not in (Format, FastFormat) for t in transforms
        ):
            raise TypeError("custom transform containers/Format subclasses require their own integration")
    indices = [i for i, t in enumerate(transforms) if type(t) is Format and t.return_mask]
    already_native = persistent and any(type(t) is FastFormat and t.return_mask for t in transforms)
    if not indices and not already_native:
        raise ValueError("no supported segmentation Format transform found")
    replacements = [(i, FastFormat.from_reference(transforms[i])) for i in indices]
    if persistent:
        native_formats = [t for t in transforms if type(t) is FastFormat] + [t for _, t in replacements]
        if any(t._maskops_mode != "auto" or t._maskops_budget != 64 * 1024**2 for t in native_formats):
            raise TypeError("persistent acceleration requires the default FastFormat mode and scratch budget")
        dataset.format_class = FastFormat
        dataset.collate_fn = shared_collate_fn
    for i, replacement in replacements:
        transforms[i] = replacement
    return len(replacements)
