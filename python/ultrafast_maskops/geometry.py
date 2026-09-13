"""Exact batched replicas of the pinned Ultralytics segment geometry.

Profiling augmented segmentation loading shows the per-instance
``segment2box`` loop inside ``RandomPerspective.apply_segments`` and the
per-polygon ``resample_segments`` loop costing several times more than mask
rasterization. Both are dominated by interpreter dispatch of small NumPy calls,
not arithmetic. Each is replaced by one native call per sample.

Results are byte-identical to the pinned reference: the affine matrix product
stays in NumPy, native arithmetic follows the reference operation by operation
in the input precision, and np.interp's multiply-add fusion (a property of the
NumPy build) is calibrated against the installed NumPy before first use.
Importing this module never patches Ultralytics.
"""

import functools
import hashlib
import inspect
import threading

import numpy as np
from ultralytics.data import augment, dataset
from ultralytics.data.augment import Compose, RandomPerspective
from ultralytics.utils import ops

from . import _native

_SOURCE_HASHES = {
    "resample_segments": "bc969b2f05d167b05dfc0d4203332cbc4f0c0b22afe3429339477872db8f47ff",
    "segment2box": "1085b7f35db3ecd9336d60ae363211808dcfb488e5ec1c344dbf300b75997e0d",
    "RandomPerspective.apply_segments": "7d09bad9f6f5acdcbd5fe06b0b6456dd3ba396066c977561a4a1ffad6225b59c",
    "YOLODataset.update_labels_info": "7bb1a13a6db3aa30898cf5ed231403d6bdbc6451897cf5e6592509c7f5ae5b0b",
}
_BOXES = {np.dtype(np.float32): _native.segment_boxes_f32, np.dtype(np.float64): _native.segment_boxes_f64}
_UNSET = object()
_interp_fused = _UNSET
_interp_lock = threading.Lock()


def check_geometry_profile():
    objects = {
        "resample_segments": ops.resample_segments,
        "segment2box": ops.segment2box,
        "RandomPerspective.apply_segments": RandomPerspective.apply_segments,
        "YOLODataset.update_labels_info": dataset.YOLODataset.update_labels_info,
    }
    for name, expected in _SOURCE_HASHES.items():
        if hashlib.sha256(inspect.getsource(objects[name]).encode()).hexdigest() != expected:
            raise RuntimeError(f"unsupported Ultralytics source for {name}; retain the original geometry")
    if augment.segment2box is not ops.segment2box or dataset.resample_segments is not ops.resample_segments:
        raise RuntimeError("overridden Ultralytics geometry alias; retain the original geometry")


def _packable(segments):
    return len(segments) > 0 and all(
        isinstance(s, np.ndarray) and s.dtype == np.float32 and s.ndim == 2 and s.shape[1] == 2 and len(s)
        for s in segments
    )


def _pack(segments):
    offsets = np.zeros(len(segments) + 1, dtype=np.int64)
    np.cumsum([len(s) for s in segments], out=offsets[1:])
    points = np.require(np.concatenate(segments), np.float32, ("C", "A"))
    return points, offsets


def _calibrate():
    """Select the np.interp multiply-add variant matching the installed NumPy.

    float32 outputs of resample_segments usually hide a one-ulp float64
    difference, so the variants are compared on np.interp's float64 output,
    where they differ for a sizable fraction of random queries.
    """
    rng = np.random.default_rng(20260913)
    fp = rng.random(64).astype(np.float32).astype(np.float64) * 640
    x = np.sort(np.concatenate([rng.random(20000) * 63, np.arange(64.0)]))
    expected = np.interp(x, np.arange(64.0), fp)
    variants = {fused: _native.interp_values(x, fp, fused) for fused in (True, False)}
    if variants[True].tobytes() == variants[False].tobytes():
        return None  # cannot tell the variants apart on this host
    fused = next((f for f, got in variants.items() if got.tobytes() == expected.tobytes()), None)
    if fused is None:
        return None
    # End-to-end check on polygons of every resample branch before enabling.
    segments = [rng.random((k, 2), dtype=np.float32) * np.float32(640) for k in (3, 17, 999, 1000, 1001, 1600)]
    reference = np.stack(ops.resample_segments([s.copy() for s in segments], n=1000), axis=0)
    points, offsets = _pack(segments)
    return fused if _native.resample_stack(points, offsets, 1000, fused).tobytes() == reference.tobytes() else None


def interp_fused():
    """True/False for the matching np.interp variant; None disables native resampling."""
    global _interp_fused
    if _interp_fused is _UNSET:
        with _interp_lock:
            if _interp_fused is _UNSET:
                _interp_fused = _calibrate()
    return _interp_fused


def resample_stack(segments, n):
    """``np.stack(resample_segments(segments, n), axis=0)`` in one native call."""
    fused = interp_fused()
    if fused is None or not _packable(segments):
        return np.stack(ops.resample_segments(list(segments), n=n), axis=0)
    points, offsets = _pack(segments)
    return _native.resample_stack(points, offsets, int(n), fused)


def segment_boxes(segments, width, height, clip):
    """``np.stack([segment2box(s, width, height) ...])`` plus the in-place clip.

    Batches containing NaN, infinity or negative zero run the reference code
    unchanged: NumPy's reductions and clip loops give such values a
    shape-dependent sign of zero that only the identical calls reproduce.
    """
    kernel = _BOXES.get(segments.dtype)
    aligned = segments.flags.c_contiguous and segments.flags.aligned and segments.flags.writeable
    bboxes = kernel(segments, int(width), int(height), bool(clip)) if kernel is not None and aligned else None
    if bboxes is None:
        bboxes = np.stack([augment.segment2box(xy, width, height) for xy in segments], 0)
        if clip:
            segments[..., 0] = segments[..., 0].clip(bboxes[:, 0:1], bboxes[:, 2:3])
            segments[..., 1] = segments[..., 1].clip(bboxes[:, 1:2], bboxes[:, 3:4])
    return bboxes


class FastRandomPerspective(RandomPerspective):
    """RandomPerspective whose segment boxes and clipping run natively."""

    def apply_segments(self, segments, M, size):
        """Pinned apply_segments; the NumPy matrix product is unchanged."""
        n, num = segments.shape[:2]
        if n == 0:
            return [], segments

        xy = np.ones((n * num, 3), dtype=segments.dtype)
        segments = segments.reshape(-1, 2)
        xy[:, :2] = segments
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]
        segments = xy.reshape(n, -1, 2)
        bboxes = segment_boxes(segments, size[0], size[1], not self.preserve_obb)
        return bboxes, segments


def _update_labels_info(self, label):
    """Pinned YOLODataset.update_labels_info with one native resampling call."""
    bboxes = label.pop("bboxes")
    segments = label.pop("segments", [])
    keypoints = label.pop("keypoints", None)
    bbox_format = label.pop("bbox_format")
    normalized = label.pop("normalized")

    # NOTE: do NOT resample oriented boxes
    segment_resamples = 100 if self.use_obb else 1000
    if len(segments) > 0:
        # make sure segments interpolate correctly if original length is greater than segment_resamples
        max_len = max(len(s) for s in segments)
        segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
        # list[np.array(segment_resamples, 2)] * num_samples
        segments = resample_stack(segments, segment_resamples)
    else:
        segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
    label["instances"] = dataset.Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
    return label


def _swap(root):
    """Retype every base RandomPerspective reachable from a transform graph."""
    seen, pending, count = set(), [root], 0
    while pending:
        node = pending.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if type(node) is RandomPerspective:
            node.__class__ = FastRandomPerspective  # shared by Mosaic/MixUp/CutMix/CopyPaste pipelines
            count += 1
        if isinstance(node, Compose):
            pending.extend(node.transforms)
        pending.append(getattr(node, "pre_transform", None))
    return count


def _build_transforms(self, original, hyp=None):
    transforms = original(self, hyp)
    _swap(transforms)
    return transforms


def accelerate_geometry(data, *, persistent=False):
    """Accelerate this dataset's segment resampling and affine boxes/clipping.

    Returns the number of RandomPerspective transforms replaced. persistent=True
    keeps the acceleration when transforms are rebuilt (e.g. close_mosaic).
    Call after ultrafast_maskops.ultralytics.accelerate_dataset when combining.
    """
    check_geometry_profile()
    if type(persistent) is not bool:
        raise TypeError("persistent must be a bool")
    if (
        not isinstance(data, dataset.YOLODataset)
        or type(data).update_labels_info is not dataset.YOLODataset.update_labels_info
    ):
        raise TypeError("custom datasets overriding update_labels_info require their own integration")
    count = _swap(getattr(data, "transforms", None))
    data.update_labels_info = functools.partial(_update_labels_info, data)
    if persistent:
        builder = getattr(data.build_transforms, "__func__", None)
        if builder is not dataset.YOLODataset.build_transforms:
            raise TypeError("persistent acceleration requires the base build_transforms hook")
        data.build_transforms = functools.partial(_build_transforms, data, builder)
    return count
