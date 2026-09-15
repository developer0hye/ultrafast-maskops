"""RF-DETR adapter: polygons ride through the geometric transforms and are
rasterized once, at the pixels those transforms keep.

RF-DETR's ``ConvertCoco`` rasterizes every polygon at full resolution with
pycocotools, and each ``Resize``/``RandomSizedCrop``/``RandomHorizontalFlip``
then resamples the ``(N, H, W)`` mask tensor. This adapter keeps the polygons
in the target instead, records the geometric operations the random transforms
chose, and evaluates the pycocotools fill only at the sampled pixels. Boxes,
labels, images and random draws are untouched: the traced transforms call the
RF-DETR implementations and only add a note of what they did.

Use ``accelerate_dataset(dataset)`` on an RF-DETR ``CocoDetection`` built with
``include_masks=True``. Samples whose segmentations are not plain polygon lists
(RLE, bbox-shaped lists) take RF-DETR's own path unchanged.
"""

import hashlib
import inspect

import numpy as np
import torch
from rfdetr.datasets import _torchvision as tv
from rfdetr.datasets import coco as rf_coco
from rfdetr.datasets.transforms import Normalize
from torchvision.transforms.v2 import ToDtype, ToImage

from . import _native
from ._chain import index_maps

_SOURCE_HASHES = {
    "ConvertCoco": "db0a038a400bc5dddb05759e2d4c2aaa1817400ccbc488f1b484a847d32dabe5",
    "convert_coco_poly_to_mask": "a5deeb42fac0487f4c1b7c9e117a0fece66b00d5f0a469d7963f1c9acff67a8b",
    "CocoDetection.__getitem__": "833b703e3784ea6232da7464482cc827875d34c9d9d759dc41de6b736bfbb2e8",
    "Resize": "365bfbb03701a8d4ca96855f93e1c5752c076946626c90ad075bdb7729a68353",
    "RandomResize": "a04b5c20cc3c3fa092dcd30c99cd0821e6b80d8976c0ea5868bd7fe5dff21e62",
    "RandomSizedCrop": "afe0ab80162204fabd96b9765bd0f23fc6973a87e22ebcec1cc60979ed593886",
    "RandomHorizontalFlip": "c7c4811304730f3576149a14fc7c77872aa5818fcc5c0a7dc6746aa6e5434f08",
    "crop": "e8ff010709b3dc917ae32df6dc6a1b7982939938b0a50b0c28eac0c526f241b5",
    "_filter_per_instance_fields": "3c31085d7c4310e7010d0839c30e20fad5367b17b1bd0c526f32a6be15411b17",
    "_apply_to_masks": "8067ccf40aa2b7d1b9ee51233279a06eb4c9fed5163d7e5740f88eea17dbc3ce",
    "Compose": "ef8441faf445973eb34885c6195a2ff0f89ce48dbe26e776ad757bc60f2352f9",
    "RandomSelect": "a4b7628712b1959de455e157e4b8fa521897e8603d6b87a2dc4fe181a1b9fac0",
    "RandomChoice": "39e2aae0302a299188f12f1eb3ad87e3b8302376359962e7b0a4b83e10d84a51",
}


def _pinned():
    return {
        "ConvertCoco": rf_coco.ConvertCoco,
        "convert_coco_poly_to_mask": rf_coco.convert_coco_poly_to_mask,
        "CocoDetection.__getitem__": rf_coco.CocoDetection.__getitem__,
        "Resize": tv.Resize,
        "RandomResize": tv.RandomResize,
        "RandomSizedCrop": tv.RandomSizedCrop,
        "RandomHorizontalFlip": tv.RandomHorizontalFlip,
        "crop": tv.crop,
        "_filter_per_instance_fields": tv._filter_per_instance_fields,
        "_apply_to_masks": tv._apply_to_masks,
        "Compose": tv.Compose,
        "RandomSelect": tv.RandomSelect,
        "RandomChoice": tv.RandomChoice,
    }


def source_hashes():
    """SHA-256 of the installed RF-DETR sources this adapter replicates."""
    return {name: hashlib.sha256(inspect.getsource(value).encode()).hexdigest() for name, value in _pinned().items()}


def check_profile():
    """Reject RF-DETR sources whose mask or transform code differs from the replicated one."""
    for name, actual in source_hashes().items():
        if actual != _SOURCE_HASHES[name]:
            raise RuntimeError(f"unsupported RF-DETR source for {name}; keep the original dataset")


class Polygons:
    """The polygons of one instance, as float64 (x, y, ...) coordinate arrays.

    Not a Sequence: torchvision's tree transforms leave it alone, while the
    list holding one per instance is filtered like every per-instance field.
    """

    __slots__ = ("polygons",)

    def __init__(self, polygons):
        self.polygons = polygons


class Trace:
    """The geometric operations applied so far, shared by every copy of a target."""

    __slots__ = ("ops",)

    def __init__(self):
        self.ops = []


def _record(target, op):
    if target is not None and "maskops_trace" in target:
        target["maskops_trace"].ops.append(op)


def _eligible(segmentation):
    """Whether pycocotools would treat the segmentation as polygons (frPoly)."""
    if segmentation is None:
        return True
    if not isinstance(segmentation, list):
        return False
    return not segmentation or (all(isinstance(p, list) for p in segmentation) and len(segmentation[0]) > 4)


class FastConvertCoco:
    """ConvertCoco that keeps polygons in the target instead of masks.

    Everything else is RF-DETR's ConvertCoco with include_masks=False; the
    instance filter is recomputed here exactly as ConvertCoco does, so the
    polygon list lines up with the kept boxes. Samples pycocotools would not
    rasterize as polygons go through the original ConvertCoco.
    """

    def __init__(self, convert):
        check_profile()
        if not convert.include_masks:
            raise ValueError("the dataset was built without masks; nothing to accelerate")
        self.include_masks = True
        self.eager = convert
        self.lean = rf_coco.ConvertCoco(
            include_masks=False,
            include_keypoints=convert.include_keypoints,
            cat2label=convert.cat2label,
            num_keypoints_per_class=[convert.num_keypoints] if convert.num_keypoints else None,
        )

    def __call__(self, image, target):
        anno = [obj for obj in target["annotations"] if "iscrowd" not in obj or obj["iscrowd"] == 0]
        if not anno or "segmentation" not in anno[0]:
            return self.eager(image, target)
        segmentations = [obj.get("segmentation", []) for obj in anno]
        if not all(_eligible(s) for s in segmentations):
            return self.eager(image, target)
        image, out = self.lean(image, target)
        w, h = image.size
        boxes = np.asarray([obj["bbox"] for obj in anno], dtype=np.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, np.float32(w))
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, np.float32(h))
        keep = np.flatnonzero((boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0]))
        if len(keep) != int(out["boxes"].shape[0]):
            raise RuntimeError("instance filter mismatch with ConvertCoco")
        out["maskops_polygons"] = [
            Polygons([np.asarray(p, dtype=np.float64) for p in segmentations[i] or []]) for i in keep
        ]
        out["maskops_trace"] = Trace()
        return image, out


class TracedResize(tv.Resize):
    def __call__(self, image, target):
        old = tv._image_size(image)
        image, target = super().__call__(image, target)
        _record(target, ("resize", old, self.size))
        return image, target


class TracedRandomResize(tv.RandomResize):
    # RandomResize.__call__ with its Resize replaced; same random draws.
    def __call__(self, image, target):
        index = int(torch.randint(len(self.sizes), ()).item()) if len(self.sizes) > 1 else 0
        size = self.sizes[index]
        height, width = tv._image_size(image)
        return TracedResize(self._get_size(height, width, size, self.max_size))(image, target)


class TracedRandomSizedCrop(tv.RandomSizedCrop):
    # RandomSizedCrop.__call__ with the crop and Resize noted; same random draws.
    def __call__(self, image, target):
        height, width = tv._image_size(image)
        min_crop, max_crop = self.min_max_height
        max_crop = min(max_crop, height, width)
        min_crop = min(min_crop, max_crop)
        crop_size = int(torch.randint(min_crop, max_crop + 1, ()).item()) if max_crop > min_crop else max_crop
        crop_height = min(crop_size, height)
        crop_width = min(crop_size, width)
        top = int(torch.randint(0, height - crop_height + 1, ()).item()) if height > crop_height else 0
        left = int(torch.randint(0, width - crop_width + 1, ()).item()) if width > crop_width else 0
        image, target = tv.crop(image, target, top, left, crop_height, crop_width)
        _record(target, ("crop", top, left, crop_height, crop_width))
        return TracedResize(self.size)(image, target)


class TracedRandomHorizontalFlip(tv.RandomHorizontalFlip):
    def __call__(self, image, target):
        # The original returns its inputs untouched when the draw says no flip.
        out_image, out_target = super().__call__(image, target)
        if out_image is not image:
            _record(out_target, ("hflip",))
        return out_image, out_target


class MaterializeMasks:
    """Rasterizes the traced polygons into target['masks'] at the output size."""

    def __call__(self, image, target):
        if target is None or "maskops_trace" not in target:
            return image, target
        target = target.copy()
        trace = target.pop("maskops_trace")
        polygons = target.pop("maskops_polygons")
        h, w = (int(v) for v in target["orig_size"])
        rows, cols = index_maps(trace.ops, h, w)
        masks = _native.rfdetr_masks([p.polygons for p in polygons], h, w, rows, cols)
        target["masks"] = torch.from_numpy(masks)
        return image, target


_PASSTHROUGH = (ToImage, ToDtype, Normalize, MaterializeMasks)


def trace_transforms(transform):
    """A copy of an RF-DETR transform tree whose geometric leaves record what they did.

    Only the transform types RF-DETR's torchvision pipeline builds are
    accepted; anything else (an Albumentations wrapper, a custom transform)
    raises, since its effect on masks would be unknown.
    """
    kind = type(transform)
    if kind is tv.Compose:
        return tv.Compose([trace_transforms(t) for t in transform.transforms])
    if kind is tv.RandomSelect:
        return tv.RandomSelect(trace_transforms(transform.transform1), trace_transforms(transform.transform2), transform.p)
    if kind is tv.RandomChoice:
        return tv.RandomChoice([trace_transforms(t) for t in transform.transforms])
    if kind is tv.Resize:
        return TracedResize(transform.size)
    if kind is tv.RandomResize:
        return TracedRandomResize(transform.sizes, transform.max_size)
    if kind is tv.RandomSizedCrop:
        return TracedRandomSizedCrop(transform.min_max_height, transform.size)
    if kind is tv.RandomHorizontalFlip:
        return TracedRandomHorizontalFlip(transform.p, transform.keypoint_flip_pairs)
    if isinstance(transform, _PASSTHROUGH):
        return transform
    raise RuntimeError(f"unsupported transform {kind.__name__}; keep the original dataset")


def accelerate_dataset(dataset):
    """Switch an RF-DETR CocoDetection (include_masks=True) to deferred rasterization, in place.

    The dataset's ``prepare`` becomes a FastConvertCoco and its transforms a
    traced copy ending in MaterializeMasks. Returns the dataset.
    """
    check_profile()
    if not isinstance(dataset.prepare, rf_coco.ConvertCoco):
        raise ValueError("expected an RF-DETR CocoDetection with its ConvertCoco prepare step")
    dataset.prepare = FastConvertCoco(dataset.prepare)
    transforms = dataset._transforms
    traced = trace_transforms(transforms) if transforms is not None else tv.Compose([])
    if type(traced) is not tv.Compose:
        traced = tv.Compose([traced])
    traced.transforms.append(MaterializeMasks())
    dataset._transforms = traced
    return dataset
