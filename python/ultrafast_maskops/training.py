"""A drop-in Ultralytics segmentation trainer that accelerates its datasets.

    from ultralytics import YOLO
    from ultrafast_maskops.training import FastSegmentationTrainer

    YOLO("yolo11n-seg.pt").train(data="coco.yaml", trainer=FastSegmentationTrainer)

The trainer, model, loss, augmentation parameters and data files are the
unmodified Ultralytics ones. Only the datasets it builds are changed, after
construction, by accelerate_dataset (mask rasterization) and
accelerate_geometry (segment resampling and the affine boxes and clipping),
both in persistent mode so the acceleration survives close_mosaic. Every
batch is byte-identical to the unmodified trainer's.
"""

from ultralytics.models.yolo.segment import SegmentationTrainer

from .geometry import accelerate_geometry
from .ultralytics import accelerate_dataset


def accelerate(dataset):
    """Apply both accelerations to a YOLODataset built for segmentation."""
    accelerate_dataset(dataset, persistent=True)
    accelerate_geometry(dataset, persistent=True)
    return dataset


class FastSegmentationTrainer(SegmentationTrainer):
    """SegmentationTrainer whose train and val datasets are accelerated."""

    def build_dataset(self, img_path, mode="train", batch=None):
        return accelerate(super().build_dataset(img_path, mode, batch))
