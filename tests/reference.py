# Ultralytics AGPL-3.0 reference; unmodified function bodies from pinned source.
from __future__ import annotations
import cv2
import numpy as np

def polygon2mask(
    imgsz: tuple[int, int], polygons: list[np.ndarray], color: int = 1, downsample_ratio: int = 1
) -> np.ndarray:
    """Convert a list of polygons to a binary mask of the specified image size.

    Args:
        imgsz (tuple[int, int]): The size of the image as (height, width).
        polygons (list[np.ndarray]): A list of polygons. Each polygon is a 1D array of coordinates with length M, where
            M % 2 = 0 (alternating x, y values).
        color (int, optional): The color value to fill in the polygons on the mask.
        downsample_ratio (int, optional): Factor by which to downsample the mask.

    Returns:
        (np.ndarray): A binary mask of the specified image size with the polygons filled in.
    """
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # Note: fillPoly first then resize is trying to keep the same loss calculation method when mask-ratio=1
    return cv2.resize(mask, (nw, nh))

def polygons2masks(
    imgsz: tuple[int, int], polygons: list[np.ndarray], color: int, downsample_ratio: int = 1
) -> np.ndarray:
    """Convert a list of polygons to a set of binary masks of the specified image size.

    Args:
        imgsz (tuple[int, int]): The size of the image as (height, width).
        polygons (list[np.ndarray]): A list of polygons. Each polygon is an array of coordinates that can be reshaped to
            (-1, 2) as (x, y) point pairs.
        color (int): The color value to fill in the polygons on the masks.
        downsample_ratio (int, optional): Factor by which to downsample each mask.

    Returns:
        (np.ndarray): A set of binary masks of the specified image size with the polygons filled in.
    """
    return np.array([polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio) for x in polygons])

def polygons2masks_overlap(
    imgsz: tuple[int, int], segments: list[np.ndarray], downsample_ratio: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Return a downsampled overlap mask and sorted area indices."""
    masks = np.zeros(
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8,
    )
    areas = []
    ms = []
    for segment in segments:
        mask = polygon2mask(
            imgsz,
            [segment.reshape(-1)],
            downsample_ratio=downsample_ratio,
            color=1,
        )
        ms.append(mask.astype(masks.dtype))
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    # Running max: the old `masks + mask` sum hit 2 * i + 1 and overflowed uint8 past 128 overlapping instances
    for i in range(len(segments)):
        np.maximum(masks, ms[i] * (i + 1), out=masks)
    return masks, index
