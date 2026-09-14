"""The pinned Ultralytics mask functions, unmodified (AGPL-3.0).

Inputs the native kernel does not handle (other downsample ratios, sides not
divisible by 4, colors other than 1, several contours in one mask, coordinates
outside the exact envelope, or a cv2 whose results the calibration could not
reproduce) run these with the cv2 the caller has installed, exactly as the
reference dataset pipeline would.

Source: ultralytics/data/utils.py at 795a556942a12fe0124cf767888194a1d0b83e2e.
"""

from __future__ import annotations

import numpy as np


def polygon2mask(
    imgsz: tuple[int, int], polygons: list[np.ndarray], color: int = 1, downsample_ratio: int = 1
) -> np.ndarray:
    import cv2

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
    return np.array([polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio) for x in polygons])


def polygons2masks_overlap(
    imgsz: tuple[int, int], segments: list[np.ndarray], downsample_ratio: int = 1
) -> tuple[np.ndarray, np.ndarray]:
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
