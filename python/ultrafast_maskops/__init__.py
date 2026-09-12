"""Exact polygon mask operations for the documented, tested reference profile.

Importing this package never patches Ultralytics. No implicit fallback is used.
"""
import operator
import threading

import numpy as np

from . import _native

__version__ = "0.1.0a1"


def _coordinates(value):
    a = np.asarray(value)
    if a.dtype.kind not in "iuf" or not np.isfinite(a).all():
        raise ValueError("coordinates must be finite real numbers")
    if a.size and (a.min() < -(2**31) or a.max() > 2**31 - 1):
        raise ValueError("coordinates must fit int32 before conversion")
    return np.ascontiguousarray(a, dtype=np.int32)


class PackedPolygons:
    """Owned immutable snapshot; one contour per instance."""

    def __init__(self, points, offsets):
        points = np.asarray(points)
        offsets = np.asarray(offsets)
        if points.dtype != np.int32 or offsets.dtype != np.int64:
            raise ValueError("packed points/offsets require int32/int64")
        self._native = _native.Polygons(np.ascontiguousarray(points), np.ascontiguousarray(offsets))

    @classmethod
    def from_segments(cls, segments):
        if isinstance(segments, np.ndarray) and segments.ndim >= 2:
            a = _coordinates(segments)
            n = len(a)
            if n:
                a = a.reshape(n, -1, 2)
                return cls(a.reshape(-1, 2), np.arange(n + 1, dtype=np.int64) * a.shape[1])
        arrays = [_coordinates(s).reshape(-1, 2) for s in segments]
        offsets = np.empty(len(arrays) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum([len(a) for a in arrays], out=offsets[1:])
        points = np.concatenate(arrays) if arrays else np.empty((0, 2), dtype=np.int32)
        return cls(points, offsets)

    def __len__(self):
        return len(self._native)


def _dimensions(imgsz, ratio):
    if len(imgsz) != 2:
        raise ValueError("imgsz must be (height, width)")
    h, w = map(operator.index, imgsz)
    r = operator.index(ratio)
    if min(h, w, r) <= 0 or not h // r or not w // r:
        raise ValueError("positive dimensions and nonzero downsampled size required")
    return h, w, r


class Rasterizer:
    """Per-worker scratch reuse. Calls serialize; previous outputs remain owned.

    overlap mode 'retained' holds N resized uint8 masks; 'bounded' renders twice
    with O(HW + hw + N) scratch, excluding packed inputs and the returned output.
    'auto' retains masks only when they fit scratch_limit_bytes with raster scratch.
    """

    def __init__(self, num_threads=1, scratch_limit_bytes=64 * 1024**2):
        if num_threads != 1:
            raise ValueError("only one native worker is currently supported")
        self.scratch_limit_bytes = operator.index(scratch_limit_bytes)
        if self.scratch_limit_bytes <= 0:
            raise ValueError("scratch_limit_bytes must be positive")
        self._core = _native.Rasterizer(self.scratch_limit_bytes)
        self._lock = threading.Lock()

    def masks(self, imgsz, packed, color=1, downsample_ratio=1):
        h, w, r = _dimensions(imgsz, downsample_ratio)
        with self._lock:
            return self._core.raster(packed._native, h, w, r, operator.index(color), True)[0]

    def overlap(self, imgsz, packed, downsample_ratio=1, *, mode="auto"):
        h, w, r = _dimensions(imgsz, downsample_ratio)
        if mode not in ("auto", "retained", "bounded"):
            raise ValueError("mode must be auto, retained, or bounded")
        required = h * w + (len(packed) + 1) * (h // r) * (w // r)
        retained = mode == "retained" or (mode == "auto" and required <= self.scratch_limit_bytes)
        if mode == "retained" and required > self.scratch_limit_bytes:
            raise ValueError("retained masks exceed scratch_limit_bytes")
        with self._lock:
            masks, areas = self._core.raster(packed._native, h, w, r, 1, retained)
            # Preserve unsigned negation (including zero areas) and NumPy tie order.
            order = np.argsort(-areas)
            result = self._core.compose(packed._native, order.astype(np.int64, copy=False), h, w, r, masks, retained)
        return result, order


def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    h, w, r = _dimensions(imgsz, downsample_ratio)
    a = _coordinates(polygons)
    # Keep the reference reshape, including its rejection of an empty outer list.
    a = a.reshape(a.shape[0], -1, 2)
    if a.shape[1] == 0:
        raise ValueError("empty contours are unsupported in the compatible wrapper")
    packed = PackedPolygons.from_segments(a)
    core = _native.Rasterizer(max(64 * 1024**2, h * w + (h // r) * (w // r)))
    return core.single(packed._native, h, w, r, operator.index(color))


def _has_empty_contour(polygons):
    if isinstance(polygons, np.ndarray) and polygons.ndim >= 2:
        return len(polygons) > 0 and polygons.size == 0
    return any(np.size(p) == 0 for p in polygons)


def polygons2masks(imgsz, polygons, color, downsample_ratio=1):
    if len(polygons) == 0:
        return np.array([])
    if _has_empty_contour(polygons):
        raise ValueError("empty contours are unsupported in the compatible wrapper")
    return Rasterizer().masks(imgsz, PackedPolygons.from_segments(polygons), color, downsample_ratio)


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    if _has_empty_contour(segments):
        raise ValueError("empty contours are unsupported in the compatible wrapper")
    return Rasterizer().overlap(imgsz, PackedPolygons.from_segments(segments), downsample_ratio)


def backend_info():
    return {"version": __version__, "backend": "C++/OpenCV", "opencv": _native.opencv_version,
            "numpy": np.__version__, "native_workers": 1, "private_opencv_threads": _native.opencv_threads(), "fallback_count": 0,
            "reference_sha": "795a556942a12fe0124cf767888194a1d0b83e2e"}
