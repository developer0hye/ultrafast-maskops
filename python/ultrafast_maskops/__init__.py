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
    # Python float comparisons avoid NumPy's weak scalar promotion rounding
    # INT_MAX up to 2**31 when the coordinate array has float32 dtype.
    if a.size and (float(a.min()) < -(2**31) or float(a.max()) > 2**31 - 1):
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


_sampled_tables = {}


def _sampled_table(h, w, r):
    """cv2.resize's value for each 2x2 sample pattern at this 4x output size.

    None when the size is outside the exact sampled path or the installed
    cv2.resize is not the same function of the four samples at every pixel.
    """
    if r != 4 or h % 4 or w % 4 or h > 1 << 15 or w > 1 << 15:
        return None
    try:
        return _sampled_tables[(h, w)]
    except KeyError:
        table = _sampled_tables[(h, w)] = _calibrate_sampled_table(h, w)
        return table


def _calibrate_sampled_table(h, w):
    try:
        import cv2
    except ImportError:
        return None
    # Pattern bits of the pixels a 4x linear downscale reads; see sampled.hpp.
    bits = np.zeros((h, w), np.uint8)
    bits[1::4, 1::4], bits[1::4, 2::4], bits[2::4, 1::4], bits[2::4, 2::4] = 8, 4, 2, 1
    sampled = bits > 0
    noise = np.random.default_rng(20260913).integers(0, 2, (h, w), dtype=np.uint8)
    size = (w // 4, h // 4)
    table = np.zeros(16, np.uint8)
    # Every output pixel sees each pattern once, with random unread pixels.
    for pattern in range(16):
        image = noise.copy()
        image[sampled] = (bits[sampled] & pattern) != 0
        out = cv2.resize(image, size)
        value = int(out.flat[0])
        if value > 1 or (out != value).any():
            return None
        table[pattern] = value
    for image in (noise, 1 - noise, np.roll(noise, 1, 0), np.roll(noise, 1, 1)):
        patterns = (image * bits).reshape(h // 4, 4, w // 4, 4).sum(axis=(1, 3))
        if not np.array_equal(cv2.resize(np.ascontiguousarray(image), size), table[patterns]):
            return None
    return table.tobytes()


class Rasterizer:
    """Per-worker scratch reuse. Calls serialize; previous outputs remain owned.

    overlap mode 'retained' holds N resized uint8 masks; 'bounded' renders twice
    with O(HW + hw + N) scratch, excluding packed inputs and the returned output.
    'auto' retains masks only when they fit scratch_limit_bytes with raster scratch.

    At a 4x downsample with sides divisible by 4 and color 1, masks are computed
    from only the pixels the downscale reads (see src/sampled.hpp); the result
    is identical, and other inputs use the full-resolution OpenCV path.
    """

    def __init__(self, num_threads=1, scratch_limit_bytes=64 * 1024**2):
        if num_threads != 1:
            raise ValueError("only one native worker is currently supported")
        self.scratch_limit_bytes = operator.index(scratch_limit_bytes)
        if self.scratch_limit_bytes <= 0:
            raise ValueError("scratch_limit_bytes must be positive")
        self._core = _native.Rasterizer(self.scratch_limit_bytes)
        self._lock = threading.Lock()

    def _retained(self, h, w, r, count, mode):
        if mode not in ("auto", "retained", "bounded"):
            raise ValueError("mode must be auto, retained, or bounded")
        required = h * w + (count + 1) * (h // r) * (w // r)
        if mode == "retained" and required > self.scratch_limit_bytes:
            raise ValueError("retained masks exceed scratch_limit_bytes")
        return mode == "retained" or (mode == "auto" and required <= self.scratch_limit_bytes)

    def _sampled_overlap(self, source, h, w, table, retained):
        areas = self._core.sampled_raster(source, h, w, table, retained)
        if areas is None:
            return None
        # Preserve unsigned negation (including zero areas) and NumPy tie order.
        order = np.argsort(-areas)
        return self._core.sampled_compose(order.astype(np.int64, copy=False)), order

    def masks(self, imgsz, packed, color=1, downsample_ratio=1):
        h, w, r = _dimensions(imgsz, downsample_ratio)
        color = operator.index(color)
        table = _sampled_table(h, w, r) if color == 1 else None
        with self._lock:
            if table is not None:
                result = self._core.sampled_masks(packed._native, h, w, table)
                if result is not None:
                    return result
            return self._core.masks(packed._native, h, w, r, color)

    def overlap(self, imgsz, packed, downsample_ratio=1, *, mode="auto"):
        h, w, r = _dimensions(imgsz, downsample_ratio)
        retained = self._retained(h, w, r, len(packed), mode)
        table = _sampled_table(h, w, r)
        with self._lock:
            if table is not None:
                result = self._sampled_overlap(packed._native, h, w, table, retained)
                if result is not None:
                    return result
            masks, areas = self._core.raster(packed._native, h, w, r, 1, retained)
            # Preserve unsigned negation (including zero areas) and NumPy tie order.
            order = np.argsort(-areas)
            result = self._core.compose(packed._native, order.astype(np.int64, copy=False), h, w, r, masks, retained)
        return result, order

    def masks_segments(self, imgsz, segments, color=1, downsample_ratio=1):
        """masks() of PackedPolygons.from_segments(segments), converting float arrays natively."""
        h, w, r = _dimensions(imgsz, downsample_ratio)
        if type(segments) is np.ndarray and segments.ndim == 3 and operator.index(color) == 1:
            table = _sampled_table(h, w, r)
            if table is not None:
                with self._lock:
                    result = self._core.sampled_masks(segments, h, w, table)
                if result is not None:
                    return result
        return self.masks(imgsz, PackedPolygons.from_segments(segments), color, downsample_ratio)

    def overlap_segments(self, imgsz, segments, downsample_ratio=1, *, mode="auto"):
        """overlap() of PackedPolygons.from_segments(segments), converting float arrays natively."""
        h, w, r = _dimensions(imgsz, downsample_ratio)
        if type(segments) is np.ndarray and segments.ndim == 3:
            table = _sampled_table(h, w, r)
            if table is not None:
                retained = self._retained(h, w, r, len(segments), mode)
                with self._lock:
                    result = self._sampled_overlap(segments, h, w, table, retained)
                if result is not None:
                    return result
        return self.overlap(imgsz, PackedPolygons.from_segments(segments), downsample_ratio, mode=mode)


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
    if len(segments) == 0:
        h, w, r = _dimensions(imgsz, downsample_ratio)
        return np.zeros((h // r, w // r), dtype=np.uint8), np.empty(0, dtype=np.intp)
    if _has_empty_contour(segments):
        raise ValueError("empty contours are unsupported in the compatible wrapper")
    return Rasterizer().overlap(imgsz, PackedPolygons.from_segments(segments), downsample_ratio)


def backend_info():
    return {
        "version": __version__,
        "backend": "C++/OpenCV",
        "opencv": _native.opencv_version,
        "numpy": np.__version__,
        "native_workers": 1,
        "build": _native.build_profile(),
        "private_opencv_threads": _native.opencv_threads(),
        "simd": _native.simd_mode(),  # "scalar" when ULTRAFAST_MASKOPS_SCALAR=1 at import
        "fallback_count": 0,
        "reference_sha": "795a556942a12fe0124cf767888194a1d0b83e2e",
    }
