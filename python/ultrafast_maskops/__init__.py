"""Exact polygon mask operations for the documented, tested reference profile.

The native kernel computes the default Ultralytics segmentation masks
(``mask_ratio=4``, sides divisible by 4) from only the pixels the 4x downscale
reads, byte-identical to ``cv2.fillPoly`` followed by ``cv2.resize``. It is
verified against the installed cv2 the first time each image size is used;
every other input, and any size the verification rejects, runs the unmodified
reference functions with that cv2. Importing this package never patches
Ultralytics.
"""

import operator
import threading

import numpy as np

from . import _native, _reference

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
        self._points = np.ascontiguousarray(points).copy()
        self._offsets = np.ascontiguousarray(offsets).copy()
        self._native = _native.Polygons(self._points, self._offsets)

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

    def contours(self):
        """The contours as the reference functions receive them: int32 (M, 2) arrays."""
        return [self._points[self._offsets[i] : self._offsets[i + 1]] for i in range(len(self._offsets) - 1)]


def _dimensions(imgsz, ratio):
    if len(imgsz) != 2:
        raise ValueError("imgsz must be (height, width)")
    h, w = map(operator.index, imgsz)
    r = operator.index(ratio)
    if min(h, w, r) <= 0 or not h // r or not w // r:
        raise ValueError("positive dimensions and nonzero downsampled size required")
    return h, w, r


_sampled_tables = {}
_sampled_lock = threading.Lock()


def _sampled_table(h, w, r):
    """cv2.resize's value for each 2x2 sample pattern at this 4x output size.

    None when the size is outside the native path, the installed cv2.resize is
    not the same function of the four samples at every pixel, or the native
    masks of a set of probe polygons differ from the reference at this size.
    """
    if r != 4 or h % 4 or w % 4 or h > 1 << 15 or w > 1 << 15:
        return None
    with _sampled_lock:
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
    rng = np.random.default_rng(20260913)
    noise = rng.integers(0, 2, (h, w), dtype=np.uint8)
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
    # The whole native path against cv2.fillPoly + cv2.resize on probe
    # polygons at this size: dense outlines, self-intersections, vertices
    # beyond every border, thin slivers, a point and a segment.
    probes = _probe_polygons(rng, h, w)
    got = _native.Rasterizer().sampled_masks(probes, h, w, table.tobytes())
    if got is None or not np.array_equal(got, _reference.polygons2masks((h, w), probes, 1, 4)):
        return None
    return table.tobytes()


def _probe_polygons(rng, h, w, count=24, points=64):
    out = np.empty((count, points, 2), np.float32)
    for i in range(count):
        if i % 4 == 0:  # a dense ring, partly outside on the odd probes
            theta = np.sort(rng.random(points)) * 2 * np.pi
            centre, radius = rng.random(2) * (w, h), rng.random() * max(h, w) * (0.7 if i % 8 else 0.3)
            out[i, :, 0], out[i, :, 1] = centre[0] + radius * np.cos(theta), centre[1] + radius * np.sin(theta)
        elif i % 4 == 1:  # a self-intersecting star
            theta = rng.permutation(points) * 2 * np.pi / points
            centre, radius = rng.random(2) * (w, h), rng.random() * max(h, w) * 0.5
            out[i, :, 0], out[i, :, 1] = centre[0] + radius * np.cos(theta), centre[1] + radius * np.sin(theta)
        elif i % 4 == 2:  # random vertices around and beyond the image, repeated
            k = int(rng.integers(1, 9))
            out[i] = np.repeat(rng.uniform(-0.3, 1.3, (k, 2)) * (w, h), -(-points // k), axis=0)[:points]
        else:  # a thin sliver along the border
            a, b = rng.uniform(-3, max(h, w) + 3, (2, 2))
            d = rng.uniform(0, 3, 2)
            out[i] = np.repeat(np.array([a, b, b + d, a + d]), points // 4, axis=0)[:points]
    return out


class Rasterizer:
    """Per-worker native engine. Calls serialize; previous outputs remain owned.

    overlap mode 'retained' keeps every instance's downscaled mask between the
    area pass and the composition; 'bounded' renders each instance twice and
    keeps O(hw + N) scratch; 'auto' retains when the masks fit
    scratch_limit_bytes. Inputs outside the native path run the reference
    functions with the installed cv2.
    """

    def __init__(self, num_threads=1, scratch_limit_bytes=64 * 1024**2):
        if num_threads != 1:
            raise ValueError("only one native worker is currently supported")
        self.scratch_limit_bytes = operator.index(scratch_limit_bytes)
        if self.scratch_limit_bytes <= 0:
            raise ValueError("scratch_limit_bytes must be positive")
        self._core = _native.Rasterizer()
        self._lock = threading.Lock()

    def _retained(self, h, w, r, count, mode):
        if mode not in ("auto", "retained", "bounded"):
            raise ValueError("mode must be auto, retained, or bounded")
        required = (count + 1) * (h // r) * (w // r)
        if mode == "retained" and required > self.scratch_limit_bytes:
            raise ValueError("retained masks exceed scratch_limit_bytes")
        return mode == "retained" or (mode == "auto" and required <= self.scratch_limit_bytes)

    def _native_overlap(self, source, h, w, r, count, mode):
        table = _sampled_table(h, w, r)
        if table is None:
            return None
        retained = self._retained(h, w, r, count, mode)
        with self._lock:
            areas = self._core.sampled_raster(source, h, w, table, retained)
            if areas is None:
                return None
            # Preserve unsigned negation (including zero areas) and NumPy tie order.
            order = np.argsort(-areas)
            return self._core.sampled_compose(order.astype(np.int64, copy=False)), order

    def _native_masks(self, source, h, w, r, color):
        table = _sampled_table(h, w, r) if color == 1 else None
        if table is None:
            return None
        with self._lock:
            return self._core.sampled_masks(source, h, w, table)

    def masks(self, imgsz, packed, color=1, downsample_ratio=1):
        h, w, r = _dimensions(imgsz, downsample_ratio)
        color = operator.index(color)
        result = self._native_masks(packed._native, h, w, r, color)
        if result is not None:
            return result
        if len(packed) == 0:  # the reference returns a 1-D empty array here
            return np.zeros((0, h // r, w // r), np.uint8)
        return _reference.polygons2masks((h, w), packed.contours(), color, r)

    def overlap(self, imgsz, packed, downsample_ratio=1, *, mode="auto"):
        h, w, r = _dimensions(imgsz, downsample_ratio)
        result = self._native_overlap(packed._native, h, w, r, len(packed), mode)
        if result is not None:
            return result
        self._retained(h, w, r, len(packed), mode)  # the mode is validated either way
        return _reference.polygons2masks_overlap((h, w), packed.contours(), r)

    def masks_segments(self, imgsz, segments, color=1, downsample_ratio=1):
        """masks() of the float segments Format receives, converted natively."""
        h, w, r = _dimensions(imgsz, downsample_ratio)
        color = operator.index(color)
        if type(segments) is np.ndarray and segments.ndim == 3:
            result = self._native_masks(segments, h, w, r, color)
            if result is not None:
                return result
        return self.masks(imgsz, PackedPolygons.from_segments(segments), color, downsample_ratio)

    def overlap_segments(self, imgsz, segments, downsample_ratio=1, *, mode="auto"):
        """overlap() of the float segments Format receives, converted natively."""
        h, w, r = _dimensions(imgsz, downsample_ratio)
        if type(segments) is np.ndarray and segments.ndim == 3:
            result = self._native_overlap(segments, h, w, r, len(segments), mode)
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
    color = operator.index(color)
    if color < 0 or color > 255:
        raise ValueError("color must be in [0,255]")
    if len(a) == 1:  # several contours in one call fill by the even-odd rule together
        result = Rasterizer()._native_masks(PackedPolygons.from_segments(a)._native, h, w, r, color)
        if result is not None:
            return result[0]
    return _reference.polygon2mask((h, w), list(a), color, r)


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
        "backend": "C++ (no bundled OpenCV); reference cv2 outside the native profile",
        "numpy": np.__version__,
        "native_workers": 1,
        "build": _native.build_profile(),
        "simd": _native.simd_mode(),  # "scalar" when ULTRAFAST_MASKOPS_SCALAR=1 at import
        "native_sizes": sorted(k for k, v in _sampled_tables.items() if v is not None),
        "rejected_sizes": sorted(k for k, v in _sampled_tables.items() if v is None),
        "reference_sha": "795a556942a12fe0124cf767888194a1d0b83e2e",
    }
