"""Bitwise checks for crop boundaries, resize dispatch and reused output buffers."""

import numpy as np
import pytest
import reference
import ultrafast_maskops as native


def equal(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("ratio", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("color", [0, 1, 127, 255])
def test_small_roi_dispatch_boundaries(ratio, color):
    shape = (128, 256)
    engine = native.Rasterizer()
    polygons = []
    # Reduced ROI widths cross scalar, half-vector, vector and tail boundaries.
    # Unaligned starts and thin triangles exercise fractional edge masks.
    for width in (1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33):
        for start in (0, 1, 3, 7, 17, 255):
            x = start - 4
            y = (start * 3) % 128 - 2
            polygons.append(np.array([[x, y], [x + width * ratio, y + 1], [x + 1, y + ratio * 2]], np.float32))
    # The same instance buffer sees earlier nonzero values, an off-image shape,
    # then a new on-image shape in bounded overlap mode.
    polygons.extend([np.array([[400, 300], [500, 300], [400, 400]], np.float32), polygons[0]])
    packed = native.PackedPolygons.from_segments(polygons)
    equal(engine.masks(shape, packed, color, ratio), reference.polygons2masks(shape, polygons, color, ratio))
    for mode in ("bounded", "retained"):
        result, order = engine.overlap(shape, packed, ratio, mode=mode)
        expected, expected_order = reference.polygons2masks_overlap(shape, polygons, ratio)
        equal(result, expected)
        equal(order, expected_order)


@pytest.mark.parametrize("shape,ratio", [((127, 255), 4), ((128, 255), 4), ((126, 255), 3), ((96, 192), 6)])
def test_non_power_of_two_and_noninteger_scales(shape, ratio):
    polygons = [np.array([[3, 5], [28, 7], [6, 41]], np.float32), np.array([[30, 3], [70, 9], [65, 71]], np.float32)]
    packed = native.PackedPolygons.from_segments(polygons)
    engine = native.Rasterizer()
    for color in (1, 127, 255):
        equal(engine.masks(shape, packed, color, ratio), reference.polygons2masks(shape, polygons, color, ratio))
    for mode in ("bounded", "retained"):
        actual = engine.overlap(shape, packed, ratio, mode=mode)
        expected = reference.polygons2masks_overlap(shape, polygons, ratio)
        for a, b in zip(actual, expected):
            equal(a, b)


def test_combined_contours_keep_original_fill_coordinates():
    outer = np.array([[33, 17], [90, 17], [90, 74], [33, 74]], np.int32)
    inner = np.array([[41, 25], [82, 25], [82, 66], [41, 66]], np.int32)
    crossing = np.array([[-100, 21], [80, 21], [90, 50], [-100, 70]], np.int32)
    for ratio in (1, 2, 4, 8):
        for contours in ([outer, inner], [outer, crossing]):
            for color in (1, 127, 255):
                equal(
                    native.polygon2mask((128, 256), contours, color, ratio),
                    reference.polygon2mask((128, 256), contours, color, ratio),
                )


def test_many_equal_and_zero_areas_keep_uint64_sort_order():
    point = np.array([[33, 17]], np.float32)
    offscreen = np.array([[300, 200]], np.float32)
    triangle = np.array([[21, 13], [63, 17], [25, 49]], np.float32)
    polygons = [x.copy() for x in [triangle, offscreen, point] * 90]
    packed = native.PackedPolygons.from_segments(polygons)
    engine = native.Rasterizer()
    for ratio in (1, 2, 4, 8):
        expected = reference.polygons2masks_overlap((128, 256), polygons, ratio)
        for mode in ("bounded", "retained"):
            actual = engine.overlap((128, 256), packed, ratio, mode=mode)
            for a, b in zip(actual, expected):
                equal(a, b)


def test_large_coordinate_precision_guard():
    # Width beyond exact half-integer float coordinates: exercise full-resize
    # fallback without allocating a two-dimensional multi-gigabyte image.
    shape, ratio = (4, (1 << 23) + 8), 4
    x = shape[1] - 8
    polygon = np.array([[x, 0], [x + 3, 0], [x, 3]], np.int32)
    packed = native.PackedPolygons.from_segments([polygon])
    engine = native.Rasterizer()
    equal(engine.masks(shape, packed, 127, ratio), reference.polygons2masks(shape, [polygon], 127, ratio))


def test_half_scale_arm_hal_area_threshold():
    # The pinned KleidiCV adapter handles half-scale images above 150000 source
    # pixels. Cropping can cross that threshold and switch to OpenCV's area path.
    shape, ratio = (640, 640), 2
    engine = native.Rasterizer()
    for width in (300, 302, 304):
        height = 496
        polygon = np.array([[12, 18], [12 + width - 1, 18], [13, 18 + height - 1]], np.int32)
        packed = native.PackedPolygons.from_segments([polygon])
        for color in (1, 127, 255):
            equal(engine.masks(shape, packed, color, ratio), reference.polygons2masks(shape, [polygon], color, ratio))
