from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys

import numpy as np
import pytest

import reference
import ultrafast_maskops as native


def equal(a, b):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("n", [0, 1, 5, 20, 100, 127, 128, 129, 255, 256, 500])
@pytest.mark.parametrize("ratio", [1, 2, 4, 8])
@pytest.mark.parametrize("mode", ["retained", "bounded"])
def test_overlap_boundaries(n, ratio, mode):
    # Identical areas, zeros, clipping, and full overlaps in the same case.
    polygon = np.array([[-0.9, 0.9], [31.9, 0], [31, 29], [0, 29]], dtype=np.float64)
    segments = [polygon.copy() if i % 3 else polygon + 100 for i in range(n)]
    expected = reference.polygons2masks_overlap((31, 33), segments, ratio)
    got = native.Rasterizer().overlap((31, 33), native.PackedPolygons.from_segments(segments), ratio, mode=mode)
    for a, b in zip(expected, got):
        equal(a, b)


@pytest.mark.parametrize("color", [0, 1, 127, 255])
def test_multiple_contours_one_fill(color):
    contours = np.array([[[0, 0], [25, 0], [25, 25], [0, 25]],
                         [[5, 5], [20, 5], [20, 20], [5, 20]]], dtype=np.float64)
    for ratio in (1, 2, 4):
        equal(reference.polygon2mask((31, 33), contours, color, ratio),
              native.polygon2mask((31, 33), contours, color, ratio))


def test_empty_contracts():
    equal(reference.polygons2masks((16, 16), [], 1), native.polygons2masks((16, 16), [], 1))
    packed = native.PackedPolygons.from_segments([])
    result = native.Rasterizer().masks((16, 16), packed)
    assert result.shape == (0, 16, 16) and result.dtype == np.uint8


def test_snapshot_lifetime_and_concurrency():
    points = np.array([[0, 0], [30, 0], [20, 20]], dtype=np.int32)
    packed = native.PackedPolygons(points, np.array([0, 3], dtype=np.int64))
    expected = reference.polygons2masks_overlap((33, 31), [points.copy()], 2)
    points[:] = 999
    engine = native.Rasterizer()
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: engine.overlap((33, 31), packed, 2), range(32)))
    for result in results:
        for a, b in zip(expected, result):
            equal(a, b)
    results[0][0][:] = 123
    equal(results[1][0], expected[0])


def test_invalid_inputs():
    for value in (np.nan, np.inf, 2**40):
        with pytest.raises(ValueError):
            native.PackedPolygons.from_segments([[[0, 0], [value, 2]]])
    for offsets in ([1, 2], [0, 3], [0, 2, 1, 2]):
        with pytest.raises(ValueError):
            native.PackedPolygons(np.zeros((2, 2), np.int32), np.array(offsets, np.int64))
    p = native.PackedPolygons.from_segments([np.zeros((3, 2))])
    with pytest.raises(ValueError):
        native.Rasterizer(scratch_limit_bytes=16).overlap((32, 32), p)
    for r in (0, -1, 100):
        with pytest.raises(ValueError):
            native.Rasterizer().masks((32, 32), p, downsample_ratio=r)


def test_private_opencv_threads_and_import_order():
    for code in (
        "import cv2; cv2.setNumThreads(3); before=cv2.getNumThreads(); import ultrafast_maskops as m; "
        "assert cv2.getNumThreads()==before; assert m.backend_info()['private_opencv_threads']==1",
        "import ultrafast_maskops as m; import cv2; cv2.setNumThreads(4); "
        "assert m.backend_info()['private_opencv_threads']==1",
    ):
        subprocess.run([sys.executable, "-c", code], check=True)


def test_ndarray_packing_and_empty_contours():
    segments = np.arange(60, dtype=np.float64).reshape(5, 6, 2)[:, ::-1]
    for a, b in zip(reference.polygons2masks_overlap((31, 33), segments, 4),
                    native.polygons2masks_overlap((31, 33), segments, 4)):
        equal(a, b)
    with pytest.raises(ValueError, match="empty contours"):
        native.polygons2masks_overlap((16, 16), [np.empty((0, 2))])


def test_10000_seeded_differential_cases():
    rng = np.random.default_rng(20260912)
    engine = native.Rasterizer()
    for case in range(10000):
        shape = tuple(map(int, rng.integers(8, 65, 2)))
        ratio = (1, 2, 4, 8)[case % 4]
        n = int(rng.integers(1, 9))
        segments = [rng.uniform(-30, 90, (int(rng.integers(1, 18)), 2)) for _ in range(n)]
        if case % 5 == 0:
            segments = [s[::-1] for s in segments]
        for segment in segments:
            segment.flags.writeable = False
        before = [s.tobytes() for s in segments]
        color = (0, 1, 127, 255)[case % 4]
        packed = native.PackedPolygons.from_segments(segments)
        equal(reference.polygons2masks(shape, segments, color, ratio),
              engine.masks(shape, packed, color, ratio))
        got = engine.overlap(shape, packed, ratio, mode="bounded" if case % 2 else "retained")
        expected = reference.polygons2masks_overlap(shape, segments, ratio)
        for a, b in zip(expected, got):
            equal(a, b)
        assert before == [s.tobytes() for s in segments]
