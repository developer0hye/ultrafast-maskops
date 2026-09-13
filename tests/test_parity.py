import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

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
    contours = np.array([[[0, 0], [25, 0], [25, 25], [0, 25]], [[5, 5], [20, 5], [20, 20], [5, 20]]], dtype=np.float64)
    for ratio in (1, 2, 4):
        equal(
            reference.polygon2mask((31, 33), contours, color, ratio),
            native.polygon2mask((31, 33), contours, color, ratio),
        )


def test_empty_contracts():
    equal(reference.polygons2masks((16, 16), [], 1), native.polygons2masks((16, 16), [], 1))
    packed = native.PackedPolygons.from_segments([])
    result = native.Rasterizer().masks((16, 16), packed)
    assert result.shape == (0, 16, 16) and result.dtype == np.uint8
    empty_contour = native.PackedPolygons(np.empty((0, 2), np.int32), np.array([0, 0], np.int64))
    for mode in ("retained", "bounded"):
        mask, order = native.Rasterizer().overlap((16, 16), empty_contour, mode=mode)
        assert not mask.any() and order.tolist() == [0]


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
    with pytest.raises(ValueError, match="int32"):
        native.PackedPolygons.from_segments(np.full((1, 3, 2), 2**31, dtype=np.float32))
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
        (
            "import cv2; cv2.setNumThreads(3); before=cv2.getNumThreads(); import ultrafast_maskops as m; "
            "assert cv2.getNumThreads()==before; assert m.backend_info()['private_opencv_threads']==1"
        ),
        (
            "import ultrafast_maskops as m; import cv2; cv2.setNumThreads(4); "
            "assert m.backend_info()['private_opencv_threads']==1"
        ),
    ):
        subprocess.run([sys.executable, "-c", code], check=True)


def test_ndarray_packing_and_empty_contours():
    segments = np.arange(60, dtype=np.float64).reshape(5, 6, 2)[:, ::-1]
    for a, b in zip(
        reference.polygons2masks_overlap((31, 33), segments, 4), native.polygons2masks_overlap((31, 33), segments, 4)
    ):
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
        equal(reference.polygons2masks(shape, segments, color, ratio), engine.masks(shape, packed, color, ratio))
        got = engine.overlap(shape, packed, ratio, mode="bounded" if case % 2 else "retained")
        expected = reference.polygons2masks_overlap(shape, segments, ratio)
        for a, b in zip(expected, got):
            equal(a, b)
        assert before == [s.tobytes() for s in segments]


def test_misaligned_packed_input_is_copied_safely():
    original = np.array([[1, 1], [27, 1], [27, 23], [1, 23], [4, 4], [18, 5], [13, 18]], np.int32)
    points = np.ndarray(original.shape, np.int32, buffer=bytearray(original.nbytes + 1), offset=1)
    offsets = np.ndarray((3,), np.int64, buffer=bytearray(25), offset=1)
    points[:] = original
    offsets[:] = [0, 4, 7]
    assert points.flags.c_contiguous and offsets.flags.c_contiguous
    assert not points.flags.aligned and not offsets.flags.aligned
    packed = native.PackedPolygons(points, offsets)
    # Extents and points must both belong to the immutable snapshot.
    points[:] = -100
    offsets[:] = 0
    actual = native.Rasterizer().overlap((31, 33), packed, 3)
    expected = reference.polygons2masks_overlap((31, 33), [original[:4], original[4:]], 3)
    assert actual[0].dtype == expected[0].dtype
    assert np.array_equal(actual[0], expected[0]) and np.array_equal(actual[1], expected[1])
    # The private composition entry point also accepts contiguous unaligned
    # NumPy input, so it must copy order bytes before reading int64 indices.
    core = native._native.Rasterizer(64 * 1024**2)
    masks, areas = core.raster(packed._native, 31, 33, 3, 1, True)
    order = np.ndarray((2,), np.int64, buffer=bytearray(17), offset=1)
    order[:] = np.argsort(-areas)
    assert not order.flags.aligned
    equal(core.compose(packed._native, order, 31, 33, 3, masks, True), expected[0])


@pytest.mark.parametrize("mode", ["retained", "bounded"])
def test_cached_geometry_extents_follow_each_raster_size(mode):
    segments = [
        np.array([[-9, -12], [60, -2], [70, 42], [-8, 40]], np.int32),
        np.array([[1, 1], [32, 1], [32, 24], [1, 24]], np.int32),
        np.array([[4, 3], [15, 28], [31, 4]], np.int32),
    ]
    packed = native.PackedPolygons.from_segments(segments)
    engine = native.Rasterizer()
    for h, w, ratio in [(31, 33, 1), (65, 47, 3), (9, 13, 2), (31, 33, 4), (65, 47, 1)]:
        expected = reference.polygons2masks_overlap((h, w), segments, ratio)
        actual = engine.overlap((h, w), packed, ratio, mode=mode)
        assert actual[0].dtype == expected[0].dtype
        assert np.array_equal(actual[0], expected[0]) and np.array_equal(actual[1], expected[1])
        # The multi-contour union must use combined extents, even after overlap
        # left the retained scratch dirty with a differently sized last contour.
        wanted = reference.polygon2mask((h, w), [segments[0], segments[1]], 1, ratio)
        pair = native.PackedPolygons.from_segments([segments[0], segments[1]])
        assert np.array_equal(engine._core.single(pair._native, h, w, ratio, 1), wanted)


def test_compiled_profile_matches_kernel_and_build_sources():
    import hashlib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    profile = native.backend_info()["build"]
    for key, name in [
        ("bindings_sha256", "src/bindings.cpp"),
        ("geometry_sha256", "src/geometry.hpp"),
        ("sampled_sha256", "src/sampled.hpp"),
        ("cmake_sha256", "CMakeLists.txt"),
        ("template_sha256", "src/build_profile.h.in"),
    ]:
        assert profile[key] == hashlib.sha256((root / name).read_bytes()).hexdigest()
    assert profile["compiler"] and profile["build_type"]


@pytest.mark.parametrize("ratio", [1, 3, 4, 7])
@pytest.mark.parametrize("color", [0, 1, 127, 255])
def test_repeated_vertices_keep_edges_holes_and_degenerate_pixels(ratio, color):
    contours = np.array(
        [
            [[1, 1], [31, 1], [31, 25], [1, 25], [1, 1]],
            [[7, 7], [24, 7], [24, 20], [7, 20], [7, 7]],
            [[12, 12]] * 5,  # A degenerate contour inside the even-odd hole.
            [[-2147483648, -2147483648]] * 5,
        ],
        dtype=np.int32,
    )
    repeated = np.repeat(contours, 31, axis=1)
    before = repeated.tobytes()
    packed = native.PackedPolygons.from_segments(repeated)
    engine = native.Rasterizer()
    shape = (33, 39)
    equal(engine.masks(shape, packed, color, ratio), reference.polygons2masks(shape, repeated, color, ratio))
    equal(
        engine._core.single(packed._native, *shape, ratio, color), reference.polygon2mask(shape, repeated, color, ratio)
    )
    wanted = reference.polygons2masks_overlap(shape, repeated, ratio)
    for mode in ["retained", "bounded"]:
        actual = engine.overlap(shape, packed, ratio, mode=mode)
        equal(actual[0], wanted[0])
        equal(actual[1], wanted[1])
    assert repeated.tobytes() == before


@pytest.mark.parametrize("count", [255, 256])
def test_repeated_points_keep_instance_count_dtype_and_tie_order(count):
    segments = np.full((count, 43, 2), 9, dtype=np.int32)
    packed = native.PackedPolygons.from_segments(segments)
    assert len(packed) == count
    wanted = reference.polygons2masks_overlap((17, 19), segments, 1)
    actual = native.Rasterizer().overlap((17, 19), packed, 1)
    equal(actual[0], wanted[0])
    equal(actual[1], wanted[1])
    assert int(actual[0].max()) == count
