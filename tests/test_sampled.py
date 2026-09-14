"""The exact 4x sampled path against cv2.fillPoly + cv2.resize and the reference."""

import os
import platform
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import reference
import ultrafast_maskops as native
from ultrafast_maskops import _native

IDENTITY = bytes(range(16))


def equal(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


def fill(shape, contour):
    full = np.zeros(shape, np.uint8)
    cv2.fillPoly(full, np.asarray([contour.reshape(-1)], dtype=np.int32).reshape(1, -1, 2), color=1)
    return full


def patterns(full):
    return (8 * full[1::4, 1::4] + 4 * full[1::4, 2::4] + 2 * full[2::4, 1::4] + full[2::4, 2::4]).astype(np.uint8)


def sampled_patterns(shape, contours):
    packed = native.PackedPolygons.from_segments(contours)
    return _native.Rasterizer().sampled_masks(packed._native, shape[0], shape[1], IDENTITY)


def ring(rng, count, centre, radius, jitter=0.0):
    theta = np.sort(rng.random(count)) * 2 * np.pi
    r = radius * (1 + jitter * (rng.random(count) - 0.5))
    return np.stack([centre[0] + r * np.cos(theta), centre[1] + r * np.sin(theta)], 1)


def contours_for(rng, case, h, w):
    """One family of contours per case; all are exercised at several sizes."""
    kind = case % 9
    if kind == 0:  # random vertices around and beyond the image
        return [rng.uniform(-0.3, 1.3, (int(rng.integers(1, 40)), 2)) * (w, h) for _ in range(6)]
    if kind == 1:  # dense resampled outlines, as after resample_segments
        return [ring(rng, 1000, rng.random(2) * (w, h), rng.random() * max(h, w) * 0.6 + 1, 0.4) for _ in range(6)]
    if kind == 2:  # self-intersecting stars
        out = []
        for _ in range(6):
            k = int(rng.integers(3, 25))
            theta = rng.permutation(k) * 2 * np.pi / k
            c, r = rng.random(2) * (w, h), rng.random() * max(h, w)
            out.append(np.stack([c[0] + r * np.cos(theta), c[1] + r * np.sin(theta)], 1))
        return out
    if kind == 3:  # axis-aligned edges on and just outside the borders
        xs = np.array([-1, 0, 1, 2, 3, w - 3, w - 2, w - 1, w, w + 1])
        ys = np.array([-1, 0, 1, 2, 3, h - 3, h - 2, h - 1, h, h + 1])
        return [np.stack([rng.choice(xs, 4), rng.choice(ys, 4)], 1)[[0, 1, 1, 0]] * [1, 1] for _ in range(6)] + [
            np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            for x0, x1, y0, y1 in rng.choice(np.r_[xs, ys], (6, 4))
        ]
    if kind == 4:  # degenerate: points, segments, repeated and collinear vertices
        p = rng.uniform(-2, max(h, w) + 2, (6, 2))
        return [
            p[:1],
            p[:2],
            np.repeat(p[:1], 5, 0),
            np.stack([np.linspace(p[0, 0], p[1, 0], 7), np.linspace(p[0, 1], p[1, 1], 7)], 1),
            np.r_[p[:3], p[:3][::-1]],
            np.r_[p[:1], p[:1] + 0.5, p[:1]],
        ]
    if kind == 5:  # thin slivers and steep lines
        out = []
        for _ in range(6):
            a, b = rng.uniform(-10, max(h, w) + 10, (2, 2))
            d = rng.uniform(0, 3, 2)
            out.append(np.array([a, b, b + d, a + d]))
        return out
    if kind == 6:  # far outside, lines crossing the whole image
        big = 4 * max(h, w)
        return [rng.uniform(-big, big, (int(rng.integers(3, 8)), 2)) for _ in range(6)]
    if kind == 7:  # near the envelope; vertices within |x| < 2**24
        lim = 2**24 - 1
        return [np.array([[-lim, -lim], [lim, -3], [5, lim], [rng.integers(0, w), rng.integers(0, h)]]) for _ in range(3)]
    # kind 8: nested and overlapping rectangles with shared edges
    out = []
    for _ in range(6):
        x0, y0 = rng.integers(-4, w, 2) if w > 4 else (0, 0)
        x1, y1 = x0 + rng.integers(0, w + 4), y0 + rng.integers(0, h + 4)
        out.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64))
    return out


SHAPES = [(4, 4), (8, 12), (12, 32), (28, 36), (32, 32), (64, 48), (100, 132), (160, 160), (480, 640), (640, 640)]


@pytest.mark.parametrize("shape", SHAPES)
def test_sampled_fill_matches_opencv_fill_poly(shape):
    rng = np.random.default_rng(shape[0] * 1000 + shape[1])
    h, w = shape
    for case in range(90):
        contours = [np.asarray(c, np.float64) for c in contours_for(rng, case, h, w)]
        got = sampled_patterns(shape, contours)
        for i, c in enumerate(contours):
            expected = patterns(fill(shape, c))
            if not np.array_equal(got[i], expected):
                bad = np.argwhere(got[i] != expected)[:5].tolist()
                pytest.fail(f"case {case} contour {i}: patterns differ at {bad} for {c.tolist()[:8]}")


def test_every_small_polygon_on_a_small_grid():
    # All triangles with vertices on a 5x5 lattice straddling a 8x8 image corner.
    lattice = [(x, y) for x in (-2, 0, 1, 2, 5) for y in (-1, 1, 2, 6, 9)]
    shape = (8, 8)
    contours = [np.array([a, b, c]) for a in lattice for b in lattice for c in lattice[::3]]
    got = sampled_patterns(shape, contours)
    expected = np.array([patterns(fill(shape, c)) for c in contours])
    equal(got, expected)


@pytest.mark.parametrize("shape", [(4, 8), (8, 8), (16, 28), (32, 32), (28, 36), (480, 640), (640, 640)])
def test_calibrated_table_reproduces_cv2_resize(shape):
    table = native._sampled_table(shape[0], shape[1], 4)
    if table is None:
        # Correct but slow: another cv2.resize build may not be one function
        # of the four samples at every pixel. The reference platform must be.
        assert not (sys.platform == "darwin" and platform.machine() == "arm64")
        pytest.skip("cv2.resize here is not a per-pixel function of the 2x2 samples")
    assert len(table) == 16 and max(table) <= 1
    assert table[0] == 0 and table[15] == 1
    rng = np.random.default_rng(3)
    for _ in range(5):
        full = (rng.random(shape) < rng.random()).astype(np.uint8)
        equal(np.frombuffer(table, np.uint8)[patterns(full)], cv2.resize(full, (shape[1] // 4, shape[0] // 4)))


def test_ineligible_sizes_have_no_table():
    assert native._sampled_table(30, 32, 4) is None
    assert native._sampled_table(32, 32, 2) is None
    assert native._sampled_table(4 << 15, 32, 4) is None


@pytest.mark.parametrize("shape", [(4, 4), (8, 12), (28, 36), (64, 48), (160, 160), (480, 640), (640, 640)])
@pytest.mark.parametrize("mode", ["retained", "bounded"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_overlap_and_masks_match_reference(shape, mode, dtype):
    rng = np.random.default_rng(shape[0] + shape[1] * 7 + (mode == "bounded") + (dtype == np.float32) * 3)
    engine = native.Rasterizer()
    h, w = shape
    for case in range(27):
        contours = contours_for(rng, case, h, w)
        m = max(len(c) for c in contours)
        # (N, M, 2) arrays as in Instances.segments: pad by repeating the last vertex.
        segments = np.stack([np.r_[c, np.repeat(c[-1:], m - len(c), 0)] for c in contours]).astype(dtype)
        expected = reference.polygons2masks_overlap(shape, segments, 4)
        for got in (
            engine.overlap(shape, native.PackedPolygons.from_segments(segments), 4, mode=mode),
            engine.overlap_segments(shape, segments, 4, mode=mode),
        ):
            equal(got[0], expected[0])
            equal(got[1], expected[1])
        expected_masks = reference.polygons2masks(shape, segments, 1, 4)
        equal(engine.masks(shape, native.PackedPolygons.from_segments(segments), 1, 4), expected_masks)
        equal(engine.masks_segments(shape, segments, 1, 4), expected_masks)


@pytest.mark.parametrize("n", [1, 2, 17, 255, 256, 300])
def test_overlap_ranks_ties_and_int32_output(n):
    rng = np.random.default_rng(n)
    shape = (64, 96)
    base = np.array([[3, 5], [60, 7], [58, 50], [4, 44]], np.float32)
    segments = np.stack([base + rng.integers(-8, 30, 2) * (i % 4 != 0) for i in range(n)])  # repeated areas
    for mode in ("retained", "bounded"):
        got = native.Rasterizer().overlap_segments(shape, segments, 4, mode=mode)
        expected = reference.polygons2masks_overlap(shape, segments, 4)
        equal(got[0], expected[0])
        equal(got[1], expected[1])


def test_arrays_outside_the_sampled_path_keep_reference_results_and_errors():
    shape = (32, 32)
    engine = native.Rasterizer()
    square = np.array([[[2, 2], [29, 3], [27, 30], [1, 28]]], np.float32)
    cases = [
        square.astype(np.int64),  # integer dtype
        np.asfortranarray(np.repeat(square, 3, 0)),  # not C-contiguous
        square.astype(">f4"),  # non-native byte order
        np.array([[[2, 2], [2**25, 3], [27, 30]]], np.float64),  # outside the exact envelope
        np.repeat(square, 2, 0)[:, ::-1],  # negative strides
    ]
    for segments in cases:
        expected = reference.polygons2masks_overlap(shape, segments, 4)
        got = engine.overlap_segments(shape, segments, 4)
        equal(got[0], expected[0])
        equal(got[1], expected[1])
        equal(engine.masks_segments(shape, segments, 1, 4), reference.polygons2masks(shape, segments, 1, 4))
    for value in (np.nan, np.inf, -np.inf):
        bad = square.copy()
        bad[0, 1, 0] = value
        with pytest.raises(ValueError, match="finite"):
            engine.overlap_segments(shape, bad, 4)
        with pytest.raises(ValueError, match="finite"):
            engine.masks_segments(shape, bad, 1, 4)


def test_sampled_state_contract():
    core = _native.Rasterizer()
    table = native._sampled_table(32, 32, 4)
    with pytest.raises(ValueError, match="preceding sampled_raster"):
        core.sampled_compose(np.zeros(0, np.int64))
    square = np.array([[[2, 2], [29, 3], [27, 30]]], np.float32)
    areas = core.sampled_raster(square, 32, 32, table, True)
    with pytest.raises(ValueError, match="same contours"):
        core.sampled_compose(np.zeros(2, np.int64) + [0, 1])
    with pytest.raises(ValueError, match="permutation"):
        core.sampled_compose(np.array([1], np.int64))
    assert core.sampled_compose(np.argsort(-areas)).dtype == np.uint8
    with pytest.raises(ValueError, match="binary"):
        core.sampled_raster(square, 32, 32, IDENTITY, True)
    with pytest.raises(ValueError, match="divisible by 4"):
        core.sampled_masks(square, 30, 32, table)


FIXTURE = Path(__file__).parent / "fixtures" / "coco-format-inputs.npz"


def captured_calls(path):
    data = np.load(path)
    points, offsets, counts, verts, hw, ratio = (data[k] for k in ("points", "offsets", "instances", "vertices", "hw", "ratio"))
    for i in range(len(counts)):
        n, v = int(counts[i]), int(verts[i])
        if n:
            yield points[offsets[i] : offsets[i + 1]].reshape(n, v, 2), (int(hw[i][0]), int(hw[i][1])), int(ratio[i])


@pytest.mark.parametrize(
    "path, minimum",
    [
        pytest.param(FIXTURE, 40, id="fixture"),
        pytest.param(
            os.environ.get("MASKOPS_FORMAT_INPUTS"),
            1000,
            id="full-capture",
            marks=pytest.mark.skipif(
                not os.environ.get("MASKOPS_FORMAT_INPUTS"), reason="set MASKOPS_FORMAT_INPUTS to the full capture"
            ),
        ),
    ],
)
def test_captured_format_inputs(path, minimum):
    # Real augmented COCO polygons as Format receives them (float32 (N, 1000, 2)).
    engine = native.Rasterizer()
    checked = 0
    for segments, shape, r in captured_calls(path):
        expected = reference.polygons2masks_overlap(shape, segments, r)
        got = engine.overlap_segments(shape, segments, r)
        equal(got[0], expected[0])
        equal(got[1], expected[1])
        equal(engine.masks_segments(shape, segments, 1, r), reference.polygons2masks(shape, segments, 1, r))
        checked += 1
    assert checked >= minimum


def test_simd_switch_selects_the_scalar_twins():
    # The scalar twins compute the same bytes; the CI runs the parity tests
    # under ULTRAFAST_MASKOPS_SCALAR=1 as well. Here: the switch is honoured.
    code = "import ultrafast_maskops as m; print(m.backend_info()['simd'])"
    modes = {}
    for value in ("", "1"):
        env = {k: v for k, v in os.environ.items() if k != "ULTRAFAST_MASKOPS_SCALAR"}
        if value:
            env["ULTRAFAST_MASKOPS_SCALAR"] = value
        modes[value] = subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True)
        modes[value] = modes[value].stdout.strip()
    assert modes["1"] == "scalar"
    assert modes[""] in ("neon", "sse2", "scalar")
    if platform.machine().lower() in ("arm64", "aarch64", "x86_64", "amd64"):
        assert modes[""] != "scalar"
