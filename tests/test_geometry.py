"""Byte parity of the native segment geometry with the pinned Ultralytics source."""

import copy
import pickle
import random

import numpy as np
import pytest
import torch
from PIL import Image
from ultrafast_maskops import _native, geometry
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data import augment, dataset
from ultralytics.utils import ops


def same(a, b):
    assert type(a) is type(b), (type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    elif isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif hasattr(a, "__dict__") and type(a).__module__.startswith("ultralytics"):
        same(vars(a), vars(b))
    else:
        assert a == b


def polygons(rng, count, lengths, low, high, dtype=np.float32):
    return [(rng.random((int(rng.choice(lengths)), 2)) * (high - low) + low).astype(dtype) for _ in range(count)]


def test_calibration_matches_installed_numpy():
    assert geometry.interp_fused() in (True, False)


@pytest.mark.parametrize("n", [100, 1000, 1001, 1500])
@pytest.mark.parametrize("scale", [(0, 1), (-40, 700), (1e-7, 3e-7), (3e4, 3e5)])
def test_resample_matches_reference(n, scale):
    rng = np.random.default_rng(n + int(scale[1] * 10))
    lengths = [3, 4, 5, 10, 99, 100, 101, 498, 499, 500, 998, 999, 1000, 1001, 1002, 1499, 1500, 2000]
    segments = polygons(rng, 60, lengths, *scale)
    expected = np.stack(ops.resample_segments([s.copy() for s in segments], n=n), axis=0)
    same(geometry.resample_stack(segments, n), expected)


def test_resample_unusual_inputs_use_reference():
    rng = np.random.default_rng(1)
    mixed = [rng.random((5, 2)).astype(np.float64), rng.random((7, 2)).astype(np.float32)]
    same(geometry.resample_stack(mixed, 1000), np.stack(ops.resample_segments([s.copy() for s in mixed], n=1000)))
    repeated = [np.zeros((4, 2), np.float32), np.full((1, 2), 0.5, np.float32)]
    same(geometry.resample_stack(repeated, 50), np.stack(ops.resample_segments([s.copy() for s in repeated], n=50)))


def reference_boxes(segments, w, h, clip):
    segments = segments.copy()
    bboxes = np.stack([augment.segment2box(xy, w, h) for xy in segments], 0)
    if clip:
        segments[..., 0] = segments[..., 0].clip(bboxes[:, 0:1], bboxes[:, 2:3])
        segments[..., 1] = segments[..., 1].clip(bboxes[:, 1:2], bboxes[:, 3:4])
    return bboxes, segments


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("clip", [True, False])
@pytest.mark.parametrize("case", ["inside", "crossing", "outside", "borders", "corners", "degenerate"])
def test_segment_boxes_match_reference(dtype, clip, case):
    rng = np.random.default_rng(
        ["inside", "crossing", "outside", "borders", "corners", "degenerate"].index(case) * 2 + clip
    )
    w, h = 640, 480
    n, m = 64, 97
    if case == "inside":
        seg = rng.random((n, m, 2)) * (w, h)
    elif case == "crossing":
        seg = rng.random((n, m, 2)) * (w + 400, h + 400) - 200
    elif case == "outside":
        seg = rng.random((n, m, 2)) * 100 + (w + 10, -300)
    elif case == "borders":
        seg = np.round(rng.random((n, m, 2)) * (w + 40, h + 40) - 20) / 1.0
        seg[:, ::3, 0] = rng.choice([0, w], (n, len(range(0, m, 3))))
    elif case == "corners":
        # Large polygons enclosing corners, plus vertices exactly at corners.
        theta = np.linspace(0, 2 * np.pi, m, endpoint=False)
        centre = rng.random((n, 1, 2)) * (w, h)
        radius = rng.random((n, 1, 1)) * 900 + 50
        seg = centre + radius * np.stack([np.cos(theta), np.sin(theta)], -1)
        seg[: n // 2, 0] = (0, 0)
        seg[n // 2 :, 1] = (w, h)
    else:
        seg = np.repeat(rng.random((n, 1, 2)) * (w + 200, h + 200) - 100, m, axis=1)
        seg[::2, ::2] += 1e-3
    seg = np.ascontiguousarray(seg.astype(dtype))
    expected_boxes, expected_segments = reference_boxes(seg, w, h, clip)
    got = seg.copy()
    same(geometry.segment_boxes(got, w, h, clip), expected_boxes)
    same(got, expected_segments)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -0.0])
def test_nonfinite_and_negative_zero_batches_use_reference(dtype, value):
    seg = (np.random.default_rng(3).random((6, 20, 2)) * 700 - 30).astype(dtype)
    seg[:, ::4, 0] = 0.0  # positive zeros that the reference may clip to -0.0
    seg[4, 7, 1] = value
    untouched = seg.copy()
    kernel = _native.segment_boxes_f32 if dtype == np.float32 else _native.segment_boxes_f64
    assert kernel(seg, 640, 640, True) is None
    assert seg.tobytes() == untouched.tobytes()  # nothing modified before deferring
    expected_boxes, expected_segments = reference_boxes(seg, 640, 640, True)
    got = seg.copy()
    same(geometry.segment_boxes(got, 640, 640, True), expected_boxes)
    same(got, expected_segments)


def random_matrix(rng, perspective, dtype):
    m = np.eye(3)
    m[:2, :2] = (rng.random((2, 2)) * 0.6 + 0.7) * rng.choice([-1, 1])
    m[:2, 2] = rng.random(2) * 400 - 200
    m[2, :2] = (rng.random(2) * 2 - 1) * perspective
    return m.astype(dtype)


@pytest.mark.parametrize("perspective", [0.0, 0.001])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fast_random_perspective_matches_reference(perspective, dtype):
    rng = np.random.default_rng(7)
    for preserve_obb in (False, True):
        ref = augment.RandomPerspective(degrees=10, translate=0.2, scale=0.9, shear=5, perspective=perspective)
        ref.preserve_obb = preserve_obb
        fast = copy.copy(ref)
        fast.__class__ = geometry.FastRandomPerspective
        for _ in range(20):
            num = int(rng.choice([1000, 1001, 1003]))  # includes BLAS tail sizes
            segments = (rng.random((int(rng.integers(0, 40)), num, 2)) * 1200 - 200).astype(np.float32)
            matrix, size = random_matrix(rng, perspective, dtype), (640, 480)
            same(fast.apply_segments(segments.copy(), matrix, size), ref.apply_segments(segments.copy(), matrix, size))


def corpus(root, count=24):
    images, labels = root / "images", root / "labels"
    images.mkdir()
    labels.mkdir()
    rng = np.random.default_rng(11)
    for i in range(count):
        size = (int(rng.integers(90, 200)), int(rng.integers(90, 200)))
        Image.fromarray(rng.integers(0, 255, (*size[::-1], 3), dtype=np.uint8)).save(images / f"{i}.jpg")
        rows = []
        for _ in range(int(rng.integers(1, 12))):
            k = int(rng.integers(3, 80))
            centre, radius = rng.random(2) * 0.8 + 0.1, rng.random() * 0.3 + 0.02
            theta = np.sort(rng.random(k)) * 2 * np.pi
            pts = np.clip(centre + radius * np.stack([np.cos(theta), np.sin(theta)], 1), 0, 1)
            rows.append(" ".join([str(int(rng.integers(0, 3)))] + [f"{v:.6f}" for v in pts.reshape(-1)]))
        (labels / f"{i}.txt").write_text("\n".join(rows))
    return images


def build(images, **changes):
    hyp = copy.deepcopy(DEFAULT_CFG)
    for k, v in changes.items():
        setattr(hyp, k, v)
    return dataset.YOLODataset(
        img_path=str(images),
        imgsz=160,
        batch_size=4,
        augment=True,
        hyp=hyp,
        task="segment",
        data={"names": {0: "a", 1: "b", 2: "c"}},
    )


def perspectives(root):
    found, pending, seen = [], [root], set()
    while pending:
        node = pending.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, augment.RandomPerspective):
            found.append(node)
        if isinstance(node, augment.Compose):
            pending.extend(node.transforms)
        pending.append(getattr(node, "pre_transform", None))
    return found


def samples(ds, count, seed=5):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return [ds[i % len(ds)] for i in range(count)]


@pytest.mark.parametrize("changes", [{}, {"mixup": 0.5, "copy_paste": 0.5}, {"perspective": 0.0005, "degrees": 30}])
def test_accelerated_augmented_samples_are_identical(tmp_path, changes):
    images = corpus(tmp_path)
    reference = build(images, **changes)
    accelerated = build(images, **changes)
    assert geometry.accelerate_geometry(accelerated, persistent=True) >= 1
    same(samples(accelerated, 48), samples(reference, 48))
    # close_mosaic rebuilds transforms; persistent acceleration survives.
    hyp = copy.deepcopy(DEFAULT_CFG)
    reference.close_mosaic(copy.deepcopy(hyp))
    accelerated.close_mosaic(copy.deepcopy(hyp))
    found = perspectives(accelerated.transforms)
    assert found and all(type(t) is geometry.FastRandomPerspective for t in found)
    same(samples(accelerated, 24, seed=9), samples(reference, 24, seed=9))


def test_accelerated_dataset_pickles_for_spawned_workers(tmp_path):
    ds = build(corpus(tmp_path))
    geometry.accelerate_geometry(ds, persistent=True)
    restored = pickle.loads(pickle.dumps(ds))
    same(samples(restored, 8), samples(ds, 8))


def test_profile_guards_reject_changed_sources(monkeypatch):
    geometry.check_geometry_profile()
    monkeypatch.setattr(geometry, "_SOURCE_HASHES", {**geometry._SOURCE_HASHES, "segment2box": "0" * 64})
    with pytest.raises(RuntimeError, match="segment2box"):
        geometry.check_geometry_profile()
