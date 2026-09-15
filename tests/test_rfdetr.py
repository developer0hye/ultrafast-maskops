"""RF-DETR path: the deferred pycocotools fill equals RF-DETR's own masks.

Two layers. The kernel is checked against pycocotools + torchvision on random
polygons and random resize/crop/flip chains. The adapter is checked against an
unmodified RF-DETR CocoDetection on a synthetic COCO dataset: with the same
seed both produce byte-identical images, boxes, labels and masks.
"""

import json
import random

import numpy as np
import pytest

coco_mask = pytest.importorskip("pycocotools.mask")

from ultrafast_maskops import _native  # noqa: E402
from ultrafast_maskops._chain import index_maps  # noqa: E402


def reference_masks(instances, h, w):
    """RF-DETR's convert_coco_poly_to_mask, restricted to polygon lists."""
    planes = []
    for polygons in instances:
        if len(polygons) == 0:
            planes.append(np.zeros((h, w), dtype=bool))
            continue
        decoded = coco_mask.decode(coco_mask.frPyObjects([list(map(float, p)) for p in polygons], h, w))
        if decoded.ndim < 3:
            decoded = decoded[..., None]
        planes.append(decoded.any(axis=2))
    return np.stack(planes) if planes else np.zeros((0, h, w), dtype=bool)


def apply_chain(masks, ops):
    """Applies traced operations to a bool (N, H, W) tensor the way RF-DETR's transforms do."""
    import torch
    from torchvision import tv_tensors
    from torchvision.transforms.v2 import functional as tv_functional

    out = tv_tensors.Mask(torch.from_numpy(masks), dtype=torch.bool)
    for op in ops:
        if op[0] == "resize":
            out = tv_functional.resize(out, list(op[2]))
        elif op[0] == "crop":
            out = tv_functional.crop(out, top=op[1], left=op[2], height=op[3], width=op[4])
        elif op[0] == "hflip":
            out = tv_functional.horizontal_flip(out)
    return out.as_subclass(torch.Tensor).numpy()


def random_polygon(rng, h, w):
    kind = rng.random()
    n = rng.randint(3, 12)
    if kind < 0.6:  # a star-shaped polygon inside the image
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        pts = []
        for i in range(n):
            angle = 2 * np.pi * i / n + rng.uniform(-0.3, 0.3)
            radius = rng.uniform(0.5, 0.5 * max(h, w))
            pts += [cx + radius * np.cos(angle), cy + radius * np.sin(angle)]
        return pts
    if kind < 0.8:  # arbitrary, self-intersecting, partly outside
        return [rng.uniform(-0.3 * w, 1.3 * w) if i % 2 == 0 else rng.uniform(-0.3 * h, 1.3 * h) for i in range(2 * n)]
    if kind < 0.9:  # integer and half-integer vertices, repeated points, tiny
        base = [rng.choice([0, 0.5, 1, 1.5, 2, 2.5, 3]) + rng.randint(0, max(1, w - 4)) for _ in range(n)]
        pts = []
        for i in range(n):
            x, y = base[i], rng.choice([0, 0.5, 1]) + rng.randint(0, max(1, h - 2))
            pts += [x, y] * rng.choice([1, 1, 2])
        return pts
    return [rng.uniform(0, w), rng.uniform(0, h)] * rng.randint(1, 3)  # degenerate


def random_ops(rng, h, w):
    ops = []
    if rng.random() < 0.5:
        size = (rng.randint(8, 3 * h), rng.randint(8, 3 * w))
        ops.append(("resize", (h, w), size))
        h, w = size
    if rng.random() < 0.6:
        ch, cw = rng.randint(1, h), rng.randint(1, w)
        top, left = rng.randint(0, h - ch), rng.randint(0, w - cw)
        ops.append(("crop", top, left, ch, cw))
        h, w = ch, cw
        size = (rng.randint(8, 2 * h + 8), rng.randint(8, 2 * w + 8))
        ops.append(("resize", (h, w), size))
        h, w = size
    if rng.random() < 0.5:
        ops.append(("hflip",))
    return ops


@pytest.mark.parametrize("seed", range(40))
def test_kernel_matches_pycocotools_through_random_chains(seed):
    pytest.importorskip("torchvision")
    rng = random.Random(seed)
    h, w = rng.randint(4, 90), rng.randint(4, 90)
    instances = [[random_polygon(rng, h, w) for _ in range(rng.randint(0, 3))] for _ in range(rng.randint(0, 5))]
    # pycocotools reads a 4-number first entry as a box; later short entries are polygons.
    instances = [polys for polys in instances if not polys or len(polys[0]) > 4]
    expected = reference_masks(instances, h, w)
    for _ in range(6):
        ops = random_ops(rng, h, w)
        rows, cols = index_maps(ops, h, w)
        got = _native.rfdetr_masks([[np.asarray(p, dtype=np.float64) for p in polys] for polys in instances], h, w, rows, cols)
        want = apply_chain(expected, ops)
        assert got.dtype == np.bool_ and got.shape == want.shape, ops
        np.testing.assert_array_equal(got, want, err_msg=f"ops={ops}")


def test_kernel_identity_chain_is_the_full_raster():
    rng = random.Random(1234)
    h, w = 128, 160
    instances = [[random_polygon(rng, h, w) for _ in range(2)] for _ in range(20)]
    instances = [polys for polys in instances if len(polys[0]) > 4]
    rows, cols = index_maps([], h, w)
    got = _native.rfdetr_masks([[np.asarray(p) for p in polys] for polys in instances], h, w, rows, cols)
    np.testing.assert_array_equal(got, reference_masks(instances, h, w))


def test_kernel_handles_padding_and_empty_maps():
    rows = np.array([-1, -1, 0, 0, 1, -1], dtype=np.int32)
    cols = np.array([-1, 1, 0, -1], dtype=np.int32)  # flipped, padded both sides
    poly = [np.array([0.0, 0.0, 3.0, 0.0, 3.0, 3.0, 0.0, 3.0])]
    got = _native.rfdetr_masks([poly], 3, 3, rows, cols)
    assert got.shape == (1, 6, 4)
    assert got[0].tolist() == [[0] * 4, [0] * 4, [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0] * 4]
    assert _native.rfdetr_masks([], 3, 3, rows, cols).shape == (0, 6, 4)
    with pytest.raises(ValueError):
        _native.rfdetr_masks([poly], 3, 3, np.array([1, 0], dtype=np.int32), cols)


# --- end to end against RF-DETR --------------------------------------------
# Pinned to RF-DETR develop @ 2398ce3c (1.11.0.dev0); other sources are rejected
# by check_profile, which these tests report as a failure, not a skip.


def write_synthetic_coco(root, rng, images=6):
    """A tiny COCO dataset: random images, polygon/RLE/crowd annotations."""
    from PIL import Image

    (root / "images").mkdir()
    records, annotations = [], []
    ann_id = 1
    for image_id in range(1, images + 1):
        h, w = rng.randint(300, 700), rng.randint(300, 700)  # above the 384 px crop floor
        Image.fromarray(np.random.RandomState(image_id).randint(0, 255, (h, w, 3), dtype=np.uint8)).save(
            root / "images" / f"{image_id}.jpg"
        )
        records.append({"id": image_id, "file_name": f"{image_id}.jpg", "height": h, "width": w})
        for _ in range(rng.randint(0, 6)):
            polygons = [random_polygon(rng, h, w) for _ in range(rng.randint(1, 3))]
            polygons = [p for p in polygons if len(p) > 4]
            if not polygons:
                continue
            xs, ys = [], []
            for p in polygons:
                xs += p[0::2]
                ys += p[1::2]
            x0, y0 = max(min(xs), 0), max(min(ys), 0)
            x1, y1 = min(max(xs), w), min(max(ys), h)
            if rng.random() < 0.1:  # a degenerate box ConvertCoco drops
                x1 = x0
            entry = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": rng.choice([1, 3, 7]),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": (x1 - x0) * (y1 - y0),
                "iscrowd": 1 if rng.random() < 0.1 else 0,
                "segmentation": polygons,
            }
            if rng.random() < 0.1:  # RLE: takes RF-DETR's own path
                rle = coco_mask.merge(coco_mask.frPyObjects(polygons, h, w))
                entry["segmentation"] = {"size": rle["size"], "counts": rle["counts"].decode()}
            annotations.append(entry)
            ann_id += 1
    categories = [{"id": i, "name": f"c{i}"} for i in (1, 3, 7)]
    (root / "annotations.json").write_text(
        json.dumps({"images": records, "annotations": annotations, "categories": categories})
    )


def build(root, resolution, square):
    from rfdetr.datasets import coco as rf_coco

    if square:
        transforms = rf_coco.make_coco_transforms_square_div_64("train", resolution, multi_scale=True)
    else:
        transforms = rf_coco.make_coco_transforms("train", resolution)
    return rf_coco.CocoDetection(
        root / "images", root / "annotations.json", transforms, include_masks=True, remap_category_ids=True
    )


@pytest.mark.parametrize("square", [True, False])
def test_accelerated_dataset_matches_rfdetr(tmp_path, square):
    pytest.importorskip("rfdetr")
    import torch

    from ultrafast_maskops.rfdetr import accelerate_dataset

    rng = random.Random(7 if square else 8)
    write_synthetic_coco(tmp_path, rng)
    reference = build(tmp_path, 96, square)
    accelerated = accelerate_dataset(build(tmp_path, 96, square))
    compared = 0
    for index in range(len(reference)):
        for seed in range(12):
            torch.manual_seed(seed)
            image_a, target_a = reference[index]
            torch.manual_seed(seed)
            image_b, target_b = accelerated[index]
            assert torch.equal(image_a, image_b)
            assert set(target_a) == set(target_b), (set(target_a) ^ set(target_b))
            for key in target_a:
                assert torch.equal(target_a[key], target_b[key]), (index, seed, key)
            assert target_b["masks"].dtype == torch.bool
            compared += 1
    assert compared == 12 * len(reference)


def test_accelerate_rejects_unknown_transforms(tmp_path):
    pytest.importorskip("rfdetr")
    from rfdetr.datasets import coco as rf_coco

    from ultrafast_maskops.rfdetr import accelerate_dataset

    write_synthetic_coco(tmp_path, random.Random(3), images=1)
    dataset = build(tmp_path, 64, True)
    dataset._transforms.transforms.append(lambda image, target: (image, target))
    with pytest.raises(RuntimeError, match="unsupported transform"):
        accelerate_dataset(dataset)
    dataset = rf_coco.CocoDetection(tmp_path / "images", tmp_path / "annotations.json", None, include_masks=False)
    with pytest.raises(ValueError, match="without masks"):
        accelerate_dataset(dataset)
