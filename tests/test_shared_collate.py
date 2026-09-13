"""Independent batch parity and ownership for the opt-in shared collator."""

import copy
import pickle
from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace

import pytest
import torch
from ultralytics.data.dataset import YOLODataset

from ultrafast_maskops import _shared_collate as shared


def equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert left.stride() == right.stride()
        assert torch.equal(left, right)
        assert left.requires_grad == right.requires_grad
    elif isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def samples(counts=(0, 2, 3), overlap=False):
    result = []
    for i, count in enumerate(counts):
        result.append(
            {
                "img": torch.arange(3 * 16 * 16, dtype=torch.int64).reshape(3, 16, 16).to(torch.uint8) + i,
                "masks": torch.full((1 if overlap else count, 4, 4), i, dtype=torch.uint8),
                "bboxes": torch.arange(count * 4, dtype=torch.float32).reshape(count, 4),
                "cls": torch.full((count, 1), float(i)),
                "batch_idx": torch.zeros(count),
                "im_file": str(i),
                "ori_shape": (16, 16),
                "ratio_pad": ((1.0, 1.0), (0, 0)),
            }
        )
    return result


@pytest.fixture
def worker(monkeypatch):
    # Isolate value/allocation assertions from transport. Separate round-trip
    # tests and real spawned-loader lifecycle tests exercise the packet below.
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: SimpleNamespace(id=0))
    monkeypatch.setattr(shared, "_encode_batch", lambda batch: batch)


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("counts", [(0, 0), (0, 2, 3), (1,), (9, 1, 7)])
def test_batch_values_order_mutations_and_shared_ownership(worker, overlap, counts):
    batch = samples(counts, overlap)
    expected_input, actual_input = copy.deepcopy(batch), copy.deepcopy(batch)
    expected = YOLODataset.collate_fn(expected_input)
    actual = shared.shared_collate_fn(actual_input)
    equal(expected, actual)
    equal(expected_input, actual_input)  # Preserve batch_idx in-place offsets.
    for key, value in actual.items():
        if isinstance(value, torch.Tensor):
            assert value.is_shared()
            if value.numel():
                assert all(value.data_ptr() != sample[key].data_ptr() for sample in actual_input)
    old = {k: v.clone() for k, v in actual.items() if isinstance(v, torch.Tensor)}
    subsequent = shared.shared_collate_fn(samples(counts, overlap))
    subsequent["img"].zero_()
    for key, value in old.items():
        assert torch.equal(actual[key], value), key  # No reuse while an old batch is alive.


def test_zero_worker_path_is_the_reference_without_shared_allocation(monkeypatch):
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: None)
    original = samples()
    expected = YOLODataset.collate_fn(copy.deepcopy(original))
    actual = shared.shared_collate_fn(copy.deepcopy(original))
    equal(expected, actual)
    assert not actual["img"].is_shared()


def test_direct_fields_are_shared_before_ipc_preparation(worker, monkeypatch):
    original = shared._prepare_shared

    def observe(value):
        if isinstance(value, torch.Tensor):
            assert value.is_shared(), "common fields must not need heap-to-shared copying"
        original(value)

    monkeypatch.setattr(shared, "_prepare_shared", observe)
    shared.shared_collate_fn(samples())


@pytest.mark.parametrize("kind", ["promotion", "noncontiguous", "channels_last", "requires_grad"])
def test_reference_fallback_preserves_unusual_tensor_contract(worker, kind):
    if kind == "promotion":
        batch = [{"masks": torch.ones(2, 3, dtype=d)} for d in (torch.int8, torch.float32)]
    elif kind == "noncontiguous":
        batch = [{"masks": torch.arange(12).reshape(3, 4).T} for _ in range(2)]
    elif kind == "channels_last":
        batch = [{"masks": torch.ones(2, 3, 4, 5).contiguous(memory_format=torch.channels_last)} for _ in range(2)]
    else:
        batch = [{"masks": torch.ones(2, 3, requires_grad=True)} for _ in range(2)]
    expected_batch, actual_batch = copy.deepcopy(batch), copy.deepcopy(batch)
    expected = YOLODataset.collate_fn(expected_batch)
    actual = shared.shared_collate_fn(actual_batch)
    equal(expected, actual)
    assert actual["masks"].is_shared()
    if kind == "requires_grad":
        expected["masks"].sum().backward()
        actual["masks"].sum().backward()
        for a, b in zip(expected_batch, actual_batch):
            assert torch.equal(a["masks"].grad, b["masks"].grad)


def test_other_base_fields_and_nested_metadata(worker):
    batch = []
    for i in range(2):
        batch.append(
            {
                "text_feats": torch.ones(2, 3),
                "semantic_mask": torch.zeros(4, 4),
                "sem_masks": torch.ones(2, 4, 4),
                "depth": torch.ones(4, 4),
                "visuals": torch.ones(i + 1, 3),
                "keypoints": torch.ones(i + 1, 2, 3),
                "segments": torch.ones(i + 1, 4, 2),
                "obb": torch.ones(i + 1, 5),
                "metadata": {"tensor": torch.ones(2), "label": str(i)},
            }
        )
    expected = YOLODataset.collate_fn(copy.deepcopy(batch))
    actual = shared.shared_collate_fn(copy.deepcopy(batch))
    equal(expected, actual)
    assert actual["visuals"].is_shared()
    assert actual["metadata"][0]["tensor"].is_shared()


@pytest.mark.parametrize("key,shapes", [("img", [(3, 4), (4, 3)]), ("masks", [(2, 3), (2, 4)]), ("masks", [(), ()])])
def test_invalid_shapes_retain_reference_error(worker, key, shapes):
    batch = [{key: torch.ones(shape)} for shape in shapes]
    with pytest.raises(RuntimeError) as expected:
        YOLODataset.collate_fn(copy.deepcopy(batch))
    with pytest.raises(type(expected.value)) as actual:
        shared.shared_collate_fn(copy.deepcopy(batch))
    assert str(actual.value) == str(expected.value)


def test_instance_install_is_pickleable_idempotent_and_not_global():
    dataset = object.__new__(YOLODataset)
    untouched = object.__new__(YOLODataset)
    assert shared.share_dataset_batches(dataset) == 1
    assert shared.share_dataset_batches(dataset) == 0
    assert untouched.collate_fn is YOLODataset.collate_fn
    assert pickle.loads(pickle.dumps(dataset.collate_fn)) is shared.shared_collate_fn


def test_unknown_collator_rejected_without_mutation():
    dataset = object.__new__(YOLODataset)

    def custom(batch):
        return batch

    dataset.collate_fn = custom
    with pytest.raises(TypeError, match="custom collators"):
        shared.share_dataset_batches(dataset)
    assert dataset.collate_fn is custom


def test_unknown_source_profile_rejected_without_mutation(monkeypatch):
    dataset = object.__new__(YOLODataset)
    monkeypatch.setattr(shared, "_COLLATE_SHA256", "unknown")
    with pytest.raises(RuntimeError, match="collate_fn source"):
        shared.share_dataset_batches(dataset)
    assert dataset.collate_fn is YOLODataset.collate_fn


@pytest.mark.parametrize("kind", ["sparse", "meta"])
def test_unsupported_tensor_outputs_require_explicit_integration(worker, kind):
    tensor = torch.eye(2).to_sparse() if kind == "sparse" else torch.ones(2, device="meta")
    with pytest.raises(TypeError, match="dense strided CPU tensors"):
        shared.shared_collate_fn([{"metadata": tensor}])


@pytest.mark.parametrize("overlap", [False, True])
def test_real_packet_round_trip_preserves_reference_batch(monkeypatch, overlap):
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: SimpleNamespace(id=0))
    batch = samples(overlap=overlap)
    expected = YOLODataset.collate_fn(copy.deepcopy(batch))
    packet = shared.shared_collate_fn(batch)
    assert type(packet.payload) is bytes
    decoded = ForkingPickler.loads(ForkingPickler.dumps(packet))
    equal(expected, decoded)
    assert decoded["img"].is_shared()


def test_packet_contains_handles_not_a_copy_of_tensor_data():
    tensor = torch.zeros(3, 512, 512, dtype=torch.uint8).share_memory_()
    packet = shared._encode_batch({"img": tensor})
    assert type(packet.payload) is bytes
    assert len(packet.payload) < tensor.numel() // 8
    tensor.fill_(17)
    decoded = ForkingPickler.loads(ForkingPickler.dumps(packet))
    assert torch.equal(decoded["img"], tensor)
    decoded["img"].fill_(29)
    assert (tensor == 29).all()  # The payload carries a shared handle, not pixels.
