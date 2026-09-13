"""Profile-guarded CPU batch collation into owned shared storage.

The key handling follows Ultralytics YOLODataset.collate_fn (AGPL-3.0), pinned
below. Direct allocation follows PyTorch's default tensor collation pattern.
"""

import hashlib
import inspect
import os
from functools import lru_cache

import torch
from ultralytics.data.dataset import YOLODataset

_REFERENCE_COLLATE = YOLODataset.collate_fn
_COLLATE_SHA256 = "124bfcaa17398abe7749d85f6a63d0303bedbc1400184005ac772621f6c9a420"


def _check_reference_profile():
    if torch.__version__.split("+")[0] != "2.10.0":
        raise RuntimeError("shared collation currently requires PyTorch 2.10.0")
    if YOLODataset.collate_fn is not _REFERENCE_COLLATE:
        raise RuntimeError("Ultralytics collate_fn was replaced; retain the original collator")
    if hashlib.sha256(inspect.getsource(_REFERENCE_COLLATE).encode()).hexdigest() != _COLLATE_SHA256:
        raise RuntimeError("unsupported Ultralytics collate_fn source")


@lru_cache(maxsize=1)
def _check_process_profile(pid):
    _check_reference_profile()


def check_shared_collate_profile(dataset):
    _check_reference_profile()
    collator = getattr(dataset, "collate_fn", None)
    if not isinstance(dataset, YOLODataset) or (
        collator is not _REFERENCE_COLLATE and collator is not shared_collate_fn
    ):
        raise TypeError("custom collators require their own shared-batch integration")


def _combine(values, *, stack):
    operation = torch.stack if stack else torch.cat
    first = values[0]
    direct = type(first) is torch.Tensor and all(
        type(v) is torch.Tensor
        and v.device.type == "cpu"
        and v.layout == torch.strided
        and not v.is_nested
        and not v.requires_grad
        and v.dtype == first.dtype
        and v.is_contiguous()
        and not (v.ndim == 4 and v.is_contiguous(memory_format=torch.channels_last))
        and not (v.ndim == 5 and v.is_contiguous(memory_format=torch.channels_last_3d))
        for v in values
    )
    if direct:
        direct = (
            all(v.shape == first.shape for v in values)
            if stack
            else (first.ndim > 0 and all(v.ndim == first.ndim and v.shape[1:] == first.shape[1:] for v in values))
        )
    if not direct:
        # Preserve promotion, unusual layouts, autograd and PyTorch's errors.
        # The result is shared on this worker thread before queue publication.
        return operation(values, 0)
    shape = (len(values), *first.shape) if stack else (sum(v.shape[0] for v in values), *first.shape[1:])
    storage = first._typed_storage()._new_shared(sum(v.numel() for v in values), device=first.device)
    out = first.new(storage).resize_(shape)
    return operation(values, 0, out=out)


def _prepare_shared(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.layout != torch.strided or value.is_nested:
            raise TypeError("shared collation supports dense strided CPU tensors")
        value.share_memory_()
    elif isinstance(value, dict):
        for child in value.values():
            _prepare_shared(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _prepare_shared(child)


def shared_collate_fn(batch):
    """Preserve the pinned batch contract; prepare worker tensors before IPC.

    Common contiguous homogeneous stack/cat fields are written directly into
    shared storage. Other supported tensors use reference operations and eager
    sharing. There is no reusable pool: each returned batch owns its storage.
    Outside a DataLoader worker, call the original collator unchanged.
    """
    _check_process_profile(os.getpid())
    if torch.utils.data.get_worker_info() is None:
        return _REFERENCE_COLLATE(batch)
    new_batch = {}
    batch = [dict(sorted(b.items())) for b in batch]
    keys = batch[0].keys()
    values = list(zip(*[list(b.values()) for b in batch]))
    for i, key in enumerate(keys):
        value = values[i]
        if key in {"img", "text_feats", "semantic_mask", "sem_masks", "depth"}:
            value = _combine(value, stack=True)
        elif key == "visuals":
            value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
        if key in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
            value = _combine(value, stack=False)
        new_batch[key] = value
    if "batch_idx" in new_batch:
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i
        new_batch["batch_idx"] = _combine(new_batch["batch_idx"], stack=False)
    _prepare_shared(new_batch)
    return new_batch


def share_dataset_batches(dataset):
    """Install this instance's guarded collator before constructing workers.

    Return 1 when installed, or 0 if already installed. No module/class globals
    or trainer/worker shutdown methods are changed.
    """
    check_shared_collate_profile(dataset)
    if dataset.collate_fn is shared_collate_fn:
        return 0
    dataset.collate_fn = shared_collate_fn
    return 1
