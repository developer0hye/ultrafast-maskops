"""Hashing checks for the macOS/Linux paired-wheel benchmark helper."""

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

if sys.platform == "win32":
    pytest.skip("paired-wheel harness uses POSIX resource APIs", allow_module_level=True)

SCRIPT = Path(__file__).resolve().parents[1] / "bench/compare_wheels.py"
SPEC = importlib.util.spec_from_file_location("compare_wheels", SCRIPT)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


class NoBytesCopy(np.ndarray):
    def tobytes(self, *args, **kwargs):
        raise AssertionError("output hashing must not copy through tobytes")


@pytest.mark.parametrize("dtype", [np.uint8, np.int32, np.int64])
@pytest.mark.parametrize("shape", [(0,), (0, 8, 8), (2, 3, 4)])
def test_output_hash_matches_bytes_without_copy(dtype, shape):
    array = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    expected = hashlib.sha256(array.tobytes()).hexdigest()
    assert compare.hashes([array.view(NoBytesCopy)]) == [
        {"shape": list(shape), "dtype": str(array.dtype), "sha256": expected}
    ]


def test_noncontiguous_output_is_rejected():
    array = np.arange(12, dtype=np.uint8).reshape(3, 4)[:, ::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        compare.hashes([array])
