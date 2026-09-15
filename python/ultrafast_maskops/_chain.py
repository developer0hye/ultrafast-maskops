"""Index maps of a chain of torchvision nearest resizes, crops and horizontal flips.

Each operation is a tuple: ("resize", (old_h, old_w), (new_h, new_w)),
("crop", top, left, height, width) or ("hflip",). The maps say which source
row and column every output row and column reads, -1 where a crop padded.
"""

import numpy as np


def nearest(n_in, n_out):
    """Source index read by each output index of a torch 'nearest' resize.

    torch computes floor(i * scale) with scale = float32(n_in) / float32(n_out)
    and the product in float32, clamped to n_in - 1; equal sizes are identity.
    """
    if n_in == n_out:
        return np.arange(n_out, dtype=np.int64)
    scale = np.float32(n_in) / np.float32(n_out)
    index = np.floor(np.arange(n_out, dtype=np.float32) * scale).astype(np.int64)
    return np.minimum(index, n_in - 1)


def crop(index, start, length):
    out = np.full(length, -1, dtype=np.int64)
    lo, hi = max(start, 0), min(start + length, len(index))
    if hi > lo:
        out[lo - start : hi - start] = index[lo:hi]
    return out


def index_maps(ops, h, w):
    """Output row -> source row and output column -> source column after ops (int32)."""
    rows, cols = np.arange(h, dtype=np.int64), np.arange(w, dtype=np.int64)
    for op in ops:
        if op[0] == "resize":
            (old_h, old_w), (new_h, new_w) = op[1], op[2]
            if (old_h, old_w) != (len(rows), len(cols)):
                raise RuntimeError("resize trace does not match the canvas")
            rows, cols = rows[nearest(old_h, new_h)], cols[nearest(old_w, new_w)]
        elif op[0] == "crop":
            _, top, left, height, width = op
            rows, cols = crop(rows, top, height), crop(cols, left, width)
        elif op[0] == "hflip":
            cols = cols[::-1]
        else:
            raise RuntimeError(f"unknown traced operation {op[0]!r}")
    return rows.astype(np.int32), np.ascontiguousarray(cols, dtype=np.int32)
