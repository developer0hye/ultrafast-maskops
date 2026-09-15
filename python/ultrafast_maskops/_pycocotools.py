"""Which double arithmetic the installed pycocotools was compiled with.

rleFrPoly computes ``scale * v + .5`` and ``ys + s * t + .5`` in double. Built
as written (the x86-64 wheels), every product is rounded before the addition;
built with floating-point contraction (the arm64 wheels), each pair is one
fused multiply-add with a single rounding. The two agree on almost every
polygon and differ by a pixel on some; the native kernel implements both, and
this module picks the one that reproduces the installed pycocotools on a set
of polygons known to separate them. Every probe is a case found by comparing
the two flavours of the kernel, and each flavour matched a real pycocotools
build on it: unfused on x86-64 Linux, fused on arm64 macOS.
"""

import numpy as np

from . import _native

# Polygons on a 64 x 64 canvas whose masks differ between the two flavours.
_PROBES = [
    [47.61, 1.7, 50.7, 55.08, 47.2, 43.6, 7.699, 38.866],
    [11.0, 1.4, 60.74, 3.062, 21.131, 35.943],
    [2.48, 55.23, 20.88, 2.31, 5.6, 29.217, 25.1, 49.412, 39.02, 3.59],
    [15.8, 23.4, 62.1, 2.2, 60.1, 40.1],
    [21.6, 23.2, 22.504, 4.1, 3.64, 40.1],
    [16.72, 43.651, 54.14, 0.186, 31.7, 17.3, 8.6, 11.582, 20.83, 20.7],
    [53.6, 47.152, 12.19, 29.73, 3.4, 56.9, 50.907, 28.9, 43.6, 7.9],
    [17.52, 34.62, 61.524, 5.96, 33.8, 10.8],
    [19.6, 56.1, 39.5, 54.848, 54.42, 27.9, 0.648, 55.858, 37.087, 16.577],
    [51.452, 61.67, 34.415, 8.499, 12.807, 0.12, 27.507, 17.77, 9.64, 61.7],
    [35.76, 0.909, 4.9, 37.88, 31.04, 1.02, 12.07, 26.899],
    [7.3, 30.643, 28.15, 19.9, 27.949, 4.64, 2.24, 59.8, 55.37, 42.69],
]
_SIDE = 64
_cached = False
_flavour = None


def probe_masks(fused):
    """The kernel's masks of the probes on the identity chain, one per probe."""
    index = np.arange(_SIDE, dtype=np.int32)
    return _native.rfdetr_masks([[np.asarray(p)] for p in _PROBES], _SIDE, _SIDE, index, index, fused)


def reference_masks():
    """pycocotools' masks of the probes."""
    import pycocotools.mask as coco_mask

    out = np.empty((len(_PROBES), _SIDE, _SIDE), dtype=bool)
    for i, p in enumerate(_PROBES):
        out[i] = coco_mask.decode(coco_mask.frPyObjects([p], _SIDE, _SIDE))[:, :, 0].astype(bool)
    return out


def arithmetic():
    """'unfused' or 'fused', whichever reproduces the installed pycocotools; None if neither.

    Computed once per process (DataLoader workers started by spawn redo it,
    which takes about a millisecond). The probes must separate the flavours
    and exactly one flavour must match every probe; anything else is None.
    """
    global _cached, _flavour
    if _cached:
        return _flavour
    unfused, fused = probe_masks(False), probe_masks(True)
    reference = reference_masks()
    separated = all(not np.array_equal(unfused[i], fused[i]) for i in range(len(_PROBES)))
    matches = {"unfused": np.array_equal(unfused, reference), "fused": np.array_equal(fused, reference)}
    if separated and sum(matches.values()) == 1:
        _flavour = "unfused" if matches["unfused"] else "fused"
    else:
        _flavour = None
    _cached = True
    return _flavour


def fused():
    """True when the installed pycocotools uses contracted arithmetic; raises if it is neither flavour."""
    flavour = arithmetic()
    if flavour is None:
        raise RuntimeError("the installed pycocotools matches neither double-arithmetic flavour of the kernel")
    return flavour == "fused"
