"""Exact 4-um sampling of the fixed 512/384-overlap binary predictor.

The reference ROI implementation sums two class logits at every 0.5-um
pixel and takes argmax. We retain logits only at the prespecified 4-um pixel
centres. Since every class at a sampled pixel has the same patch count,
argmax of sums is exactly the sampled ROI prediction.
"""

import numpy as np

PATCH = 512
STRIDE = 384
SCALE = 8
CENTRE = 4


def starts(length):
    if length <= PATCH:
        return [0]
    positions = list(range(0, length - PATCH + 1, STRIDE))
    if positions[-1] != length - PATCH:
        positions.append(length - PATCH)
    return positions


def sampled_axis(length):
    if length % SCALE:
        raise ValueError('Calibrated dimensions must be multiples of eight')
    return np.arange(CENTRE, length, SCALE, dtype=np.int64)


def accumulate(scores, counts, logits, top, left, samples_y, samples_x):
    """Add one patch's logits at every 4-um centre it covers.

    `scores` is `[2, H/8, W/8]`, `counts` is `[H/8, W/8]`, and
    `logits` is `[2, 512, 512]`. Works with RAM arrays or memmaps.
    """
    if logits.shape != (2, PATCH, PATCH):
        raise ValueError('Two-output network patch shape differs')
    ay = int(np.searchsorted(samples_y, top, side='left'))
    by = int(np.searchsorted(samples_y, top + PATCH, side='left'))
    ax = int(np.searchsorted(samples_x, left, side='left'))
    bx = int(np.searchsorted(samples_x, left + PATCH, side='left'))
    if ay == by or ax == bx:
        raise ValueError('Patch covers no comparison pixels')
    ly = samples_y[ay:by] - top
    lx = samples_x[ax:bx] - left
    scores[:, ay:by, ax:bx] += logits[:, ly[:, None], lx[None, :]]
    counts[ay:by, ax:bx] += 1


def prediction(scores, counts):
    if np.any(counts == 0):
        raise RuntimeError('Network patch grid left unsampled pixels')
    return (np.argmax(scores, axis=0) + 1).astype(np.uint8)
