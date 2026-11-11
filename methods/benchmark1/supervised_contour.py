"""Bora ROI contour polish for a complete supervised 24-class mask.

Apply after the frozen classifier's argmax, before an H&E display overlay.
The source mask is on a calibrated frame with measured X/Y micrometres per
pixel. Validation labels are not used to select the width or area threshold.
"""
import math

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, label


def polish_supervised_roi(mask: np.ndarray, mpp_xy: tuple[float, float]) -> np.ndarray:
    if mask.dtype != np.uint8 or mask.ndim != 2 or mask.min() < 1 or mask.max() > 24:
        raise ValueError('Expected positive uint8 24-class ROI mask')
    if len(mpp_xy) != 2 or min(mpp_xy) <= 0:
        raise ValueError('Measured X/Y spacing required')
    sigma_x, sigma_y = 1.5 / mpp_xy[0], 1.5 / mpp_xy[1]
    best = np.full(mask.shape, -np.inf, np.float32)
    result = np.zeros(mask.shape, np.uint8)
    for cls in np.unique(mask):
        response = gaussian_filter((mask == cls).astype(np.float32),
                                   sigma=(sigma_y, sigma_x), mode='mirror', truncate=4)
        take = response > best
        best[take], result[take] = response[take], cls
    assert np.all(result > 0)
    eight = np.ones((3, 3), bool)
    threshold = math.ceil(400 / (mpp_xy[0] * mpp_xy[1]))
    for _ in range(8):
        changed = 0
        for cls in np.unique(result):
            cc, count = label(result == cls, eight)
            sizes = np.bincount(cc.ravel(), minlength=count + 1)
            for component in np.flatnonzero((sizes < threshold) & (np.arange(count + 1) > 0)):
                island = cc == component
                ring = binary_dilation(island, eight) & ~island
                neighbors = result[ring]
                neighbors = neighbors[neighbors != cls]
                if len(neighbors) == 0:
                    raise ValueError('Small island has no adjacent class')
                result[island] = np.bincount(neighbors, minlength=25).argmax()
                changed += 1
        if not changed:
            break
    else:
        raise RuntimeError('Area repair did not converge')
    for cls in np.unique(result):
        cc, count = label(result == cls, eight)
        if count and np.bincount(cc.ravel(), minlength=count + 1)[1:].min() < threshold:
            raise AssertionError('Undersized final component')
    return result
