"""Physical 6 µm contour smoothing for registered supervised WSI web views.

The frozen Virchow2–PLS score grid has shape [24, token_y, token_x].  A halo
of more than four Gaussian standard deviations is rendered around each view;
the final Bora support is sampled from a 4 µm raster of its polygon union.
This changes the web overlay only, not the saved score grid or whole-slide
comparator masks.  It does not perform global 2,000 µm² vector cleanup.
"""
import math

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def supervised_view(scores, support_4um: Image.Image, box_xywh, output_wh):
    """Return uint8 class IDs on fixed tissue support and a change report."""
    x, y, width, height = box_xywh
    out_w, out_h = output_wh
    if scores.shape[0] != 24 or min(width, height, out_w, out_h) <= 0:
        raise ValueError('Expected a 24-class score grid and positive view geometry')
    mpp_x, mpp_y = .5 * width / out_w, .5 * height / out_h
    sigma_x, sigma_y = 6 / mpp_x, 6 / mpp_y
    halo = math.ceil(4 * max(sigma_x, sigma_y)) + 8
    dx, dy = halo * width / out_w, halo * height / out_h
    big_w, big_h = out_w + 2 * halo, out_h + 2 * halo
    transform = ((width + 2 * dx) / (8 * big_w), 0, (x - dx) / 8,
                 0, (height + 2 * dy) / (8 * big_h), (y - dy) / 8)
    support = np.asarray(support_4um.transform((big_w, big_h), Image.AFFINE,
                                                transform, resample=Image.NEAREST)) > 0
    extent = ((x - dx) / 14, (y - dy) / 14,
              (x + width + dx) / 14, (y + height + dy) / 14)
    best = np.full((big_h, big_w), -np.inf, np.float32)
    raw = np.zeros((big_h, big_w), np.uint8)
    for cls in range(1, 25):
        score_image = Image.fromarray(np.asarray(scores[cls - 1], np.float32), 'F')
        response = np.asarray(score_image.transform((big_w, big_h), Image.EXTENT,
                                                    extent, resample=Image.BICUBIC), np.float32)
        take = response > best
        best[take], raw[take] = response[take], cls
    raw[~support] = 0
    best.fill(-np.inf)
    result = np.zeros_like(raw)
    for cls in np.unique(raw):
        if cls == 0:
            continue
        response = gaussian_filter((raw == cls).astype(np.float32),
                                   sigma=(sigma_y, sigma_x), mode='mirror', truncate=4)
        take = response > best
        best[take], result[take] = response[take], cls
    result[~support] = 0
    core = result[halo:halo + out_h, halo:halo + out_w].copy()
    old = raw[halo:halo + out_h, halo:halo + out_w]
    inside = support[halo:halo + out_h, halo:halo + out_w]
    if not np.array_equal(core > 0, inside):
        raise AssertionError('Bora tissue support changed')
    return core, {'sigma_um': 6, 'mpp_xy': [mpp_x, mpp_y],
                  'halo_pixels': halo, 'changed_supported_pixels': int(np.count_nonzero(core != old)),
                  'supported_pixels': int(inside.sum())}
