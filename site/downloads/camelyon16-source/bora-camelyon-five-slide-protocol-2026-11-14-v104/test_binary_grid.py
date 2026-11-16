"""Small exact-sampling regression for the two-logit CAM WSI band adapter."""
import importlib.util
from pathlib import Path

import numpy as np


SOURCE = Path('/mnt/sata_ssd/ChatGPT/Bora/migration/'
              'bora-camelyon-five-slide-protocol-2026-11-14-v103/binary_grid.py')
spec = importlib.util.spec_from_file_location('binary_grid', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_exact_sampling_from_overlapping_logits():
    rng = np.random.default_rng(241113)
    height, width = 1024, 1024
    sampled_y, sampled_x = module.sampled_axis(height), module.sampled_axis(width)
    scores = np.zeros((2, len(sampled_y), len(sampled_x)), np.float32)
    counts = np.zeros((len(sampled_y), len(sampled_x)), np.uint8)
    full = np.zeros((2, height, width), np.float32)
    full_counts = np.zeros((height, width), np.uint8)
    for top in module.starts(height):
        for left in module.starts(width):
            logits = rng.normal(size=(2, 512, 512)).astype(np.float32)
            module.accumulate(scores, counts, logits, top, left, sampled_y, sampled_x)
            full[:, top:top+512, left:left+512] += logits
            full_counts[top:top+512, left:left+512] += 1
    assert np.all(counts) and np.all(full_counts)
    assert np.array_equal(scores, full[:, sampled_y[:, None], sampled_x[None, :]])
    assert np.array_equal(counts, full_counts[sampled_y[:, None], sampled_x[None, :]])
    assert np.array_equal(module.prediction(scores, counts),
                          np.argmax(full[:, sampled_y[:, None], sampled_x[None, :]], axis=0) + 1)


if __name__ == '__main__':
    test_exact_sampling_from_overlapping_logits()
    print('BINARY_GRID_EXACT_SAMPLING_VERIFIED')
