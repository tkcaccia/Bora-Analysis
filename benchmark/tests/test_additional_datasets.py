import numpy as np

from benchmark.run_additional_datasets import score


def test_binary_score_counts_false_positives_outside_target():
    truth = np.zeros((12, 12), np.uint8)
    truth[3:9, 3:9] = 1
    pred = truth.copy()
    pred[0:2, 0:2] = 1
    result = score(pred, truth)
    assert result['foreground_iou'] == 36 / 40
    assert result['foreground_dice'] == 72 / 76
    assert result['pixel_accuracy'] < 1


def test_missing_coarse_target_has_zero_iou_and_undefined_boundary():
    truth = np.zeros((12, 12), np.uint8)
    truth[4:6, 4:6] = 1
    result = score(np.zeros_like(truth), truth)
    assert result['foreground_iou'] == 0
    assert result['boundary_f1']['8'] is None
    assert result['foreground_pixels'] == 0
