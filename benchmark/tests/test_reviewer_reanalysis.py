import numpy as np

from benchmark.reanalyze_bora_pilot import pair_metrics


def test_pair_aware_boundary_rejects_wrong_class_on_same_contour():
    truth = np.full((50, 50), 1, np.uint8)
    truth[:, 25:] = 2
    prediction = truth.copy()
    prediction[:, 25:] = 3
    result = pair_metrics(prediction, truth)
    assert result['f1_8'] == 0.0
    assert result['missing_or_extra_pairs'] == ['1-2', '1-3']
