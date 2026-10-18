import numpy as np

from benchmark.analyze_cmpb_boundaries import score


def test_missing_tumor_stroma_interface_is_failure():
    truth = np.ones((32, 32), np.uint8)
    truth[:, 16:] = 3
    predicted = np.full_like(truth, 3)
    result = score(predicted, truth, (0.5, 0.5))
    assert result['reference_boundary_pixels'] > 0
    assert result['predicted_boundary_pixels'] == 0
    assert result['f1_8um'] == 0.0
    assert result['assd_um'] is None


def test_boundary_partner_must_be_stroma():
    truth = np.ones((32, 32), np.uint8)
    truth[:, 16:] = 3
    predicted = np.ones_like(truth)
    predicted[:, 16:] = 5
    assert score(predicted, truth, (0.5, 0.5))['f1_8um'] == 0.0
