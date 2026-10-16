import numpy as np

from benchmark.run_bora_pilot import internal_border, metrics, normalize_truth


def test_zero_can_be_a_real_tumour_class_before_tiling():
    raw = np.array([[0, 0, 1], [0, 1, 3]], dtype=np.uint8)
    truth, mapping = normalize_truth(raw, [3])
    assert mapping == {0: 1, 1: 2}
    assert truth.tolist() == [[1, 1, 2], [1, 2, 0]]


def test_internal_border_excludes_tissue_background_edges():
    labels = np.array([[0, 1, 1], [0, 1, 2], [0, 1, 2]], dtype=np.uint8)
    border = internal_border(labels)
    assert not border[:, 0].any()
    assert not border[0, 1]
    assert border[1, 1] and border[1, 2]
    same = metrics(labels, labels)
    assert same['macro_iou'] == 1.0
    assert same['internal_boundary_f1']['8'] == 1.0
