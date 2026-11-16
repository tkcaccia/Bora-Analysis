"""Synthetic, source-free checks of the future CAM binary score contract."""

from pathlib import Path
import importlib.util
import json
import subprocess
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).with_name("binary_score.py")
spec = importlib.util.spec_from_file_location("bora_cam_binary_score_v2", SOURCE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def canvas(shape=(12, 12)):
    reference = np.zeros(shape, np.uint8)
    support = np.ones(shape, np.uint8)
    annotated = np.ones(shape, np.uint8)
    excluded = np.zeros(shape, np.uint8)
    return reference, support, annotated, excluded


def test_perfect_positive_lesion_has_exact_metrics():
    ref, support, annotated, excluded = canvas()
    ref[3:7, 4:8] = 1
    result = module.score_pair(ref, ref.copy(), ref.copy(),
                               support, annotated, excluded)
    assert result["scored_tissue_pixels"] == 144
    for mode in ("native", "shared_support"):
        item = result[mode]
        assert (item["tp"], item["fp"], item["fn"], item["tn"]) == (16, 0, 0, 128)
        assert item["precision"] == item["recall"] == item["dice"] == item["iou"] == 1
        assert item["boundary"]["f1"] == 1
        assert item["boundary"]["hd95_um"] == 0
        assert item["lesions"]["matched_components"] == 1


def test_negative_slide_reports_false_positive_area_and_undefined_iou():
    ref, support, annotated, excluded = canvas()
    pred = ref.copy()
    pred[2, 2] = pred[8, 8] = 1
    result = module.score_pair(ref, pred, pred.copy(),
                               support, annotated, excluded)
    item = result["native"]
    assert (item["tp"], item["fp"], item["fn"], item["tn"]) == (0, 2, 0, 142)
    assert item["iou"] is None and item["recall"] is None
    assert item["false_positive_tissue_area_mm2"] == 2 * 16 / 1_000_000
    assert item["boundary"]["f1"] is None
    assert item["lesions"]["unmatched_predicted_components"] == 2


def test_missing_positive_prediction_is_zero_not_missing_score():
    ref, support, annotated, excluded = canvas()
    ref[3:7, 3:7] = 1
    pred = np.zeros_like(ref)
    item = module.score_pair(ref, pred, pred, support, annotated, excluded)["native"]
    assert (item["tp"], item["fp"], item["fn"]) == (0, 0, 16)
    assert item["precision"] == item["recall"] == item["dice"] == item["iou"] == 0
    assert item["boundary"]["f1"] == 0
    assert item["lesions"]["missed_reference_components"] == 1


def test_one_predicted_component_cannot_detect_two_reference_lesions():
    ref, support, annotated, excluded = canvas((16, 16))
    ref[3:6, 3:6] = 1
    ref[3:6, 10:13] = 1
    pred = ref.copy()
    pred[4, 6:10] = 1
    lesions = module.score_pair(ref, pred, pred, support, annotated, excluded)["native"]["lesions"]
    assert lesions["reference_components"] == 2
    assert lesions["predicted_components"] == 1
    assert lesions["matched_components"] == 1
    assert lesions["missed_reference_components"] == 1


def test_support_and_unknown_regions_have_explicit_denominators():
    ref, support, annotated, excluded = canvas((8, 8))
    ref[2:4, 2:4] = 1
    support[2, 2] = 0
    excluded[7, 7] = 1
    native = ref.copy()
    shared = native.copy()
    shared[2, 2] = 0
    result = module.score_pair(ref, native, shared, support, annotated, excluded)
    assert result["annotated_known_pixels"] == 63
    assert result["scored_tissue_pixels"] == 62
    assert result["reference_positive_pixels_outside_reviewed_support"] == 1
    assert result["annotated_pixels_outside_reviewed_support"] == 1
    assert result["native"]["positive_outside_reviewed_support_pixels"] == 1
    assert result["shared_support"]["positive_outside_reviewed_support_pixels"] == 0
    assert result["native"]["tp"] == result["shared_support"]["tp"] == 3


def test_wrong_grid_or_shared_support_leak_rejected():
    ref, support, annotated, excluded = canvas()
    pred = ref.copy()
    with pytest.raises(ValueError, match="grid shape differs"):
        module.score_pair(ref, pred, pred[:4], support, annotated, excluded)
    support[0, 0] = 0
    pred[0, 0] = 1
    with pytest.raises(ValueError, match="positives outside support"):
        module.score_pair(ref, pred, pred, support, annotated, excluded)


def test_cli_refuses_pending_permission_before_reading_masks(tmp_path):
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"cam_specific_test_allowed": False}))
    missing = tmp_path / "never-opened.npy"
    command = [sys.executable, str(SOURCE),
               "--permission-receipt", str(gate),
               "--spacing-x-um", "4", "--spacing-y-um", "4",
               "--output", str(tmp_path / "result.json")]
    for name in ("reference", "native", "shared", "support", "annotated", "excluded"):
        command.extend([f"--{name}", str(missing)])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "CAMELYON16 test scoring requires" in result.stderr
    assert "never-opened" not in result.stderr
    assert not (tmp_path / "result.json").exists()
