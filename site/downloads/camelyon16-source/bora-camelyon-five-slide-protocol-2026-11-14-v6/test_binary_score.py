"""Synthetic, source-free checks of the future CAM binary score contract."""

from pathlib import Path
import importlib.util
import json
import subprocess
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).with_name("binary_score.py")
spec = importlib.util.spec_from_file_location("bora_cam_binary_score_v6", SOURCE)
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


def test_maximum_matching_avoids_greedy_undercount():
    # Greedy would take (reference 1, prediction 1) and stop at one match.
    # Two acceptable one-to-one pairs are available at the fixed IoU cutoff.
    candidates = [(0.9, 1, 1), (0.8, 1, 2), (0.8, 2, 1)]
    assert module._maximum_component_matches(candidates, 2, 2) == 2
    assert module._maximum_component_matches([], 2, 2) == 0


def test_support_and_unknown_regions_have_explicit_denominators():
    ref, support, annotated, excluded = canvas((8, 8))
    ref[2:4, 2:4] = 1
    support[2, 2] = 0
    excluded[7, 7] = 1
    native = ref.copy()
    shared = native.copy()
    shared[2, 2] = 255
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
    with pytest.raises(ValueError, match="must use 255 exactly outside support"):
        module.score_pair(ref, pred, pred, support, annotated, excluded)


def test_255_only_outside_support_and_not_a_predicted_lesion():
    ref, support, annotated, excluded = canvas((8, 8))
    ref[3:5, 3:5] = 1
    support[0, 0] = 0
    native = ref.copy()
    native[0, 0] = 1
    shared = native.copy()
    shared[0, 0] = 255
    result = module.score_pair(ref, native, shared, support, annotated, excluded)
    assert result["native"]["positive_outside_reviewed_support_pixels"] == 1
    assert result["shared_support"]["positive_outside_reviewed_support_pixels"] == 0
    assert result["native"]["iou"] == result["shared_support"]["iou"] == 1
    assert result["shared_support"]["lesions"]["predicted_components"] == 1
    shared[1, 1] = 255
    with pytest.raises(ValueError, match="must use 255 exactly outside support"):
        module.score_pair(ref, native, shared, support, annotated, excluded)
    native[1, 1] = 255
    with pytest.raises(ValueError, match="only binary 0/1"):
        module.score_pair(ref, native, shared, support, annotated, excluded)


def test_new_shared_encoding_matches_previous_scored_domain():
    old_source = SOURCE.parent.parent / "bora-camelyon-five-slide-protocol-2026-11-14-v4/binary_score.py"
    old_spec = importlib.util.spec_from_file_location("bora_cam_binary_score_v4", old_source)
    old = importlib.util.module_from_spec(old_spec)
    assert old_spec.loader is not None
    old_spec.loader.exec_module(old)
    ref, support, annotated, excluded = canvas((12, 12))
    ref[3:7, 4:8] = 1
    support[:2, :2] = 0
    native = ref.copy()
    native[0, 0] = 1
    old_shared = np.where(support, native, 0).astype(np.uint8)
    new_shared = np.where(support, native, 255).astype(np.uint8)
    prior = old.score_pair(ref, native, old_shared, support, annotated, excluded)
    current = module.score_pair(ref, native, new_shared, support, annotated, excluded)
    for variant in ("native", "shared_support"):
        for metric in ("tp", "fp", "fn", "tn", "precision", "recall", "dice", "iou",
                       "false_positive_tissue_area_mm2"):
            assert current[variant][metric] == prior[variant][metric]
        assert current[variant]["lesions"] == prior[variant]["lesions"]
        assert current[variant]["boundary"] == prior[variant]["boundary"]


def test_binary_grid_output_feeds_new_scorer_without_background_relabelling():
    grid_source = SOURCE.parent.parent / "bora-camelyon-binary-inference-2026-11-14-v1/binary_grid.py"
    grid_spec = importlib.util.spec_from_file_location("bora_cam_binary_grid_v1", grid_source)
    grid = importlib.util.module_from_spec(grid_spec)
    assert grid_spec.loader is not None
    grid_spec.loader.exec_module(grid)
    ref, support, annotated, excluded = canvas((8, 8))
    ref[2:4, 2:4] = 1
    support[:2, :2] = 0
    logits = np.zeros((2, 8, 8), np.float32)
    logits[0] = 1
    logits[1, 2:4, 2:4] = 2
    counts = np.ones((8, 8), np.uint8)
    native, shared = grid.finalize_binary_chunk(logits, counts, support)
    assert np.all(shared[:2, :2] == 255)
    scored = module.score_pair(ref, native, shared, support, annotated, excluded)
    assert scored["shared_support"]["iou"] == 1.0
    assert scored["shared_support"]["positive_outside_reviewed_support_pixels"] == 0


def test_new_scorer_receipts_feed_failure_aware_three_slide_summary():
    study_source = SOURCE.parent.parent / "bora-camelyon-five-slide-protocol-2026-11-14-v5/study_aggregate.py"
    study_spec = importlib.util.spec_from_file_location("bora_cam_study_aggregate_v5", study_source)
    study = importlib.util.module_from_spec(study_spec)
    assert study_spec.loader is not None
    study_spec.loader.exec_module(study)
    slides = [
        {"slide_id": "tumor_A", "role": "positive", "patient_id": "P1", "source_sha256": "a" * 64},
        {"slide_id": "tumor_B", "role": "positive", "patient_id": "P2", "source_sha256": "b" * 64},
        {"slide_id": "normal_C", "role": "negative", "patient_id": "P3", "source_sha256": "c" * 64},
    ]
    outcomes, scores = [], {}
    for slide in slides:
        ref, support, annotated, excluded = canvas((8, 8))
        if slide["role"] == "positive":
            ref[2:4, 2:4] = 1
        support[0, 0] = 0
        native = ref.copy()
        shared = np.where(support, native, 255).astype(np.uint8)
        row = module.score_pair(ref, native, shared, support, annotated, excluded)
        row["status"] = "SCORED_REGISTERED_BINARY_WSI"
        row["inputs_sha256"] = {"native": "d" * 64, "shared": "e" * 64,
                                 "reference": "f" * 64, "support": "0" * 64}
        scores["Bora", slide["slide_id"]] = row
        outcomes.append({"method_id": "Bora", "slide_id": slide["slide_id"],
                         "source_sha256": slide["source_sha256"],
                         "native_prediction_sha256": "d" * 64,
                         "shared_prediction_sha256": "e" * 64,
                         "status": "COMPLETE", "elapsed_seconds": 1.0,
                         "peak_host_bytes": 1024, "peak_gpu_bytes": None})
    frozen = {"format": "bora-camelyon-study-input-v1", "test_slides": slides,
              "training_patient_ids": ["P0"],
              "methods": [{"method_id": "Bora", "semantic_binary": True}],
              "outcomes": outcomes}
    summary = study.aggregate(frozen, scores)["methods"][0]
    assert summary["completion_rate"] == 1.0
    assert summary["binary_endpoints"]["shared_support"]["positive_iou_all_selected_failure_zero"] == 1.0
    assert summary["binary_endpoints"]["shared_support"]["negative_specificity_all_selected_failure_zero"] == 1.0


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
