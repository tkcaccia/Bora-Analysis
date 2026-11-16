"""Synthetic checks only; no CAMELYON16 source/reference data are opened."""

from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("study_aggregate", HERE / "study_aggregate.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
SCORER_PATH = HERE.parent / 'bora-camelyon-five-slide-protocol-2026-11-14-v4/binary_score.py'
SCORER_SPEC = importlib.util.spec_from_file_location('cam_binary_score', SCORER_PATH)
SCORER = importlib.util.module_from_spec(SCORER_SPEC)
assert SCORER_SPEC.loader is not None
SCORER_SPEC.loader.exec_module(SCORER)


def study(*, semantic=True):
    slides = [
        {"slide_id": "tumor_X", "role": "positive", "patient_id": "P1", "source_sha256": "a" * 64},
        {"slide_id": "tumor_Y", "role": "positive", "patient_id": None, "source_sha256": "b" * 64},
        {"slide_id": "normal_Z", "role": "negative", "patient_id": "P3", "source_sha256": "c" * 64},
    ]
    method = {"method_id": "Bora" if semantic else "KODAMA", "semantic_binary": semantic}
    outcomes = [{"method_id": method["method_id"], "slide_id": slide["slide_id"],
                 "source_sha256": slide["source_sha256"],
                 "native_prediction_sha256": "d" * 64,
                 "shared_prediction_sha256": "e" * 64,
                 "status": "COMPLETE", "elapsed_seconds": 2.0,
                 "peak_host_bytes": 100, "peak_gpu_bytes": 10}
                for slide in slides]
    return {"format": "bora-camelyon-study-input-v1", "test_slides": slides,
            "training_patient_ids": ["P0"], "methods": [method], "outcomes": outcomes}


def score(*, tp, fp, fn, tn, negative=False):
    def row():
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "reference_positive_pixels": tp + fn,
                "iou": None if negative else tp / (tp + fp + fn)}
    return {"status": "SCORED_REGISTERED_BINARY_WSI",
            "scored_tissue_pixels": tp + fp + fn + tn,
            "inputs_sha256": {"native": "d" * 64, "shared": "e" * 64,
                              "reference": "f" * 64, "support": "0" * 64},
            "native": row(), "shared_support": row()}


def scores():
    return {("Bora", "tumor_X"): score(tp=6, fp=2, fn=2, tn=10),
            ("Bora", "tumor_Y"): score(tp=8, fp=1, fn=1, tn=10),
            ("Bora", "normal_Z"): score(tp=0, fp=2, fn=0, tn=18, negative=True)}


def test_complete_study_separates_positive_iou_and_negative_specificity():
    result = MODULE.aggregate(study(), scores())
    method = result["methods"][0]
    endpoint = method["binary_endpoints"]["native"]
    assert method["selected_slides"] == 3 and method["completion_rate"] == 1.0
    assert endpoint["selected_positive_slides"] == 2
    assert endpoint["scored_positive_slides"] == 2
    assert endpoint["positive_iou_conditional_mean"] == pytest.approx((0.6 + 0.8) / 2)
    assert endpoint["positive_iou_all_selected_failure_zero"] == pytest.approx(0.7)
    assert endpoint["negative_specificity_conditional_mean"] == pytest.approx(0.9)
    assert result["unknown_test_patients"] == 1 and result["no_population_ci"] is True


def test_failed_positive_keeps_selected_denominator_and_runtime():
    spec = study()
    spec["outcomes"][1].update({"status": "FAILED", "reason": "timeout"})
    data = scores()
    del data[("Bora", "tumor_Y")]
    method = MODULE.aggregate(spec, data)["methods"][0]
    endpoint = method["binary_endpoints"]["shared_support"]
    assert method["completion_rate"] == pytest.approx(2 / 3)
    assert method["observed_elapsed_seconds_all_selected"] == 6.0
    assert endpoint["scored_positive_slides"] == 1
    assert endpoint["positive_iou_conditional_mean"] == pytest.approx(0.6)
    assert endpoint["positive_iou_all_selected_failure_zero"] == pytest.approx(0.3)
    assert endpoint["selected_negative_slides"] == 1


def test_anonymous_method_has_completion_but_no_semantic_score():
    result = MODULE.aggregate(study(semantic=False), {})["methods"][0]
    assert result["completion_rate"] == 1.0
    assert result["binary_endpoints"] is None


def test_missing_outcome_and_patient_overlap_rejected():
    spec = study()
    spec["outcomes"].pop()
    with pytest.raises(ValueError, match="Exactly one outcome"):
        MODULE.aggregate(spec, scores())
    spec = study()
    spec["training_patient_ids"] = ["P1"]
    with pytest.raises(ValueError, match="overlap"):
        MODULE.aggregate(spec, scores())


def test_source_and_prediction_hash_registration_rejected():
    spec = study()
    spec["outcomes"][0]["source_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="source hash differs"):
        MODULE.aggregate(spec, scores())
    spec = study()
    spec["outcomes"][0]["native_prediction_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="scored prediction hash differs"):
        MODULE.aggregate(spec, scores())


def test_paired_methods_must_use_same_reference_and_support():
    spec = study()
    spec["methods"].append({"method_id": "UNet", "semantic_binary": True})
    spec["outcomes"].extend([{**outcome, "method_id": "UNet"}
                             for outcome in spec["outcomes"]])
    data = scores()
    data.update({("UNet", slide): deepcopy(score_receipt)
                 for (_, slide), score_receipt in scores().items()})
    data[("UNet", "tumor_X")]["inputs_sha256"]["reference"] = "9" * 64
    with pytest.raises(ValueError, match="reference differs across methods"):
        MODULE.aggregate(spec, data)


def test_invalid_reference_semantics_and_iou_arithmetic_rejected():
    data = scores()
    data[("Bora", "normal_Z")]["native"]["iou"] = 0.0
    with pytest.raises(ValueError, match="negative slide"):
        MODULE.aggregate(study(), data)
    data = scores()
    data[("Bora", "tumor_X")]["native"]["iou"] = 1.0
    with pytest.raises(ValueError, match="IoU arithmetic"):
        MODULE.aggregate(study(), data)


def test_current_permission_gate_refuses_before_loading_study_input(tmp_path):
    gate = Path('/mnt/sata_ssd/ChatGPT/Bora/migration/bora-r-camelyon-2026-11-14-v1/PERMISSION_GATE_2026-11-14.json')
    result = subprocess.run([sys.executable, str(HERE / 'study_aggregate.py'),
                             '--permission-receipt', str(gate),
                             '--study-input', str(tmp_path / 'not-present.json'),
                             '--output', str(tmp_path / 'out.json')],
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert 'PermissionError' in result.stderr
    assert not (tmp_path / 'out.json').exists()


def test_negative_failure_keeps_failure_aware_specificity_zero():
    spec = study()
    spec["outcomes"][2].update({"status": "FAILED", "reason": "inference-failure"})
    data = scores()
    del data[("Bora", "normal_Z")]
    endpoint = MODULE.aggregate(spec, data)["methods"][0]["binary_endpoints"]["native"]
    assert endpoint["selected_negative_slides"] == 1
    assert endpoint["scored_negative_slides"] == 0
    assert endpoint["negative_specificity_conditional_mean"] is None
    assert endpoint["negative_specificity_all_selected_failure_zero"] == 0


def test_v4_pixel_scorer_output_can_feed_study_aggregator():
    spec = study()
    data = {}
    for slide in spec['test_slides']:
        reference = np.zeros((12, 12), dtype=np.uint8)
        if slide['role'] == 'positive':
            reference[3:8, 3:8] = 1
        support = np.ones_like(reference)
        excluded = np.zeros_like(reference)
        native = reference.copy()
        shared = reference.copy()
        score_receipt = SCORER.score_pair(reference, native, shared, support,
                                          support, excluded)
        score_receipt['status'] = 'SCORED_REGISTERED_BINARY_WSI'
        score_receipt['inputs_sha256'] = {
            'native': 'd' * 64, 'shared': 'e' * 64,
            'reference': 'f' * 64, 'support': '0' * 64}
        data[('Bora', slide['slide_id'])] = score_receipt
    result = MODULE.aggregate(spec, data)['methods'][0]['binary_endpoints']['native']
    assert result['positive_iou_all_selected_failure_zero'] == 1.0
    assert result['negative_specificity_all_selected_failure_zero'] == 1.0
