"""Synthetic metadata tests; no real CAMELYON16 slide is selected here."""

from pathlib import Path
import importlib.util
import json
import subprocess
import sys

import pytest


SOURCE = Path(__file__).with_name("select_metadata_only.py")
spec = importlib.util.spec_from_file_location("cam_metadata_selection_v3", SOURCE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def entry(slide_id, center, patient_id):
    return {"slide_id": slide_id, "partition": "public_development",
            "official_listing_verified": True,
            "reference_kind": ("lesion_mask" if slide_id.startswith("tumor_")
                               else "negative_slide_label"),
            "center": center, "patient_id": patient_id, "source_sha256": None}


def fabricated_manifest():
    return {"format": "bora-camelyon16-official-metadata-v1", "slides": [
        entry("tumor_093", "A", "train-tumor"),
        entry("normal_093", "A", "train-normal"),
        entry("tumor_001", "B", "test-t1"),
        entry("tumor_002", "A", "test-t2"),
        entry("tumor_003", "C", "test-t3"),
        entry("normal_001", "D", "test-n1"),
        entry("normal_002", "B", "test-n2"),
    ]}


def test_center_diversity_and_fixed_two_positive_one_normal():
    result = module.propose(fabricated_manifest())
    assert result["test_slide_ids"] == ["tumor_001", "tumor_003", "normal_001"]
    assert result["patient_relationship_status"] == "NO_DUPLICATE_KNOWN_IDS_IN_SUPPLIED_METADATA"
    assert result["center_provenance_status"] == "KNOWN_FOR_ALL_SELECTED_IN_SUPPLIED_METADATA"
    assert result["status"] == "PROPOSED_NOT_FROZEN"
    assert result["source_hashes_complete"] is False


def test_input_order_cannot_change_selected_id_order():
    metadata = fabricated_manifest()
    first = module.propose(metadata)
    metadata["slides"] = list(reversed(metadata["slides"]))
    second = module.propose(metadata)
    assert first["test_slide_ids"] == second["test_slide_ids"]
    assert first["selected_metadata"] == second["selected_metadata"]


def test_known_training_patient_overlap_excludes_candidate():
    metadata = fabricated_manifest()
    metadata["slides"][2]["patient_id"] = "train-tumor"
    result = module.propose(metadata)
    assert "tumor_001" not in result["test_slide_ids"]
    # Both normal candidates give the same center count here; fixed ID hash wins.
    assert result["test_slide_ids"] == ["tumor_002", "tumor_003", "normal_002"]


def test_unknown_identity_is_disclosed_not_called_disjoint():
    metadata = fabricated_manifest()
    metadata["slides"][0]["patient_id"] = None
    metadata["slides"][3]["center"] = None
    result = module.propose(metadata)
    assert result["patient_relationship_status"] == "UNVERIFIED_MISSING_PATIENT_METADATA"


def test_duplicate_or_wrong_reference_kind_rejected():
    metadata = fabricated_manifest()
    metadata["slides"].append(metadata["slides"][2].copy())
    with pytest.raises(ValueError, match="Unique entries"):
        module.propose(metadata)
    metadata = fabricated_manifest()
    metadata["slides"][2]["reference_kind"] = "negative_slide_label"
    with pytest.raises(ValueError, match="reference kind"):
        module.propose(metadata)


def test_cli_refuses_current_pending_gate_before_metadata_read(tmp_path):
    pending = Path(__file__).parents[1] / \
        "bora-r-camelyon-2026-11-14-v1/PERMISSION_GATE_2026-11-14.json"
    missing = tmp_path / "never-opened.json"
    result = subprocess.run([
        sys.executable, str(SOURCE), "--metadata", str(missing),
        "--permission-receipt", str(pending),
        "--output", str(tmp_path / "proposal.json")],
        capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "held pending use permission" in result.stderr
    assert "never-opened" not in result.stderr
    assert not (tmp_path / "proposal.json").exists()
