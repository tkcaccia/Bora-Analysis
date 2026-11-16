"""Deterministic CAMELYON16 test-ID proposal from authorized metadata only.

No images, lesion masks, thumbnails or feature arrays are accepted. The CLI
requires an affirmative use receipt; tests call the pure selector on fabricated
metadata. Output remains a proposal until official source hashes are audited.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path


TRAINING = ("tumor_093", "normal_093")
SALT = "bora-camelyon16-five-slide-metadata-v3-2026-11-14"
MAX_ENTRIES = 500


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _optional(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be null or a nonblank normalized string")
    return value


def _entry(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Each source entry must be an object")
    slide = value.get("slide_id")
    if not isinstance(slide, str) or not (
        slide.startswith("tumor_") or slide.startswith("normal_")) or \
       len(slide.split("_")[-1]) != 3 or not slide.split("_")[-1].isdigit():
        raise ValueError("Expected official tumor_NNN or normal_NNN slide ID")
    label = "tumor" if slide.startswith("tumor_") else "normal"
    reference = value.get("reference_kind")
    if reference != ("lesion_mask" if label == "tumor" else "negative_slide_label"):
        raise ValueError(f"{slide}: reference kind differs from fixed binary endpoint")
    if value.get("partition") != "public_development" or \
       value.get("official_listing_verified") is not True:
        raise ValueError(f"{slide}: official public development listing not verified")
    patient = _optional(value.get("patient_id"), "patient_id")
    center = _optional(value.get("center"), "center")
    source_hash = _optional(value.get("source_sha256"), "source_sha256")
    if source_hash is not None and (len(source_hash) != 64 or
                                    any(c not in "0123456789abcdef" for c in source_hash)):
        raise ValueError("source_sha256 must be lowercase SHA256 when present")
    return {"slide_id": slide, "slide_label": label,
            "reference_kind": reference,
            "patient_id": patient, "center": center,
            "source_sha256": source_hash}


def _patient_conflict(entries: tuple[dict, ...]) -> bool:
    known = [entry["patient_id"] for entry in entries if entry["patient_id"] is not None]
    return len(known) != len(set(known))


def propose(metadata: dict) -> dict:
    """Propose exactly two tumor and one normal IDs without inspecting content."""
    if metadata.get("format") != "bora-camelyon16-official-metadata-v1" or \
       not isinstance(metadata.get("slides"), list) or \
       not 5 <= len(metadata["slides"]) <= MAX_ENTRIES:
        raise ValueError("A bounded, official-source metadata manifest is required")
    entries = [_entry(item) for item in metadata["slides"]]
    by_id = {item["slide_id"]: item for item in entries}
    if len(by_id) != len(entries) or any(slide not in by_id for slide in TRAINING):
        raise ValueError("Unique entries including both frozen training slides are required")
    if by_id["tumor_093"]["slide_label"] != "tumor" or \
       by_id["normal_093"]["slide_label"] != "normal":
        raise ValueError("The frozen training labels differ")
    training = (by_id[TRAINING[0]], by_id[TRAINING[1]])
    if _patient_conflict(training):
        raise ValueError("The two training slides share a documented patient")
    tumors = [item for item in entries if item["slide_label"] == "tumor" and
              item["slide_id"] not in TRAINING]
    normals = [item for item in entries if item["slide_label"] == "normal" and
               item["slide_id"] not in TRAINING]
    if len(tumors) < 2 or not normals:
        raise ValueError("Two additional tumor and one normal public slides are required")
    ranks = {item["slide_id"]: int(_digest(
        f"{SALT}:{item['slide_id']}".encode()), 16) for item in entries}
    choice = None
    score_best = None
    for left, right in combinations(tumors, 2):
        for normal in normals:
            selected = (left, right, normal)
            if _patient_conflict(training + selected):
                continue
            known_centers = {item["center"] for item in training + selected
                             if item["center"] is not None}
            known_patients = sum(item["patient_id"] is not None for item in selected)
            # Center diversity and verified identity are chosen before the
            # deterministic ID hash. No image statistic enters this order.
            key = (len(known_centers), known_patients,
                   -sum(ranks[item["slide_id"]] for item in selected),
                   tuple(sorted(item["slide_id"] for item in selected)))
            if score_best is None or key > score_best:
                score_best, choice = key, selected
    if choice is None:
        raise ValueError("No patient-compatible two-positive/one-normal triple")
    choice = tuple(sorted(choice, key=lambda item: (item["slide_label"] != "tumor",
                                                  item["slide_id"])))
    selected = [*training, *choice]
    known_patient_ids = [item["patient_id"] for item in selected]
    patient_status = ("NO_DUPLICATE_KNOWN_IDS_IN_SUPPLIED_METADATA" if all(known_patient_ids)
                      and len(set(known_patient_ids)) == 5 else
                      "UNVERIFIED_MISSING_PATIENT_METADATA")
    center_status = ("KNOWN_FOR_ALL_SELECTED_IN_SUPPLIED_METADATA" if all(item["center"] for item in selected)
                     else "INCOMPLETE_CENTER_METADATA")
    return {"format": "bora-camelyon16-metadata-proposal-v3",
            "status": "PROPOSED_NOT_FROZEN",
            "selection_rule": "maximize known-center diversity, then known test patient IDs, then deterministic SHA256 rank",
            "salt": SALT, "training_slide_ids": list(TRAINING),
            "test_slide_ids": [item["slide_id"] for item in choice],
            "selected_metadata": selected,
            "patient_relationship_status": patient_status,
            "center_provenance_status": center_status,
            "source_hashes_complete": all(item["source_sha256"] is not None for item in selected),
            "images_or_annotations_opened": False,
            "cam_specific_training_or_scoring_allowed_by_this_proposal": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--permission-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    permission = json.loads(args.permission_receipt.read_text())
    if permission.get("cam_specific_training_allowed_now") is not True or \
       permission.get("source_terms_and_cam_specific_training_permission_resolved") is not True:
        raise PermissionError("CAMELYON16 study metadata selection is held pending use permission")
    if args.output.exists():
        raise FileExistsError(args.output)
    data = args.metadata.read_bytes()
    result = propose(json.loads(data))
    result["input_metadata_sha256"] = _digest(data)
    result["permission_receipt_sha256"] = _digest(args.permission_receipt.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        with temp.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, args.output)
    finally:
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
