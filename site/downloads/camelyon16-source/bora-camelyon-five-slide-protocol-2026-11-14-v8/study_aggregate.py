"""Permission-gated CAMELYON16 binary technical-pilot aggregation.

This module consumes only frozen per-slide score receipts. It never opens a
slide, expert mask or prediction raster. Positive-slide IoU and negative-slide
specificity remain separate endpoints; missing outcomes contribute zero only
to explicitly failure-aware selected-slide summaries.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
TRAINING_IDS = {"tumor_093", "normal_093"}
PROJECT_PERMISSION_GATE = (Path(__file__).resolve().parents[1] /
    "bora-r-camelyon-2026-11-14-v1/PERMISSION_GATE_2026-11-14.json")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_permission(permission: dict) -> None:
    required = (
        "cam_specific_test_allowed",
        "source_terms_and_cam_specific_training_permission_resolved",
        "test_cohort_frozen",
        "scorer_frozen",
    )
    if any(permission.get(name) is not True for name in required):
        raise PermissionError("CAMELYON16 scoring/aggregation requires positive use, cohort and scorer receipts")


def _finite_nonnegative(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _score_confusion(score: dict, role: str, slide_id: str) -> dict:
    if score.get("status") != "SCORED_REGISTERED_BINARY_WSI":
        raise ValueError(f"{slide_id}: per-slide score not complete")
    if score.get("format") != "bora-camelyon-binary-score-prototype-v7" or \
       score.get("score_domain") != "official_annotated_known_pixels_including_outside_tissue_support":
        raise ValueError(f"{slide_id}: official-known-domain scorer version required")
    scored = score.get("scored_known_pixels")
    if not isinstance(scored, int) or scored <= 0:
        raise ValueError(f"{slide_id}: no scored officially known pixels")
    if score.get("annotated_known_pixels") != scored:
        raise ValueError(f"{slide_id}: scored and annotated known domains differ")
    for variant in ("native", "shared_support"):
        row = score.get(variant)
        if not isinstance(row, dict):
            raise ValueError(f"{slide_id}: missing {variant} scores")
        counts = [row.get(name) for name in ("tp", "fp", "fn", "tn")]
        if any(not isinstance(v, int) or v < 0 for v in counts) or sum(counts) != scored:
            raise ValueError(f"{slide_id}: {variant} confusion denominator differs")
        if row.get("reference_positive_pixels") != counts[0] + counts[2]:
            raise ValueError(f"{slide_id}: {variant} reference count differs")
        if role == "positive":
            if counts[0] + counts[2] <= 0:
                raise ValueError(f"{slide_id}: selected positive slide has no scored lesion")
            iou = row.get("iou")
            if not isinstance(iou, (int, float)) or not math.isfinite(iou) or not 0 <= iou <= 1:
                raise ValueError(f"{slide_id}: undefined positive-slide IoU")
            if not math.isclose(iou, counts[0] / (counts[0] + counts[1] + counts[2]), abs_tol=1e-12):
                raise ValueError(f"{slide_id}: IoU arithmetic differs")
        elif role == "negative":
            if counts[0] or counts[2] or row.get("iou") is not None:
                raise ValueError(f"{slide_id}: negative slide has positive reference metrics")
        else:
            raise ValueError("Unknown cohort role")
    if (score["native"]["reference_positive_pixels"] !=
            score["shared_support"]["reference_positive_pixels"]):
        raise ValueError(f"{slide_id}: native/shared reference domains differ")
    return score


def aggregate(spec: dict, score_receipts: dict[tuple[str, str], dict]) -> dict:
    """Return failure-aware per-method summaries for one prespecified 3-slide set.

    ``score_receipts`` is supplied by the gated CLI after SHA verification.
    Source-free synthetic callers can exercise the arithmetic directly.
    """
    if spec.get("format") != "bora-camelyon-study-input-v1":
        raise ValueError("Unrecognized study input format")
    slides = spec.get("test_slides")
    methods = spec.get("methods")
    outcomes = spec.get("outcomes")
    if not isinstance(slides, list) or len(slides) != 3 or \
            not isinstance(methods, list) or not methods or not isinstance(outcomes, list):
        raise ValueError("Three test slides, methods and outcomes are required")
    ids = [slide.get("slide_id") for slide in slides]
    roles = [slide.get("role") for slide in slides]
    if (any(not isinstance(v, str) or not v for v in ids) or
            len(set(ids)) != 3 or set(ids) & TRAINING_IDS or
            sorted(roles) != ["negative", "positive", "positive"]):
        raise ValueError("Freeze two distinct positive and one distinct negative test slide")
    for slide in slides:
        if not isinstance(slide.get("source_sha256"), str) or not SHA256.fullmatch(slide["source_sha256"]):
            raise ValueError("Each test source needs a SHA256 identity")
        if slide.get("patient_id") is not None and not isinstance(slide["patient_id"], str):
            raise ValueError("Patient ID must be text or explicitly unknown")
    known = [slide["patient_id"] for slide in slides if slide.get("patient_id")]
    training_patients = spec.get("training_patient_ids")
    if (not isinstance(training_patients, list) or
            any(not isinstance(x, str) or not x for x in training_patients) or
            len(set(known)) != len(known) or set(known) & set(training_patients)):
        raise ValueError("Known test patients overlap or training identities are absent")
    method_ids = [method.get("method_id") for method in methods]
    if any(not isinstance(mid, str) or not mid for mid in method_ids) or len(set(method_ids)) != len(method_ids):
        raise ValueError("Method identities must be unique")
    if any(type(method.get("semantic_binary")) is not bool for method in methods):
        raise ValueError("Each method must declare binary-semantic eligibility")
    keys = [(row.get("method_id"), row.get("slide_id")) for row in outcomes]
    expected = {(method_id, slide_id) for method_id in method_ids for slide_id in ids}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Exactly one outcome is required for every selected slide and method")
    indexed = {key: row for key, row in zip(keys, outcomes)}
    role_by_id = dict(zip(ids, roles))
    source_by_id = {slide["slide_id"]: slide["source_sha256"] for slide in slides}
    reference_hashes = {}
    support_hashes = {}
    results = []
    for method in methods:
        method_id = method["method_id"]
        semantic = method["semantic_binary"]
        rows = []
        for slide_id in ids:
            outcome = indexed[(method_id, slide_id)]
            status = outcome.get("status")
            if status not in {"COMPLETE", "FAILED", "UNSCORABLE"}:
                raise ValueError(f"{method_id}/{slide_id}: unknown outcome")
            if outcome.get("source_sha256") != source_by_id[slide_id]:
                raise ValueError(f"{method_id}/{slide_id}: frozen source hash differs")
            elapsed = _finite_nonnegative(outcome.get("elapsed_seconds"), "elapsed_seconds")
            host = outcome.get("peak_host_bytes")
            gpu = outcome.get("peak_gpu_bytes")
            host = None if host is None else _finite_nonnegative(host, "peak_host_bytes")
            gpu = None if gpu is None else _finite_nonnegative(gpu, "peak_gpu_bytes")
            key = (method_id, slide_id)
            score = score_receipts.get(key)
            if status != "COMPLETE" and outcome.get("score_path") is not None:
                raise ValueError(f"{method_id}/{slide_id}: incomplete outcome cannot name a score")
            if status == "COMPLETE" and semantic:
                if score is None:
                    raise ValueError(f"{method_id}/{slide_id}: complete semantic result has no score")
                _score_confusion(score, role_by_id[slide_id], slide_id)
                inputs = score.get("inputs_sha256", {})
                for name in ("native", "shared"):
                    bound_hash = outcome.get(f"{name}_prediction_sha256")
                    if not isinstance(bound_hash, str) or not SHA256.fullmatch(bound_hash) or \
                       inputs.get(name) != bound_hash:
                        raise ValueError(f"{method_id}/{slide_id}: scored prediction hash differs")
                for name, seen in (("reference", reference_hashes),
                                   ("support", support_hashes)):
                    value = inputs.get(name)
                    if not isinstance(value, str) or not SHA256.fullmatch(value):
                        raise ValueError(f"{method_id}/{slide_id}: missing scored {name} hash")
                    if slide_id in seen and seen[slide_id] != value:
                        raise ValueError(f"{slide_id}: {name} differs across methods")
                    seen[slide_id] = value
            elif score is not None:
                raise ValueError(f"{method_id}/{slide_id}: score supplied for failed or unscorable result")
            if status != "COMPLETE" and not outcome.get("reason"):
                raise ValueError(f"{method_id}/{slide_id}: failure or nonsemantic reason required")
            rows.append({"slide_id": slide_id, "role": role_by_id[slide_id],
                         "status": status, "reason": outcome.get("reason"),
                         "elapsed_seconds": elapsed,
                         "peak_host_bytes": host, "peak_gpu_bytes": gpu,
                         "score": score if semantic else None})
        complete = [row for row in rows if row["status"] == "COMPLETE"]
        result = {"method_id": method_id, "semantic_binary": semantic,
                  "selected_slides": 3, "completed_slides": len(complete),
                  "completion_rate": len(complete) / 3,
                  "observed_elapsed_seconds_all_selected": sum(row["elapsed_seconds"] for row in rows),
                  "host_peak_observed_count": sum(row["peak_host_bytes"] is not None for row in rows),
                  "peak_host_bytes_observed": max((row["peak_host_bytes"] for row in rows
                                                    if row["peak_host_bytes"] is not None), default=None),
                  "gpu_peak_observed_count": sum(row["peak_gpu_bytes"] is not None for row in rows),
                  "peak_gpu_bytes_observed": max((row["peak_gpu_bytes"] for row in rows
                                                   if row["peak_gpu_bytes"] is not None), default=None),
                  "cases": rows}
        if semantic:
            variants = {}
            for variant in ("native", "shared_support"):
                positive_values = [row["score"][variant]["iou"] for row in rows
                                   if row["role"] == "positive" and row["score"] is not None]
                negative_values = [row["score"][variant]["tn"] /
                                   (row["score"][variant]["tn"] + row["score"][variant]["fp"])
                                   for row in rows if row["role"] == "negative" and row["score"] is not None]
                variants[variant] = {
                    "selected_positive_slides": 2,
                    "scored_positive_slides": len(positive_values),
                    "positive_iou_conditional_mean": sum(positive_values) / len(positive_values)
                    if positive_values else None,
                    "positive_iou_all_selected_failure_zero": sum(positive_values) / 2,
                    "selected_negative_slides": 1,
                    "scored_negative_slides": len(negative_values),
                    "negative_specificity_conditional_mean": sum(negative_values) / len(negative_values)
                    if negative_values else None,
                    "negative_specificity_all_selected_failure_zero": sum(negative_values),
                }
            result["binary_endpoints"] = variants
        else:
            result["binary_endpoints"] = None
        results.append(result)
    return {"format": "bora-camelyon-binary-technical-aggregate-v1",
            "scope": "three selected development slides; technical pilot only",
            "selected_slides": slides,
            "known_test_patients": len(known),
            "unknown_test_patients": 3 - len(known),
            "paired_reference_sha256": reference_hashes,
            "paired_support_sha256": support_hashes,
            "no_population_ci": True,
            "methods": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permission-receipt", type=Path, required=True)
    parser.add_argument("--study-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project_gate = json.loads(PROJECT_PERMISSION_GATE.read_text())
    if project_gate.get("status") != "APPROVED" or \
       project_gate.get("cam_specific_training_allowed_now") is not True or \
       project_gate.get("three_slide_cam_test_allowed_now") is not True:
        raise PermissionError("CAMELYON16 aggregation requires an approved project permission gate")
    # Only the authoritative project gate and use receipt are opened before authorization.
    require_permission(json.loads(args.permission_receipt.read_text()))
    spec = json.loads(args.study_input.read_text())
    scores = {}
    for row in spec.get("outcomes", []):
        if row.get("status") != "COMPLETE" or not row.get("score_path"):
            continue
        path = Path(row["score_path"])
        if file_sha256(path) != row.get("score_sha256"):
            raise ValueError(f"Score receipt hash differs: {path}")
        scores[row["method_id"], row["slide_id"]] = json.loads(path.read_text())
    result = aggregate(spec, scores)
    result["study_input_sha256"] = file_sha256(args.study_input)
    result["permission_receipt_sha256"] = file_sha256(args.permission_receipt)
    result["project_permission_gate_sha256"] = file_sha256(PROJECT_PERMISSION_GATE)
    result["score_receipt_hashes"] = {
        f"{value['method_id']}/{value['slide_id']}": value["score_sha256"]
        for value in spec["outcomes"] if value.get("score_path")}
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
