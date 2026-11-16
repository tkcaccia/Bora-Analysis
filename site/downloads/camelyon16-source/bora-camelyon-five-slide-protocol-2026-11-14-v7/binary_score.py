"""Permission-gated, registered binary WSI scorer; no CAM data used to develop it.

Inputs are aligned uint8 .npy rasters on one measured pixel-centre grid.
Reference 1 means metastasis, 0 means nonmetastatic. ``annotated`` is the
officially interpretable domain, ``excluded`` covers unknown regions, and
``support`` is reviewed H&E-only tissue. Both variants use the SAME official
known-pixel denominator, including known pixels outside reviewed support.
The shared-support prediction uses 255 outside support to denote suppressed
tumor prediction, not a pixel excluded from scoring. Native predictions remain
binary 0/1 everywhere. No array is resized. The CLI checks the authoritative
project permission gate and requires a separate affirmative use receipt.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
from scipy.ndimage import distance_transform_edt, label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


MAX_PIXELS = 100_000_000  # initial technical bound, not an admissible cohort size
CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)
PROJECT_PERMISSION_GATE = (Path(__file__).resolve().parents[1] /
    "bora-r-camelyon-2026-11-14-v1/PERMISSION_GATE_2026-11-14.json")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_binary(name: str, value: np.ndarray, shape: tuple[int, int] | None = None,
                  *, allow_unscored: bool = False) -> None:
    if value.ndim != 2 or value.dtype not in (np.dtype("bool"), np.dtype("uint8")):
        raise ValueError(f"{name}: expected a 2D bool/uint8 raster")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name}: grid shape differs; resampling is forbidden")
    invalid = ((value > 1) & (value != 255)) if allow_unscored else (value > 1)
    if np.any(invalid):
        raise ValueError(f"{name}: only binary 0/1 and permitted unscored 255 pixels are allowed")


def _interfaces(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Two-sided four-neighbor transitions, never counting a domain edge."""
    edge = np.zeros(mask.shape, dtype=bool)
    h = valid[:, :-1] & valid[:, 1:] & (mask[:, :-1] != mask[:, 1:])
    v = valid[:-1, :] & valid[1:, :] & (mask[:-1, :] != mask[1:, :])
    edge[:, :-1] |= h
    edge[:, 1:] |= h
    edge[:-1, :] |= v
    edge[1:, :] |= v
    return edge


def _boundary(reference: np.ndarray, prediction: np.ndarray, valid: np.ndarray,
              spacing_yx_um: tuple[float, float], tolerance_um: float) -> dict:
    ref = _interfaces(reference, valid)
    pred = _interfaces(prediction, valid)
    nr, npred = int(np.count_nonzero(ref)), int(np.count_nonzero(pred))
    if not nr:
        return {"reference_interface_pixels": 0, "predicted_interface_pixels": npred,
                "f1": None, "hd95_um": None,
                "reason": "no_reference_metastasis_interface"}
    if not npred:
        return {"reference_interface_pixels": nr, "predicted_interface_pixels": 0,
                "f1": 0.0, "hd95_um": None,
                "reason": "prediction_has_no_metastasis_interface"}
    to_ref = distance_transform_edt(~ref, sampling=spacing_yx_um)[pred]
    to_pred = distance_transform_edt(~pred, sampling=spacing_yx_um)[ref]
    precision = float(np.mean(to_ref <= tolerance_um))
    recall = float(np.mean(to_pred <= tolerance_um))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"reference_interface_pixels": nr, "predicted_interface_pixels": npred,
            "precision": precision, "recall": recall, "f1": f1,
            "hd95_um": float(max(np.percentile(to_ref, 95),
                                  np.percentile(to_pred, 95))),
            "reason": None}


def _maximum_component_matches(candidates: list[tuple[float, int, int]],
                               n_ref: int, n_pred: int) -> int:
    """Maximum-cardinality matching after a prespecified pair-IoU cutoff."""
    if not candidates or not n_ref or not n_pred:
        return 0
    rows = np.fromiter((ri - 1 for _, ri, _ in candidates), dtype=np.int32)
    cols = np.fromiter((pi - 1 for _, _, pi in candidates), dtype=np.int32)
    graph = coo_matrix((np.ones(len(candidates), dtype=np.int8), (rows, cols)),
                       shape=(n_ref, n_pred)).tocsr()
    return int(np.count_nonzero(maximum_bipartite_matching(
        graph, perm_type="column") >= 0))


def _lesions(reference: np.ndarray, prediction: np.ndarray, valid: np.ndarray,
             match_iou: float) -> dict:
    """One-to-one 8-connected component matching at a fixed component IoU."""
    ref_ids, n_ref = label(reference & valid, structure=CONNECTIVITY_8)
    pred_ids, n_pred = label(prediction & valid, structure=CONNECTIVITY_8)
    if max(n_ref, n_pred) > 1_000_000:
        raise MemoryError("Too many lesion components for bounded scorer")
    ref_sizes = np.bincount(ref_ids[valid], minlength=n_ref + 1)
    pred_sizes = np.bincount(pred_ids[valid], minlength=n_pred + 1)
    pairs: dict[int, int] = {}
    for start in range(0, reference.shape[0], 128):
        end = min(reference.shape[0], start + 128)
        keep = valid[start:end] & (ref_ids[start:end] > 0) & (pred_ids[start:end] > 0)
        if not np.any(keep):
            continue
        encoded = (ref_ids[start:end][keep].astype(np.uint64) << np.uint64(32)) | \
            pred_ids[start:end][keep].astype(np.uint64)
        keys, counts = np.unique(encoded, return_counts=True)
        for key, count in zip(keys, counts):
            identifier = int(key)
            pairs[identifier] = pairs.get(identifier, 0) + int(count)
        if len(pairs) > 1_000_000:
            raise MemoryError("Too many overlapping lesion pairs")
    candidates = []
    for key, overlap in pairs.items():
        ri, pi = key >> 32, key & 0xffffffff
        iou = overlap / (int(ref_sizes[ri]) + int(pred_sizes[pi]) - overlap)
        if iou >= match_iou:
            candidates.append((iou, ri, pi))
    matches = _maximum_component_matches(candidates, int(n_ref), int(n_pred))
    return {"reference_components": int(n_ref),
            "predicted_components": int(n_pred),
            "matched_components": matches,
            "missed_reference_components": int(n_ref) - matches,
            "unmatched_predicted_components": int(n_pred) - matches,
            "connectivity": 8, "match_iou_threshold": match_iou}


def _one_output(reference: np.ndarray, prediction: np.ndarray, valid: np.ndarray,
                support: np.ndarray, annotated_known: np.ndarray,
                spacing_yx_um: tuple[float, float], tolerance_um: float,
                match_iou: float) -> dict:
    actual = reference.astype(bool) & valid
    guessed = (prediction == 1) & valid
    tp = int(np.count_nonzero(actual & guessed))
    n_true = int(np.count_nonzero(actual))
    n_pred = int(np.count_nonzero(guessed))
    fp, fn = n_pred - tp, n_true - tp
    tn = int(np.count_nonzero(valid)) - tp - fp - fn
    area_mm2 = spacing_yx_um[0] * spacing_yx_um[1] / 1_000_000
    if n_true:
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_true
        dice = 2 * tp / (n_true + n_pred)
        iou = tp / (n_true + n_pred - tp)
        undefined = None
    else:
        precision = recall = dice = iou = None
        undefined = "no_reference_metastasis; report false-positive area"
    tissue_fp = int(np.count_nonzero(guessed & ~actual & support.astype(bool)))
    outside_support_fp = fp - tissue_fp
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "reference_positive_pixels": n_true,
            "predicted_positive_pixels_in_known_domain": n_pred,
            "precision": precision, "recall": recall, "dice": dice, "iou": iou,
            "positive_metrics_undefined_reason": undefined,
            "false_positive_tissue_area_mm2": tissue_fp * area_mm2,
            "false_positive_known_outside_support_area_mm2": outside_support_fp * area_mm2,
            "positive_outside_reviewed_support_pixels": int(np.count_nonzero(
                (prediction == 1) & annotated_known & ~support.astype(bool))),
            "boundary": _boundary(reference.astype(bool), prediction == 1,
                                  valid, spacing_yx_um, tolerance_um),
            "lesions": _lesions(reference.astype(bool), prediction == 1,
                               valid, match_iou)}


def score_pair(reference: np.ndarray, native: np.ndarray, shared: np.ndarray,
               support: np.ndarray, annotated: np.ndarray, excluded: np.ndarray,
               *, spacing_xy_um: tuple[float, float] = (4.0, 4.0),
               tolerance_um: float = 5.0, match_iou: float = 0.1) -> dict:
    """Score aligned native and common-support outputs on one known domain."""
    shape = reference.shape
    for name, array in (("reference", reference), ("native", native),
                        ("shared", shared), ("support", support),
                        ("annotated", annotated), ("excluded", excluded)):
        _check_binary(name, array, shape, allow_unscored=name == "shared")
    if reference.size > MAX_PIXELS:
        raise MemoryError("Pixel grid exceeds versioned scorer's initial bound")
    if len(spacing_xy_um) != 2 or not all(math.isfinite(v) and v > 0 for v in spacing_xy_um) or \
       not math.isfinite(tolerance_um) or tolerance_um <= 0 or \
       not math.isfinite(match_iou) or not 0 < match_iou <= 1:
        raise ValueError("Measured spacing and prespecified metric thresholds required")
    known = annotated.astype(bool) & ~excluded.astype(bool)
    if np.any(reference.astype(bool) & ~known):
        raise ValueError("Positive reference pixels outside the official known domain")
    if np.any((shared == 255) != ~support.astype(bool)):
        raise ValueError("Common-support output must use 255 exactly outside support")
    valid = known
    if not np.any(valid):
        raise ValueError("No officially annotated known pixels remain")
    spacing_yx_um = (float(spacing_xy_um[1]), float(spacing_xy_um[0]))
    dropped = int(np.count_nonzero(known & ~support.astype(bool)))
    result = {"format": "bora-camelyon-binary-score-prototype-v7",
              "shape_yx": list(shape), "spacing_xy_um": list(spacing_xy_um),
              "score_domain": "official_annotated_known_pixels_including_outside_tissue_support",
              "annotated_known_pixels": int(np.count_nonzero(known)),
              "scored_known_pixels": int(np.count_nonzero(valid)),
              "known_pixels_inside_reviewed_support": int(np.count_nonzero(
                  known & support.astype(bool))),
              "annotated_pixels_outside_reviewed_support": dropped,
              "reference_positive_pixels_outside_reviewed_support": int(np.count_nonzero(
                  reference.astype(bool) & known & ~support.astype(bool))),
              "excluded_pixels": int(np.count_nonzero(excluded)),
              "tolerance_um": tolerance_um,
              "lesion_match_iou": match_iou}
    result["native"] = _one_output(reference, native, valid, support, known,
                                    spacing_yx_um, tolerance_um, match_iou)
    result["shared_support"] = _one_output(reference, shared, valid, support, known,
                                            spacing_yx_um, tolerance_um, match_iou)
    return result


def _load(path: Path, *, allow_unscored: bool = False) -> np.ndarray:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    _check_binary(str(path), array, allow_unscored=allow_unscored)
    return array


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "native", "shared", "support", "annotated", "excluded"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--permission-receipt", type=Path, required=True)
    parser.add_argument("--spacing-x-um", type=float, required=True)
    parser.add_argument("--spacing-y-um", type=float, required=True)
    parser.add_argument("--tolerance-um", type=float, default=5.0)
    parser.add_argument("--lesion-match-iou", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project_gate = json.loads(PROJECT_PERMISSION_GATE.read_text())
    if project_gate.get("status") != "APPROVED" or \
       project_gate.get("cam_specific_training_allowed_now") is not True or \
       project_gate.get("three_slide_cam_test_allowed_now") is not True:
        raise PermissionError("CAMELYON16 test scoring requires an approved project permission gate")
    permission = json.loads(args.permission_receipt.read_text())
    if permission.get("cam_specific_test_allowed") is not True or \
       permission.get("source_terms_and_cam_specific_training_permission_resolved") is not True or \
       permission.get("test_cohort_frozen") is not True or \
       permission.get("scorer_frozen") is not True:
        raise PermissionError("CAMELYON16 test scoring requires a positive use/cohort/scorer receipt")
    paths = {name: getattr(args, name) for name in
             ("reference", "native", "shared", "support", "annotated", "excluded")}
    arrays = {name: _load(path, allow_unscored=name == "shared")
              for name, path in paths.items()}
    pixels = arrays["reference"].size
    if pixels > MAX_PIXELS:
        raise MemoryError("Grid exceeds scorer's declared pixel bound")
    with Path("/proc/meminfo").open() as stream:
        available_kib = next(int(line.split()[1]) for line in stream
                             if line.startswith("MemAvailable:"))
    if pixels * 128 > available_kib * 1024 * 0.7:
        raise MemoryError("Estimated metric peak exceeds 70% of available host memory")
    started = perf_counter()
    result = score_pair(*(arrays[name] for name in paths),
                        spacing_xy_um=(args.spacing_x_um, args.spacing_y_um),
                        tolerance_um=args.tolerance_um,
                        match_iou=args.lesion_match_iou)
    result["scorer_wall_seconds"] = perf_counter() - started
    result["scorer_peak_host_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["inputs_sha256"] = {name: _sha256(path) for name, path in paths.items()}
    result["permission_receipt_sha256"] = _sha256(args.permission_receipt)
    result["project_permission_gate_sha256"] = _sha256(PROJECT_PERMISSION_GATE)
    result["status"] = "SCORED_REGISTERED_BINARY_WSI"
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        with temp.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, args.output)
    finally:
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
