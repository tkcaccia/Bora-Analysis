#!/usr/bin/env python3
"""Score a completed Bora field on its annotated pixels and whole groups.

The target is deliberately outside the predictor case JSON. The diagnostic
coarse baseline uses the hidden expert-to-anonymous key and is not deployable.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile
from scipy import ndimage as ndi

from build_bora_case import sha256, TILE


def _metrics(prediction: np.ndarray, reference: np.ndarray, class_ids: list[int]) -> dict:
    domain = reference > 0
    if not domain.any():
        raise ValueError("No annotated pixels")
    per_class = {}
    for code in class_ids:
        truth = (reference == code) & domain
        pred = (prediction == code) & domain
        intersection = int(np.count_nonzero(truth & pred))
        union = int(np.count_nonzero(truth | pred))
        per_class[str(code)] = dict(reference_pixels=int(truth.sum()),
                                    predicted_pixels=int(pred.sum()),
                                    intersection=intersection, union=union,
                                    iou=intersection / union if union else None,
                                    dice=2 * intersection / (int(truth.sum()) + int(pred.sum()))
                                    if truth.any() or pred.any() else None)
    observed = [per_class[str(c)]["iou"] for c in class_ids
                if per_class[str(c)]["reference_pixels"]]
    return dict(annotated_pixels=int(domain.sum()),
                pixel_accuracy=float(np.mean(prediction[domain] == reference[domain])),
                macro_iou=float(np.mean(observed)),
                macro_dice=float(np.mean([per_class[str(c)]["dice"] for c in class_ids
                                           if per_class[str(c)]["reference_pixels"]])),
                per_class=per_class)


def _oracle_coarse(case_dir: Path, audit: dict, shape: tuple[int, int]) -> np.ndarray:
    tiles = np.load(case_dir / "anonymous-tiles.npy", allow_pickle=False)
    if tiles.ndim != 2 or tiles.dtype.kind not in "iu":
        raise ValueError("Anonymous tile grid invalid")
    reversed_ids = {int(v): int(audit["target_codes"].get(str(k), {}).get("model_id", 0))
                    for k, v in audit["anonymization"]["source_to_anonymous"].items()}
    baseline = np.zeros(tiles.shape, np.uint8)
    for anonymous, role in reversed_ids.items():
        baseline[tiles == anonymous] = role
    work_shape = audit["working_shape_yx"]
    working = np.repeat(np.repeat(baseline, TILE, axis=0), TILE, axis=1)[
        :work_shape[0], :work_shape[1]]
    return np.asarray(Image.fromarray(working).resize(shape[::-1], Image.Resampling.NEAREST))


def _interface(mask: np.ndarray, a: int, b: int) -> np.ndarray:
    aa, bb = mask == a, mask == b
    edge = np.zeros(mask.shape, bool)
    horizontal = (aa[:, :-1] & bb[:, 1:]) | (bb[:, :-1] & aa[:, 1:])
    vertical = (aa[:-1, :] & bb[1:, :]) | (bb[:-1, :] & aa[1:, :])
    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    edge[:-1, :] |= vertical
    edge[1:, :] |= vertical
    return edge


def _boundary_f1(pred: np.ndarray, ref: np.ndarray, tumor: int, stroma: int,
                 mpp_xy: list[float], tolerance_um=8.0) -> dict:
    domain = ref > 0
    actual = _interface(ref, tumor, stroma)
    forecast = _interface(np.where(domain, pred, 0), tumor, stroma)
    if not actual.any():
        return dict(reference_interface_pixels=0, predicted_interface_pixels=int(forecast.sum()),
                    f1=None)
    if not forecast.any():
        return dict(reference_interface_pixels=int(actual.sum()), predicted_interface_pixels=0,
                    precision=0., recall=0., f1=0.)
    spacing = (mpp_xy[1], mpp_xy[0])
    to_reference = ndi.distance_transform_edt(~actual, sampling=spacing)
    to_prediction = ndi.distance_transform_edt(~forecast, sampling=spacing)
    precision = float(np.mean(to_reference[forecast] <= tolerance_um))
    recall = float(np.mean(to_prediction[actual] <= tolerance_um))
    return dict(reference_interface_pixels=int(actual.sum()),
                predicted_interface_pixels=int(forecast.sum()), precision=precision,
                recall=recall, f1=2 * precision * recall / (precision + recall)
                if precision + recall else 0., tolerance_um=tolerance_um)


def _group_scores(result: Path, case_dir: Path, audit: dict, reference: np.ndarray,
                  report: dict) -> list[dict]:
    groups = tifffile.imread(result / "groups.ome.tif")
    if groups.shape != reference.shape:
        raise ValueError("Group raster/native target geometry differs")
    tiles = np.load(case_dir / "anonymous-tiles.npy", allow_pickle=False)
    predictions = report["groups"]
    if len(predictions) != int(groups.max()):
        raise ValueError("Group count differs from report")
    output = []
    for compact_id, item in enumerate(predictions, 1):
        region = groups == compact_id
        values = reference[region]
        annotated = values[values > 0]
        count = int(np.count_nonzero(tiles == item["anonymous_id"]))
        if not len(annotated):
            output.append(dict(group_id=compact_id, anonymous_id=item["anonymous_id"],
                               input_tiles=count, eligible_for_accuracy=False,
                               reason="unannotated"))
            continue
        counts = np.bincount(annotated, minlength=256)
        winner = int(np.argmax(counts[1:]) + 1)
        purity = float(counts[winner] / len(annotated))
        coverage = float(len(annotated) / len(values))
        eligible = purity >= .8 and coverage >= .5
        output.append(dict(group_id=compact_id, anonymous_id=item["anonymous_id"],
                           input_tiles=count, native_pixels=int(region.sum()),
                           coverage=coverage, purity=purity, expert_class=winner,
                           predicted_class=int(item["predicted_class"]),
                           correct=bool(item["predicted_class"] == winner) if eligible else None,
                           eligible_for_accuracy=bool(eligible)))
    return output


def score(case_dir: Path, result: Path) -> dict:
    case_dir, result = case_dir.resolve(), result.resolve()
    audit = json.loads((case_dir / "CASE_COMPLETE.json").read_text())
    report = json.loads((result / "report.json").read_text())
    if sha256(case_dir / "case.json") != audit["case_sha256"]:
        raise ValueError("Predictor case manifest changed")
    if report["provenance"]["input_sha256"]["case_manifest"] != audit["case_sha256"]:
        raise ValueError("Bora output belongs to a different input case")
    for file, expected in report["output_sha256"].items():
        if sha256(result / file) != expected:
            raise ValueError("Bora output file hash differs: " + file)
    pred = tifffile.imread(result / "mask.ome.tif")
    ref = tifffile.imread(case_dir / "expert-target-private.tif")
    if pred.shape != ref.shape or pred.dtype.kind not in "iu" or ref.dtype.kind not in "iu":
        raise ValueError("Prediction and native expert target differ")
    roles = audit["canonical_role_ids"]
    codes = sorted(int(x) for x in roles.values())
    coarse_prediction = _oracle_coarse(case_dir, audit, ref.shape)
    if audit["dataset"] == "ocelot":
        source_mask = Path(audit["expert_source_path"])
        if sha256(source_mask) != audit["expert_sha256"]:
            raise ValueError("OCELOT source mask checksum differs")
        with Image.open(source_mask) as image:
            raw = np.asarray(image)
        if raw.shape != ref.shape:
            raise ValueError("OCELOT expert mask geometry differs")
        tumor = audit["canonical_role_ids"]["tumor"]
        positive = raw == 2
        weak_negative = raw == 1
        if not positive.any() or not weak_negative.any():
            raise ValueError("OCELOT case lacks tumor or broad non-cancer comparator")
        def binary_summary(value):
            return dict(expert_tumor_recall=float(np.mean(value[positive] == tumor)),
                        weak_negative_tumor_fraction=float(np.mean(value[weak_negative] == tumor)),
                        expert_tumor_pixels=int(positive.sum()),
                        weak_negative_pixels=int(weak_negative.sum()),
                        macro_iou=None, pixel_accuracy=None)
        main, oracle = binary_summary(pred), binary_summary(coarse_prediction)
        endpoint = "expert_tumor_recall_plus_automatic_broad_negative_control"
    else:
        main = _metrics(pred, ref, codes)
        oracle = _metrics(coarse_prediction, ref, codes)
        endpoint = "expert_multiclass_annotated_pixels"
    tumor, stroma = roles["tumor"], roles["stroma"]
    group = _group_scores(result, case_dir, audit, ref, report)
    included = [x for x in group if x["eligible_for_accuracy"]]
    return dict(format="bora-full-benchmark-case-score-v1", case_id=audit["case_id"],
                dataset=audit["dataset"], split=audit["split"],
                source_group=audit["source_group"],
                predictor_case_sha256=audit["case_sha256"],
                result_report_sha256=sha256(result / "report.json"),
                final=main, oracle_expert_decoded_coarse=oracle,
                endpoint=endpoint,
                tumor_stroma_interface=_boundary_f1(pred, ref, tumor, stroma,
                                                     audit["native_mpp_xy"]),
                groups=group, eligible_groups=len(included),
                group_accuracy=float(np.mean([x["correct"] for x in included]))
                if included else None,
                group_tile_count_bins={name: dict(count=len(rows),
                    accuracy=float(np.mean([r["correct"] for r in rows])) if rows else None)
                    for name, rows in (("1-4", [x for x in included if x["input_tiles"] < 5]),
                                       ("5-9", [x for x in included if 5 <= x["input_tiles"] < 10]),
                                       ("10-19", [x for x in included if 10 <= x["input_tiles"] < 20]),
                                       ("20+", [x for x in included if x["input_tiles"] >= 20]))},
                scored_on_annotated_pixels_only=True,
                baseline_uses_hidden_expert_class_key=True,
                independent_test_claim=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    value = score(args.case, args.result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(args.output)
    print(json.dumps(dict(case_id=value["case_id"],
                          macro_iou=value["final"].get("macro_iou"),
                          group_accuracy=value["group_accuracy"])))


if __name__ == "__main__":
    main()
