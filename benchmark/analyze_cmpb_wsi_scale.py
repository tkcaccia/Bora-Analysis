"""Summarize the saved colon whole-slide scale test without rerunning inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(study: Path) -> dict:
    score = json.loads((study / "EVALUATION_COMPLETE.json").read_text())
    vector = json.loads((study / "VECTOR_COMPLETE.json").read_text())
    audit = json.loads((study / "wsi-audit.json").read_text())
    shape = score["image_native_shape_yx"]
    assert shape == vector["native_shape_yx"] == [48740, 75250]
    assert score["annotation_used_by_inference"] is False
    assert vector["windows"] == audit["all_windows_area_and_shape_audited"] == 228
    assert len(audit["exact_raster_vector_samples"]) == 228
    assert all(row["different_pixels"] == 0 for row in audit["exact_raster_vector_samples"])
    assert vector["exact_geojson_raster_agreement_per_window"] is True
    assert vector["minimum_area_native_pixels"] == 20
    roles = {key: {"reference_pixels": value["reference_pixels"], "dice": value["dice"], "iou": value["iou"]}
             for key, value in score["metrics"].items()}
    assert sum(x["reference_pixels"] for x in roles.values()) == score["evaluated_pixels"]
    assert len(roles) == 5 and roles["normal_gland"]["dice"] == 0
    return {
        "scope": "one exploratory, partially annotated whole slide; exported readout chosen on this reference",
        "image_shape_yx": shape,
        "native_mpp_xy": vector["native_mpp_xy"],
        "native_pixels": shape[0] * shape[1],
        "annotated_scored_pixels": score["evaluated_pixels"],
        "scored_fraction": score["evaluated_pixels"] / (shape[0] * shape[1]),
        "windows": vector["windows"],
        "macro_dice": score["macro_dice"],
        "macro_iou": score["macro_iou"],
        "roles": roles,
        "vector_seconds": vector["seconds"],
        "vector_features": vector["source_vs_final"]["feature_count"],
        "vector_segments": vector["source_vs_final"]["final_segments"],
        "native_raster_vector_mismatch_pixels": sum(row["different_pixels"] for row in audit["exact_raster_vector_samples"]),
        "minimum_polygon_area_native_pixels": vector["minimum_area_native_pixels"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summarize(args.study), indent=2) + "\n")
