"""Build a display-only, physically smoothed tumor_087 inspection vector.

The source vector and scored 4 µm mask are immutable. This file is never used
for metrics. The 20,000 µm² cutoff applies to *both* displayed classes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import Polygon, mapping, shape


EXPECTED_SOURCE_SHA = "635805e71adae2c473f19352ab572a855554023ed687b4e0e3db06935ecd75a4"
MIN_AREA_UM2 = 20_000.0
SAMPLE_SPACING_UM = 4.0
SMOOTH_SIGMA_UM = 12.0
SIMPLIFY_TOLERANCE_UM = 0.5
NATIVE_MPP_UM = 0.5


def smooth_ring(coords: list[list[float]]) -> list[list[float]]:
    points = np.asarray(coords, dtype=np.float64)
    if len(points) < 5:
        return coords
    if np.array_equal(points[0], points[-1]):
        points = points[:-1]
    closed = np.vstack((points, points[0]))
    segments = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    distances = np.r_[0.0, np.cumsum(segments)]
    total = distances[-1]
    step_px = SAMPLE_SPACING_UM / NATIVE_MPP_UM
    count = max(8, int(np.ceil(total / step_px)))
    locations = np.arange(count, dtype=np.float64) * (total / count)
    x = np.interp(locations, distances, closed[:, 0])
    y = np.interp(locations, distances, closed[:, 1])
    sigma_samples = SMOOTH_SIGMA_UM / (total * NATIVE_MPP_UM / count)
    smoothed = np.column_stack((
        gaussian_filter1d(x, sigma_samples, mode="wrap"),
        gaussian_filter1d(y, sigma_samples, mode="wrap"),
    ))
    return np.vstack((smoothed, smoothed[0])).tolist()


def smooth_polygon(geometry: dict) -> dict:
    if geometry["type"] != "Polygon":
        raise ValueError("Expected one source Polygon per component")
    rings = geometry["coordinates"]
    candidate = Polygon(smooth_ring(rings[0]), [smooth_ring(r) for r in rings[1:]])
    if not candidate.is_valid:
        candidate = candidate.buffer(0)
    candidate = candidate.simplify(SIMPLIFY_TOLERANCE_UM / NATIVE_MPP_UM, preserve_topology=True)
    if candidate.is_empty or not candidate.is_valid:
        raise ValueError("Smoothing produced invalid or empty geometry")
    source = shape(geometry)
    area_change = abs(candidate.area - source.area) / source.area
    if area_change > 0.10:
        raise ValueError(f"Component area drift {area_change:.3%} exceeds 10%")
    return mapping(candidate), area_change


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    source = args.site / "camelyon16/inspector-vectors/tumor_087-bora.geojson.gz"
    target = args.site / "camelyon16/inspector-vectors/tumor_087-bora-review.geojson.gz"
    source_bytes = source.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == EXPECTED_SOURCE_SHA
    vector = json.loads(gzip.decompress(source_bytes))
    assert vector["native_mpp_xy"] == [NATIVE_MPP_UM, NATIVE_MPP_UM]
    out = {key: value for key, value in vector.items() if key != "features"}
    out["display_only"] = True
    out["source_saved_geojson_sha256"] = EXPECTED_SOURCE_SHA
    out["display_processing"] = {
        "minimum_component_area_um2": MIN_AREA_UM2,
        "classes": [1, 2],
        "boundary_resample_spacing_um": SAMPLE_SPACING_UM,
        "boundary_gaussian_sigma_um": SMOOTH_SIGMA_UM,
        "final_simplification_tolerance_um": SIMPLIFY_TOLERANCE_UM,
        "metric_input": False,
    }
    out["features"] = []
    omitted = {1: {"count": 0, "area_um2": 0.0}, 2: {"count": 0, "area_um2": 0.0}}
    drift = []
    for feature in vector["features"]:
        value = feature["properties"]["value"]
        assert value in omitted
        area = feature["properties"]["area_um2"]
        if area < MIN_AREA_UM2:
            omitted[value]["count"] += 1
            omitted[value]["area_um2"] += area
            continue
        geometry, area_change = smooth_polygon(feature["geometry"])
        updated = dict(feature)
        updated["geometry"] = geometry
        updated["properties"] = dict(feature["properties"])
        updated["properties"]["display_only"] = True
        updated["properties"]["display_area_um2"] = shape(geometry).area * NATIVE_MPP_UM**2
        out["features"].append(updated)
        drift.append(area_change)
    raw = json.dumps(out, separators=(",", ":"), allow_nan=False).encode()
    with target.open("wb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0, filename="") as compressed:
            compressed.write(raw)
    receipt = {
        "status": "VERIFIED_DISPLAY_ONLY",
        "source_path": str(source),
        "source_sha256": EXPECTED_SOURCE_SHA,
        "output_path": str(target),
        "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "output_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "source_feature_count": len(vector["features"]),
        "display_feature_count": len(out["features"]),
        "omitted": omitted,
        "maximum_relative_component_area_change": max(drift),
        "mean_relative_component_area_change": sum(drift) / len(drift),
        "display_processing": out["display_processing"],
        "metrics_unchanged": True,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
