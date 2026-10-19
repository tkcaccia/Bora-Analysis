"""Independently check the native WSI mask against its windowed GeoJSON.

The export is too large to rasterize in one allocation, so this verifies
selected cores directly and checks every core's topology/area audit.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine


def ring_area(ring):
    vertices = np.asarray(ring, dtype=np.float64)
    return abs(float(np.dot(vertices[:-1, 0], vertices[1:, 1]) -
                     np.dot(vertices[:-1, 1], vertices[1:, 0]))) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="*", default=[0, 13, 54, 99, 147, 193, 227])
    parser.add_argument("--all-windows", action="store_true",
                        help="Rasterize all 228 native windows, rather than selected samples")
    args = parser.parse_args()
    summary = json.loads((args.result / "VECTOR_COMPLETE.json").read_text())
    height, width = summary["native_shape_yx"]
    mask = np.memmap(args.result / "classified.final.dat", mode="r", dtype=np.uint8,
                     shape=(height, width))
    checks = sorted((args.result / "vector-windows").glob("[0-9][0-9][0-9].json"))
    if len(checks) != summary["windows"]:
        raise AssertionError(f"Found {len(checks)} of {summary['windows']} audits")
    for path in checks:
        audit = json.loads(path.read_text())
        cleanup = audit.get("small_area_cleanup", {})
        if (cleanup.get("positive_polygon_parts_below_threshold", 0) or
                cleanup.get("polygon_holes_below_threshold", 0)):
            raise AssertionError(f"Sub-20-pixel region in {path}")
        if audit.get("minimum_class_iou", 1) < .97:
            raise AssertionError(f"Shape agreement below 97% in {path}")
        if audit.get("changed_tissue_area_fraction", 0) > .015:
            raise AssertionError(f"Tissue area changed >1.5% in {path}")
    samples = []
    indices = range(summary["windows"]) if args.all_windows else args.indices
    for index in indices:
        if not 0 <= index < summary["windows"]:
            raise ValueError(f"Index {index} outside export")
        report = json.loads((args.result / "vector-windows" / f"{index:03d}.json").read_text())
        x, y, w, h = report["core_xywh"]
        features = json.loads((args.result / "vector-windows" / f"{index:03d}.geojson").read_text())
        for feature in features:
            geometry = feature["geometry"]
            polygons = ([geometry["coordinates"]] if geometry["type"] == "Polygon"
                        else geometry["coordinates"])
            for polygon in polygons:
                area = ring_area(polygon[0]) - sum(ring_area(ring) for ring in polygon[1:])
                if area < 20:
                    raise AssertionError(f"Sub-20-pixel polygon in window {index}")
                if any(ring_area(ring) < 20 for ring in polygon[1:]):
                    raise AssertionError(f"Sub-20-pixel hole in window {index}")
        rebuilt = rasterize(((feature["geometry"], int(feature["properties"]["value"]))
                             for feature in features), out_shape=(h, w), dtype=np.uint8,
                            transform=Affine(1, 0, x, 0, 1, y))
        observed = np.asarray(mask[y:y+h, x:x+w])
        differing = int(np.count_nonzero(rebuilt != observed))
        if differing:
            raise AssertionError(f"GeoJSON/mask mismatch in window {index}: {differing} pixels")
        samples.append({"window": index, "core_xywh": [x, y, w, h],
                        "pixels": w*h, "different_pixels": 0,
                        "features": len(features)})
    print(json.dumps({"all_windows_area_and_shape_audited": len(checks),
                      "exact_raster_vector_samples": samples}, indent=2))


if __name__ == "__main__":
    main()
