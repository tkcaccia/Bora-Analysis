"""Compare native-mask polygonizers on identical saved tissue masks.

Run in an environment with Bora main, Shapely, Rasterio, NumPy and tifffile:

    python benchmark/geojson_resource_benchmark.py \
      --mask /path/to/case-a/mask.ome.tif --mask /path/to/case-b/mask.ome.tif \
      --output /path/to/geojson_resources.json

Each method runs in a fresh process. Wall time excludes Python imports and
metric calculation; peak RSS includes the interpreter and imported libraries.
The rasterio baseline is a fidelity reference, not a complete Bora exporter.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter


METHODS = ("rasterio_lossless", "coverage_2px", "bora_complete")


def _rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024 if platform.system() == "Darwin" else 1024)


def worker(args: argparse.Namespace) -> None:
    import numpy as np
    import tifffile
    from rasterio.features import rasterize, shapes
    from shapely import make_valid
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union
    from bora.compact_geojson import boundary_stats
    from bora.multicpu_geojson import convert
    from bora.pipeline import export_mask

    mask_path = args.mask[0]
    mask = tifffile.imread(mask_path)
    if mask.ndim != 2 or mask.dtype.kind not in "iu":
        raise ValueError("Expected one native 2D integer tissue mask")
    values = [int(value) for value in np.unique(mask) if value > 0]
    mpp = [args.mpp_x, args.mpp_y]
    baseline = _rss_mb()
    folder = args.folder
    folder.mkdir(parents=True, exist_ok=True)
    vector_path = folder / "mask.geojson"
    start = perf_counter()
    if args.method == "rasterio_lossless":
        buckets: dict[int, list] = {}
        for geometry, value in shapes(mask, mask=mask > 0, connectivity=4):
            candidate = shape(geometry)
            buckets.setdefault(int(value), []).append(
                candidate if candidate.is_valid else make_valid(candidate)
            )
        features = [
            {"type": "Feature", "geometry": mapping(unary_union(buckets[value])),
             "properties": {"value": value}}
            for value in sorted(buckets)
        ]
        vector_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    elif args.method == "coverage_2px":
        convert(mask_path, vector_path, page=0, min_area=20, simplify=2,
                workers=1, striped_native=False)
    elif args.method == "bora_complete":
        export_mask(mask, folder, mpp, {value: f"class_{value}" for value in values})
    else:
        raise ValueError(args.method)
    seconds = perf_counter() - start
    peak = _rss_mb()

    data = json.loads(vector_path.read_text())
    features = data["features"]
    pairs = [(f["geometry"], int(f["properties"]["value"])) for f in features]
    rerendered = rasterize(pairs, out_shape=mask.shape, fill=0, dtype=np.uint8)
    per_class_iou = {}
    for value in values:
        a = mask == value
        b = rerendered == value
        per_class_iou[str(value)] = float(np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b)))
    geometries = [shape(f["geometry"]) for f in features]
    overlap = sum(geometries[i].intersection(geometries[j]).area
                  for i in range(len(geometries)) for j in range(i))
    result = {
        "method": args.method,
        "image": mask_path.parent.name,
        "shape_yx": list(mask.shape),
        "native_mpp_xy": mpp,
        "wall_seconds": seconds,
        "rss_after_import_mb": baseline,
        "peak_rss_mb": peak,
        "additional_peak_rss_mb": max(0, peak - baseline),
        "geojson_bytes": vector_path.stat().st_size,
        "features": len(features),
        "segments": boundary_stats(data)["boundary_segments"],
        "changed_pixels_vs_input_mask": int(np.count_nonzero(mask != rerendered)),
        "minimum_class_iou_vs_input_mask": min(per_class_iou.values(), default=1.0),
        "per_class_iou_vs_input_mask": per_class_iou,
        "all_classes_preserved": set(values) == {int(f["properties"]["value"]) for f in features},
        "all_geometries_valid": all(g.is_valid for g in geometries),
        "interclass_overlap_area_pixels": float(overlap),
    }
    print(json.dumps(result, separators=(",", ":")))


def driver(args: argparse.Namespace) -> None:
    masks = [Path(p).resolve() for p in args.mask]
    records = []
    for mask in masks:
        companion = mask.with_name("mask.geojson")
        if args.mpp_x is not None and args.mpp_y is not None:
            mpp = [args.mpp_x, args.mpp_y]
        elif companion.exists():
            mpp = json.loads(companion.read_text())["native_mpp_xy"]
        else:
            raise ValueError(f"Supply --mpp-x and --mpp-y or a companion mask.geojson for {mask}")
        for method in METHODS:
            for repeat in range(args.repeats):
                with tempfile.TemporaryDirectory(prefix="bora-vector-resource-") as temporary:
                    command = [sys.executable]
                    if args.no_site:
                        command.append("-S")
                    command += [str(Path(__file__).resolve()),
                               "--worker", "--mask", str(mask), "--method", method,
                               "--mpp-x", str(mpp[0]), "--mpp-y", str(mpp[1]),
                               "--folder", temporary]
                    proc = subprocess.run(command, text=True, capture_output=True,
                                          timeout=args.timeout_seconds, check=False)
                    if proc.returncode:
                        raise RuntimeError(f"{mask.parent.name} {method} failed:\n{proc.stderr}")
                    record = json.loads(proc.stdout)
                    record["repeat"] = repeat + 1
                    records.append(record)
                    print(f"{record['image']} {method} {repeat + 1}/{args.repeats}: "
                          f"{record['wall_seconds']:.3f}s, {record['peak_rss_mb']:.1f} MB", flush=True)
    summary = {}
    for method in METHODS:
        items = [record for record in records if record["method"] == method]
        summary[method] = {
            "runs": len(items),
            "median_wall_seconds": statistics.median(r["wall_seconds"] for r in items),
            "median_peak_rss_mb": statistics.median(r["peak_rss_mb"] for r in items),
            "largest_peak_rss_mb": max(r["peak_rss_mb"] for r in items),
            "median_geojson_bytes": statistics.median(r["geojson_bytes"] for r in items),
            "median_segments": statistics.median(r["segments"] for r in items),
            "largest_changed_pixels": max(r["changed_pixels_vs_input_mask"] for r in items),
            "lowest_class_iou_vs_input_mask": min(r["minimum_class_iou_vs_input_mask"] for r in items),
        }
    result = {
        "scope": f"{len(masks)} saved pre-export tissue predictions; resource comparison only, not expert benchmark accuracy",
        "methods": {
            "rasterio_lossless": "Rasterio polygonization and same-label union; no Bora cleanup or simplification",
            "coverage_2px": "Bora initial coverage polygonizer at min_area=20 and simplify=2, one worker",
            "bora_complete": "Current Bora cleanup, coverage converter, calibrated compact fitting, final region checks and OME mask write",
        },
        "timing_contract": "Fresh process per run; wall time excludes imports and metric calculation; peak RSS includes imports; methods were run serially",
        "python": sys.version,
        "platform": platform.platform(),
        "repeats_per_image_method": args.repeats,
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mask", type=Path, action="append", required=True)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--mpp-x", type=float)
    parser.add_argument("--mpp-y", type=float)
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--no-site", action="store_true", help="Suppress site imports for an isolated source checkout")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    args = parser.parse_args()
    if args.worker:
        if len(args.mask) != 1 or not args.method or not args.folder:
            parser.error("Worker needs one mask, method and folder")
        worker(args)
    else:
        if not args.output or args.repeats < 1:
            parser.error("Driver needs an output file and positive repeat count")
        driver(args)


if __name__ == "__main__":
    main()
