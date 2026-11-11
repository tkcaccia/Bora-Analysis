#!/usr/bin/env python3
"""Fill only uncovered geometry beneath tiny raster gaps in a global vector."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from rasterio.features import rasterize, shapes
from rasterio.transform import Affine
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import unary_union

ROOT = Path("/mnt/sata_ssd/ChatGPT/Bora")
DATA = Path("/mnt/sata_ssd/ChatGPT/Bora-Data")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

V30 = ROOT / "migration/benchmark2-supervised-final-2026-11-11-v30/supervised_wsi_pipeline.py"
spec = importlib.util.spec_from_file_location("v30_pipeline", V30)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
OUT = DATA / "benchmark2-supervised-final-2026-11-11-v32"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_failed_source(slug: str, context: dict) -> tuple[Path, dict]:
    source = DATA / "benchmark2-supervised-final-2026-11-11-v30" / slug
    state = json.loads((source / "STATE.json").read_text())
    if (state["status"] != "FAILED_PRESERVE_PARTIAL" or
            "sub-2000 um2 component" not in state["error"]):
        raise ValueError("This recovery requires the exact audited v30 failure")
    vector_path = source / "global-vectors/GLOBAL_VECTOR_COMPLETE.json"
    vector = json.loads(vector_path.read_text())
    if (vector["format"] != "bora-wsi-global-vector-v1" or
            vector["source_image_sha256"] != context["source"]["sha256"] or
            vector["minimum_area_um2"] != 2000):
        raise ValueError("Failed vector receipt differs")
    for relative, expected in vector["output_sha256"].items():
        if sha(source / "global-vectors" / relative) != expected:
            raise ValueError("Failed vector bytes differ: " + relative)
    assembly = json.loads((source / "assembly/ASSEMBLY_COMPLETE.json").read_text())
    if (assembly["source_image_sha256"] != context["source"]["sha256"] or
            assembly["semantic_mask_sha256"] != vector["source_semantic_mask_sha256"]):
        raise ValueError("Failed vector source binding differs")
    frame = vector["native_shape_yx"]
    if (source / "reviewed-support.uint8.dat").stat().st_size != frame[0] * frame[1]:
        raise ValueError("Failed source support size differs")
    classes = json.loads((source / "CLASS_SELECTION.json").read_text())["retained_class_ids"]
    if sorted(classes) != sorted(json.loads((source / "assembly/ASSEMBLY_COMPLETE.json").read_text())["class_ids"]):
        raise ValueError("Failed source class selection differs")
    return source, vector


def patched_working_geojson(working_path: Path, old: np.memmap,
                            cleaned: np.memmap, destination: Path,
                            mpp_xy: tuple[float, float]) -> dict:
    """Assign empty vector slivers in changed pixel cells without cutting tissue.

    A pixel-cell patch that subtracts *all* existing polygons can split a thin
    legitimate polygon into a subthreshold part (observed in failed v31).
    Intersect each changed region with the old coverage and add only its
    previously unoccupied remainder to the adjacent predicted class.
    """
    data = json.loads(working_path.read_text())
    grouped = {}
    templates = {}
    for feature in data["features"]:
        value = int(feature["properties"]["value"])
        grouped.setdefault(value, []).append(shape(feature["geometry"]))
        templates.setdefault(value, feature)
    geometries = {value: unary_union(parts) for value, parts in grouped.items()}
    patch_groups = {}
    for y in range(0, old.shape[0], 512):
        previous = np.asarray(old[y:y+512])
        updated = np.asarray(cleaned[y:y+512])
        changed = previous != updated
        if not changed.any():
            continue
        for raw, value in shapes(updated.astype(np.int16), mask=changed,
                                 connectivity=4, transform=Affine(1, 0, 0, 0, 1, y)):
            patch_groups.setdefault(int(value), []).append(shape(raw))
    if not patch_groups:
        raise ValueError("Recovery found no changed raster pixels")
    residuals = {}
    patch_count = 0
    for value, regions in patch_groups.items():
        if value == 0 or value not in geometries:
            raise ValueError("Tiny-hole repair must assign an existing tissue class")
        for region in regions:
            occupied = unary_union([g.intersection(region) for g in geometries.values()
                                    if g.intersects(region)])
            empty = region.difference(occupied)
            if empty.is_empty:
                raise ValueError("Changed raster cell has no unoccupied vector gap")
            residuals.setdefault(value, []).append(empty)
            patch_count += 1
    for value, gaps in residuals.items():
        geometries[value] = geometries[value].union(unary_union(gaps))
    area = mpp_xy[0] * mpp_xy[1]
    minimum = float("inf")
    repaired_features = []
    for value, geometry in sorted(geometries.items()):
        feature = templates[value]
        if not geometry.is_valid:
            raise ValueError("Patch produced invalid class geometry")
        for polygon in ([geometry] if geometry.geom_type == "Polygon" else geometry.geoms):
            minimum = min(minimum, polygon.area * area)
            if polygon.area * area < 2000 - 1e-4:
                raise ValueError("Patch created subthreshold polygon")
            for ring in polygon.interiors:
                if Polygon(ring).area * area < 2000 - 1e-4:
                    raise ValueError("Patch created subthreshold polygon hole")
        feature["geometry"] = mapping(geometry)
        feature["properties"]["area_pixels"] = float(geometry.area)
        feature["properties"]["area_um2"] = float(geometry.area * area)
        feature["properties"]["component_id"] = f"supervised_class_{value:02d}_global"
        feature["properties"]["component_index"] = 0
        repaired_features.append(feature)
    data["features"] = repaired_features
    data["raster_tie_repair"] = {
        "source": "v30 exact global vector", "patches": patch_count,
        "minimum_area_um2": 2000,
        "method": "union only previously unoccupied geometry within changed pixel cells"}
    prior.atomic(destination, data)
    return {"patches": patch_count, "minimum_polygon_um2": minimum,
            "feature_count": len(data["features"])}


def audit_working_raster(working: Path, expected: np.memmap, temporary: Path) -> int:
    data = json.loads(working.read_text())
    pairs = [(feature["geometry"], int(feature["properties"]["value"]))
             for feature in data["features"]]
    rendered = np.memmap(temporary, mode="w+", dtype=np.uint8, shape=expected.shape)
    rasterize(pairs, out=rendered, transform=Affine.identity(), fill=0,
              dtype=np.uint8, all_touched=False)
    mismatch = sum(int(np.count_nonzero(rendered[y:y+512] != expected[y:y+512]))
                   for y in range(0, expected.shape[0], 512))
    del rendered
    temporary.unlink()
    if mismatch:
        raise ValueError(f"Patched working vector raster differs by {mismatch} pixels")
    return mismatch


def main(slug: str) -> None:
    start = time.monotonic()
    context = prior.inputs(slug)
    source, vector = verify_failed_source(slug, context)
    out = OUT / slug
    if out.exists():
        raise FileExistsError("Preserve recovery evidence; use another isolated version")
    out.mkdir(parents=True)
    state = out / "STATE.json"
    try:
        prior.atomic(state, {"status": "RUNNING", "stage": "exact_raster_hole_repair"})
        frame = tuple(vector["native_shape_yx"])
        old_path = source / "global-vectors/exact/mask.uint8.dat"
        old = np.memmap(old_path, mode="r", dtype=np.uint8, shape=frame)
        support_path = source / "reviewed-support.uint8.dat"
        support = np.memmap(support_path, mode="r", dtype=np.uint8, shape=frame)
        mpp = tuple(vector["native_mpp_xy"])
        cleaned_path = out / "supervised-bora.final-2000um2.uint8.dat"
        cleaned, repair = prior.area_repair(old, support, cleaned_path, mpp, 2000)
        changed = sum(int(np.count_nonzero(cleaned[y:y+512] != old[y:y+512]))
                      for y in range(0, frame[0], 512))
        if changed <= 0 or changed > .001 * int(np.count_nonzero(old)):
            raise ValueError("Recovery changed an implausible fraction of the global mask")
        prior.atomic(state, {"status": "RUNNING", "stage": "pixel_edge_vector_patch",
                             "changed_pixels": changed})
        working = out / "supervised-bora.working.geojson"
        patch = patched_working_geojson(source / "global-vectors/exact/mask.geojson",
                                        old, cleaned, working, mpp)
        audit_working_raster(working, cleaned, out / "working-roundtrip.partial.dat")
        native = out / "supervised-bora.native.geojson"
        native_report = prior.native_geojson(working, context, native)
        tiff = out / "supervised-bora.mask-0p5um.tif"
        prior.write_mask_tiff(cleaned_path, frame, tiff)
        prior.atomic(state, {"status": "RUNNING", "stage": "independent_final_audit"})
        audit = prior.audit_final(cleaned_path, support_path, native, context)
        result = {"status": "SUPERVISED_BORA_FULL_SLIDE_VECTOR_VERIFIED", "slug": slug,
                  "recovery_of_failed_version": "v30", "failure_state_sha256": sha(source / "STATE.json"),
                  "failed_vector_receipt_sha256": sha(source / "global-vectors/GLOBAL_VECTOR_COMPLETE.json"),
                  "source_image_sha256": context["source"]["sha256"],
                  "score_sha256": sha(context["score_path"]),
                  "source_bora_geojson_sha256": sha(context["geo_path"]),
                  "training_model_receipt_sha256": sha(prior.MODEL),
                  "training_presence_sha256": sha(prior.PRESENCE),
                  "ontology_sha256": sha(prior.ONTOLOGY),
                  "class_ids": json.loads((source / "CLASS_SELECTION.json").read_text())["retained_class_ids"],
                  "class_selection": json.loads((source / "CLASS_SELECTION.json").read_text()),
                  "contour_smoothing_source_sha256": sha(source / "contour-6um.npy"),
                  "initial_global_vector": {"input_regions": vector["input_regions"],
                                            "exact_component_count": vector["exact_component_count"]},
                  "final_pixel_repair": repair, "changed_pixels_from_v30_exact": changed,
                  "vector_patch": patch, "native_vector": native_report,
                  "independent_geometry_audit": audit,
                  "calibrated_shape_yx": list(frame), "calibrated_mpp_xy": list(mpp),
                  "native_mpp_xy": context["geoj"]["native_mpp_xy"],
                  "output_sha256": {"mask_tiff": sha(tiff), "native_geojson": sha(native),
                                    "exact_mask": sha(cleaned_path), "working_geojson": sha(working)},
                  "source_code_sha256": sha(Path(__file__)), "gpu_used": False,
                  "whole_slide_expert_accuracy": None, "pm016_included": False,
                  "elapsed_seconds": time.monotonic()-start}
        prior.atomic(out / "FULL_SLIDE_VERIFIED.json", result)
        prior.atomic(state, {"status": "COMPLETE", "receipt_sha256": sha(out / "FULL_SLIDE_VERIFIED.json")})
        print("RECOVERY_VERIFIED", slug, "holes_changed", changed, flush=True)
    except BaseException as exc:
        prior.atomic(state, {"status": "FAILED_PRESERVE_PARTIAL", "error": repr(exc),
                             "elapsed_seconds": time.monotonic()-start})
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: repair_exact_vector.py PUBLIC_SLUG")
    main(sys.argv[1])
