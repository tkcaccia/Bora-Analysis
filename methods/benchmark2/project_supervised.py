#!/usr/bin/env python3
"""Project the frozen training-only Virchow2 tissue model over a WSI cache.

This produces a token-grid prediction, not a Bora/KODAMA segmentation.  Web
rendering and the shared tissue-support gate are separate, versioned stages.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import joblib
import numpy as np
from scipy.ndimage import distance_transform_edt


PROJECT = Path("/mnt/sata_ssd/ChatGPT/Bora")
DATA = Path("/mnt/sata_ssd/ChatGPT/Bora-Data")
HERE = Path(__file__).resolve().parent
MODEL_RECEIPT = PROJECT / "migration/benchmark1-supervised-strategy-search-2026-11-10-v1/FINAL_MODEL_FROZEN.json"
COMPARATOR_SCOPE = PROJECT / "migration/benchmark2-full-wsi-comparators-2026-11-10-v2/SCOPE.json"
CACHE = DATA / "feature-cache/Lion-Data-Bora/analysis"
OUTPUT = DATA / "benchmark2-unified-wsi-panels-2026-11-11-v1/supervised-token-grids"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def linear_form(model, prior: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Collapse PLS plus logistic to one exactly checked linear score."""
    selected, mean, scale, transformer, head, bias = model
    selected = np.asarray(selected, np.intp)

    def score(x: np.ndarray) -> np.ndarray:
        z = (x - mean) / scale
        if transformer is not None:
            z = transformer.transform(z)
        return np.asarray(head.decision_function(z), np.float64)

    zero = np.zeros((1, len(selected)), np.float64)
    basis = np.eye(len(selected), dtype=np.float64)
    intercept = float(score(zero)[0]) + prior * float(bias)
    coefficients = score(basis) - score(zero)[0]
    return selected, np.asarray(coefficients, np.float64), intercept


def source_for_slug(slug: str) -> dict:
    if slug.lower() == "pm016":
        raise ValueError("PM016 is excluded by current project policy")
    scope = json.loads(COMPARATOR_SCOPE.read_text())
    matches = [row for row in scope["slides"] if row["slug"] == slug]
    if len(matches) != 1:
        raise ValueError("Slide is outside the frozen public comparator scope")
    return matches[0]


def cache_for_source(image_sha: str) -> tuple[Path, dict, Path, dict]:
    matches = []
    for receipt_path in CACHE.glob("full1280-*/**/CACHE_COMPLETE.json"):
        if "pm016" in str(receipt_path).lower():
            continue
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != "COMPLETE" or receipt.get("channels") != 1280:
            continue
        manifest_name = Path(receipt["spatial_cache_manifest"]).name
        manifest_path = DATA / "feature-cache/Lion-Data-Bora/preprocessing/input-cache-v1/virchow2_spatial" / manifest_name
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"]["image_sha256"] == image_sha:
            matches.append((receipt_path, receipt, manifest_path, manifest))
    if len(matches) != 1:
        raise ValueError(f"Expected one local full1280 spatial cache, found {len(matches)}")
    return matches[0]


def main(slug: str) -> None:
    item = source_for_slug(slug)
    receipt = json.loads(MODEL_RECEIPT.read_text())
    if (receipt["status"] != "TRAINING_ONLY_WINNER_FROZEN" or receipt["validation_cases_opened"] != 0
            or receipt["training_cases"] != 397):
        raise ValueError("Supervised model freeze differs")
    model_path = Path(receipt["model_path"])
    if sha(model_path) != receipt["model_sha256"]:
        raise ValueError("Supervised model SHA differs")
    cache_path, cache_receipt, manifest_path, manifest = cache_for_source(item["sha256"])
    original = Path(cache_receipt["features"]["1"]["path"])
    feature_path = DATA / "feature-cache/Lion-Data-Bora" / original.relative_to("/media/user/Lion/Data/Bora")
    if not feature_path.is_file() or sha(feature_path) != cache_receipt["features"]["1"]["sha256"]:
        raise ValueError("Local factor-1 spatial feature SHA differs")
    if manifest["outputs"]["factor-1"]["sha256"] != cache_receipt["features"]["1"]["sha256"]:
        raise ValueError("Cache receipts disagree")
    grid = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    if grid.ndim != 3 or grid.shape[2] != 1280 or grid.dtype != np.float16:
        raise ValueError("Virchow2 grid shape or precision differs")
    models = joblib.load(model_path)
    if set(models) != set(range(1, 25)):
        raise ValueError("Frozen model does not have all 24 tissue classes")
    forms = {cid: linear_form(models[cid], receipt["prior_weight"]) for cid in models}
    union = np.unique(np.concatenate([entry[0] for entry in forms.values()]))
    weights = np.zeros((len(union), 24), np.float64)
    offsets = np.zeros(24, np.float64)
    for cid, (selected, coefficients, intercept) in forms.items():
        weights[np.searchsorted(union, selected), cid - 1] = coefficients
        offsets[cid - 1] = intercept
    # Verify the collapsed form against the frozen sklearn objects on real tokens.
    sample_rows = []
    for sample_y in np.linspace(0, grid.shape[0] - 1, min(64, grid.shape[0]), dtype=int):
        row = np.asarray(grid[sample_y], np.float32)
        good = row[np.isfinite(row[:, union]).all(axis=1)][:32 - sum(len(x) for x in sample_rows)]
        if len(good):
            sample_rows.append(good)
        if sum(len(x) for x in sample_rows) >= 32:
            break
    sample = np.asarray(np.concatenate(sample_rows, axis=0), np.float64) if sample_rows else np.empty((0, 1280))
    if len(sample) == 0:
        raise ValueError("No finite real tokens for linear-form check")
    for cid in range(1, 25):
        selected, _, _ = forms[cid]
        original_score = models[cid][4].decision_function(
            models[cid][3].transform((sample[:, selected] - models[cid][1]) / models[cid][2]))
        original_score = original_score + receipt["prior_weight"] * models[cid][-1]
        collapsed = sample[:, union] @ weights[:, cid - 1] + offsets[cid - 1]
        if not np.allclose(original_score, collapsed, atol=2e-4, rtol=2e-5):
            raise ValueError(f"Collapsed model differs for class {cid}")
    out = OUTPUT / slug
    if out.exists():
        raise FileExistsError("Versioned prediction already exists; use a recovery version")
    out.mkdir(parents=True)
    state = out / "STATE.json"
    state.write_text(json.dumps({"status": "RUNNING", "rows_complete": 0}) + "\n")
    started = time.monotonic()
    try:
        finite = np.empty(grid.shape[:2], np.bool_)
        for y in range(grid.shape[0]):
            finite[y] = np.isfinite(np.asarray(grid[y])[:, union]).all(axis=1)
        if not finite.any():
            raise ValueError("No finite tokens")
        if finite.all():
            nearest = None
        else:
            _, indices = distance_transform_edt(~finite, return_indices=True)
            nearest = indices
        labels = np.lib.format.open_memmap(out / "supervised-class-token-grid.npy", mode="w+",
                                             dtype=np.uint8, shape=grid.shape[:2])
        row_step = max(1, min(32, 32768 // grid.shape[1]))
        for y0 in range(0, grid.shape[0], row_step):
            y1 = min(grid.shape[0], y0 + row_step)
            if nearest is None:
                block = np.asarray(grid[y0:y1, :, union], np.float32)
            else:
                block = np.asarray(grid[nearest[0, y0:y1], nearest[1, y0:y1]][:, :, union], np.float32)
            score = block.reshape(-1, len(union)).astype(np.float64) @ weights + offsets
            labels[y0:y1] = (np.argmax(score, axis=1).reshape(y1 - y0, grid.shape[1]) + 1).astype(np.uint8)
            labels.flush()
            state.write_text(json.dumps({"status": "RUNNING", "rows_complete": y1,
                                         "rows_total": grid.shape[0]}) + "\n")
        result_path = out / "supervised-class-token-grid.npy"
        result = {"status": "VERIFIED_TRAINING_ONLY_SUPERVISED_TOKEN_GRID",
                  "slug": slug, "source_sha256": item["sha256"],
                  "feature_path": str(feature_path), "feature_sha256": sha(feature_path),
                  "cache_receipt_sha256": sha(cache_path), "spatial_manifest_sha256": sha(manifest_path),
                  "model_receipt_sha256": sha(MODEL_RECEIPT), "model_sha256": receipt["model_sha256"],
                  "token_shape_yx": list(grid.shape[:2]), "token_stride_pixels": 14,
                  "class_ids_present": [int(x) for x in np.unique(labels)],
                  "nearest_imputed_tokens": int((~finite).sum()),
                  "token_grid_path": str(result_path), "token_grid_sha256": sha(result_path),
                  "inference_seconds": time.monotonic() - started,
                  "no_expert_whole_slide_accuracy": True,
                  "background_support_applied": False}
        (out / "SUPERVISED_TOKEN_GRID_VERIFIED.json").write_text(json.dumps(result, indent=2) + "\n")
        state.write_text(json.dumps({"status": "COMPLETE", "rows_complete": grid.shape[0]}) + "\n")
        print(json.dumps({key: result[key] for key in ("status", "slug", "token_shape_yx",
                                                       "nearest_imputed_tokens", "inference_seconds")}))
    except BaseException as error:
        state.write_text(json.dumps({"status": "FAILED_PRESERVE_PARTIAL", "error": repr(error)}) + "\n")
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: project_supervised.py PUBLIC_SLUG")
    main(sys.argv[1])
