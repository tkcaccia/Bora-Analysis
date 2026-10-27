#!/usr/bin/env python3
"""Register one catalog field for the released Bora 0.1 pipeline.

This is a controlled *expert-derived coarse-input* benchmark. The expert mask
is read here to make anonymous 45-um tile labels and a separate scoring target.
Only predictors are named in case.json, which is passed to Bora. No fine mask
is read by `bora prepare` or `bora refine`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image
import tifffile

ENCODER = "3158645804b69e3f3bc4439d4116edddf0840a72"
WEIGHTS = "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b"
TILE = 90


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def case_row(root: Path, dataset: str, case_id: str) -> dict:
    table = root / "benchmark_catalog/datasets" / dataset / "cases.tsv"
    with table.open(newline="") as stream:
        matches = [r for r in csv.DictReader(stream, delimiter="\t") if r["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError("Catalog case missing or duplicated")
    return matches[0]


def feature_dirs(root: Path, case_id: str) -> tuple[Path, Path]:
    choices = [root / "preprocessing" / f"virchow2-{v}" / "features" / case_id
               for v in ("v1", "v2", "v3", "v5")]
    base = [p for p in choices if (p / "FEATURES_COMPLETE.json").is_file()]
    if len(base) != 1 or not (base[0] / "GPU_CLEANUP.json").is_file():
        raise ValueError("Exactly one complete, cleanup-verified base feature set required")
    extra = root / "preprocessing/full-rerun-2026-10-27/multiscale-v1/features" / case_id
    if not (extra / "FEATURES_COMPLETE.json").is_file() or not (extra / "GPU_CLEANUP.json").is_file():
        raise ValueError("Both context scales and their GPU cleanup receipt required")
    return base[0], extra


def _integer_mask(path: Path, native_shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.ndim != 2 or a.shape != native_shape or a.dtype.kind not in "iu":
        raise ValueError("Expert mask is not aligned integer native image")
    return a


def _working_image(native: np.ndarray, source_mpp_xy: tuple[float, float]):
    h, w = native.shape[:2]
    wh = (max(224, round(w * source_mpp_xy[0] / .5)),
          max(224, round(h * source_mpp_xy[1] / .5)))
    # Reject a narrow field instead of inflating its physical extent to224 px.
    if w * source_mpp_xy[0] / .5 < 224 or h * source_mpp_xy[1] / .5 < 224:
        raise ValueError("Physical field below Bora minimum width")
    working = np.asarray(Image.fromarray(native, "RGB").resize(wh, Image.Resampling.LANCZOS))
    spacing = (w * source_mpp_xy[0] / wh[0], h * source_mpp_xy[1] / wh[1])
    if any(abs(x - .5) > .01 for x in spacing):
        raise ValueError("Working spacing cannot be calibrated near0.5 um/px")
    return working, spacing


def _support(rgb: np.ndarray) -> np.ndarray:
    # Independent H&E white-background gate; no expert mask enters support.
    return (np.min(rgb, axis=2) < 245).astype(np.uint8)


def _coarse(native_mask: np.ndarray, working_shape: tuple[int, int], case_id: str,
            exclude_codes: tuple[int, ...] = (0, 255)):
    h, w = working_shape
    nearest = np.asarray(Image.fromarray(native_mask).resize((w, h), Image.Resampling.NEAREST))
    raw_values = sorted(int(v) for v in np.unique(nearest) if int(v) not in exclude_codes)
    if len(raw_values) < 2 or len(raw_values) > 254:
        raise ValueError("Coarse benchmark requires 2..254 annotated source classes")
    seed = int.from_bytes(hashlib.sha256(case_id.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    anonymous = rng.permutation(np.arange(1, len(raw_values) + 1, dtype=np.uint8))
    lookup = {raw: int(anon) for raw, anon in zip(raw_values, anonymous)}
    grid = np.zeros((math.ceil(h / TILE), math.ceil(w / TILE)), np.uint8)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            patch = nearest[row * TILE:min((row + 1) * TILE, h),
                            col * TILE:min((col + 1) * TILE, w)]
            if min(patch.shape) < .8 * TILE:
                continue
            valid = patch[~np.isin(patch, exclude_codes)]
            if len(valid) < .5 * patch.size:
                continue
            values, counts = np.unique(valid, return_counts=True)
            grid[row, col] = lookup[int(values[np.argmax(counts)])]
    present = sorted(int(v) for v in np.unique(grid) if v)
    if len(present) < 2:
        raise ValueError("Only one anonymous group after physical tiling")
    if int(np.count_nonzero(grid)) < 16:
        raise ValueError("Fewer than16 eligible coarse tiles")
    return grid, dict(source_to_anonymous=lookup, represented_groups=present,
                      source_values=raw_values, anonymization_seed_sha256=hashlib.sha256(case_id.encode()).hexdigest())


def _base_tokens(base: Path, native_shape: tuple[int, int], tile_wh: tuple[int, int],
                 working_shape: tuple[int, int], output: Path) -> dict:
    gh = math.ceil(native_shape[0] / tile_wh[1])
    gw = math.ceil(native_shape[1] / tile_wh[0])
    grid = np.zeros((gh * 16, gw * 16, 1280), np.float16)
    seen = set()
    chunks = sorted(base.glob("chunk-*.npz"))
    if not chunks:
        raise ValueError("No base feature chunks")
    for chunk in chunks:
        note = chunk.with_suffix(".json")
        if not note.is_file() or json.loads(note.read_text())["sha256"] != sha256(chunk):
            raise ValueError("Base feature chunk checksum differs")
        with np.load(chunk, allow_pickle=False) as data:
            tokens, table = data["tokens"], data["tile_table"]
            if tokens.shape != (len(table), 16, 16, 1280) or table.shape[1] != 6:
                raise ValueError("Base chunk shape differs")
            for tile, row in zip(tokens, table):
                index = int(row[0]); x = int(row[1]); y = int(row[2])
                if index in seen or index != (y // tile_wh[1]) * gw + x // tile_wh[0]:
                    raise ValueError("Base tile index/grid mismatch")
                if x % tile_wh[0] or y % tile_wh[1] or not np.isfinite(tile).all():
                    raise ValueError("Base tile origin or values invalid")
                seen.add(index)
                yy, xx = y // tile_wh[1] * 16, x // tile_wh[0] * 16
                grid[yy:yy + 16, xx:xx + 16] = tile
    if not seen:
        raise ValueError("No encoded base tiles")
    trimmed = grid[:math.ceil(working_shape[0] / 14), :math.ceil(working_shape[1] / 14)]
    if trimmed.shape != (math.ceil(working_shape[0] / 14),
                         math.ceil(working_shape[1] / 14), 1280):
        raise ValueError("Base token grid does not cover working image")
    np.save(output, trimmed)
    return dict(encoded_tiles=len(seen), grid_tiles=gh * gw, sha256=sha256(output),
                shape=list(trimmed.shape), omitted_near_white_tiles=gh * gw - len(seen))


def _semantic_target(raw: np.ndarray, dataset: str, mapping: dict, role_ids: dict):
    target = np.zeros(raw.shape, np.uint8)
    used = {}
    for code in np.unique(raw):
        key = str(int(code))
        item = mapping[dataset].get(key)
        if item is None:
            if key not in ("0", "255"):
                raise ValueError(f"Unverified source annotation code: {dataset}/{key}")
            continue
        if item["use"] == "score":
            role = item["canonical_role"]
            target[raw == code] = role_ids[role]
            used[key] = dict(role=role, model_id=role_ids[role])
    return target, used


def build(args: argparse.Namespace) -> dict:
    root = args.root.resolve()
    split_path = root / "preprocessing/full-rerun-2026-10-27/split-v1.json"
    split = json.loads(split_path.read_text())
    assignment = next((r for r in split["assignments"] if r["case_id"] == args.case_id), None)
    if assignment is None:
        raise ValueError("Case absent from immutable benchmark split")
    dataset = assignment["dataset"]
    row = case_row(root, dataset, args.case_id)
    source = root / row["source_image"]
    mask_path = root / row["expert_tissue_mask"]
    base, extra = feature_dirs(root, args.case_id)
    br = json.loads((base / "FEATURES_COMPLETE.json").read_text())
    er = json.loads((extra / "FEATURES_COMPLETE.json").read_text())
    identity = json.loads((base / "INPUTS.json").read_text())
    if (sha256(source) != identity["image_sha256"] or er["source_sha256"] != identity["image_sha256"]
            or er["base_receipt_sha256"] != sha256(base / "FEATURES_COMPLETE.json")
            or br["model_revision"] != ENCODER or br["model_sha256"] != WEIGHTS
            or er["encoder_revision"] != ENCODER or er["encoder_sha256"] != WEIGHTS):
        raise ValueError("Image or Virchow2 checkpoint identity differs")
    with Image.open(source) as im:
        native = np.asarray(im.convert("RGB"))
    if native.dtype != np.uint8:
        raise ValueError("Expected scanner uint8 RGB")
    mask = _integer_mask(mask_path, native.shape[:2])
    source_mpp = (float(row["mpp_x"]), float(row["mpp_y"]))
    work, work_mpp = _working_image(native, source_mpp)
    coarse_exclusions = {"bcss": (0, 7, 15, 255),
                         "ignite": (0, 1, 255)}.get(dataset, (0, 255))
    grid, anon = _coarse(mask, work.shape[:2], args.case_id, coarse_exclusions)
    mapping = json.loads(args.label_map.read_text())["mapping"]
    roles = sorted({v["canonical_role"] for name in ("bcss", "tiger", "ignite")
                    for v in mapping[name].values() if v["use"] == "score"})
    role_ids = {name: idx + 1 for idx, name in enumerate(roles)}
    target, used = _semantic_target(mask, dataset, mapping, role_ids)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".bora-case-") as temporary:
        folder = Path(temporary) / "case"
        folder.mkdir()
        Image.fromarray(work).save(folder / "he-working.png")
        tifffile.imwrite(folder / "tissue-working.tif", _support(work))
        tifffile.imwrite(folder / "tissue-native.tif", _support(native))
        np.save(folder / "anonymous-tiles.npy", grid)
        base_info = _base_tokens(base, native.shape[:2], tuple(br["native_tile_wh"]),
                                 work.shape[:2], folder / "virchow2-base.npy")
        variants = {}
        for name, expected in (("scale_1", 1.0), ("scale_2", 2.0)):
            info = er["scales"][name]
            path = extra / info["file"]
            if sha256(path) != info["sha256"] or any(abs(v - expected) > .03 for v in info["mpp_xy"]):
                raise ValueError("Context scale checksum or physical spacing differs")
            shutil.copyfile(path, folder / f"{name}.npy")
            variants[name] = dict(features=f"{name}.npy", shape_yx=info["shape_yx"],
                                  mpp_xy=info["mpp_xy"])
        tifffile.imwrite(folder / "expert-target-private.tif", target)
        colors = [[int(c) for c in np.random.default_rng(i * 41 + 17).integers(35, 225, 3)]
                  for i in range(1, int(grid.max()) + 1)]
        spec = dict(case=args.case_id, image="he-working.png", tile_labels="anonymous-tiles.npy",
                    tile_size_yx=[TILE, TILE], tissue="tissue-working.tif",
                    tokens="virchow2-base.npy", mpp_xy=list(work_mpp),
                    native_mpp_xy=list(source_mpp), native_tissue="tissue-native.tif",
                    palette=colors, variants=variants, encoder_revision=ENCODER,
                    encoder_sha256=WEIGHTS)
        (folder / "case.json").write_text(json.dumps(spec, indent=2) + "\n")
        audit = dict(format="bora-expert-coarsened-case-v1", case_id=args.case_id,
                     dataset=dataset, split=assignment["split"],
                     source_group=assignment["global_source_group"],
                     source_sha256=sha256(source), expert_sha256=sha256(mask_path),
                     expert_source_path=str(mask_path),
                     base_receipt_sha256=sha256(base / "FEATURES_COMPLETE.json"),
                     multiscale_receipt_sha256=sha256(extra / "FEATURES_COMPLETE.json"),
                     split_sha256=sha256(split_path), native_shape_yx=list(native.shape[:2]),
                     working_shape_yx=list(work.shape[:2]), native_mpp_xy=list(source_mpp),
                     working_mpp_xy=list(work_mpp), anonymization=anon,
                     target_codes=used, canonical_role_ids=role_ids,
                     label_map_sha256=sha256(args.label_map), base_tokens=base_info,
                     scored_native_pixels=int(np.count_nonzero(target)),
                     coarse_input_derived_from_expert=True,
                     expert_annotation_in_predictor_manifest=False,
                     case_sha256=sha256(folder / "case.json"))
        (folder / "CASE_COMPLETE.json").write_text(json.dumps(audit, indent=2) + "\n")
        os.rename(folder, output)
    return audit


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--case-id", required=True)
    p.add_argument("--label-map", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(build(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
