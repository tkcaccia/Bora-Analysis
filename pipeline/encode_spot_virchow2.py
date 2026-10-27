#!/usr/bin/env python3
"""Encode calibrated Visium spot-centered tiles without saving tile images."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time

from encode_virchow2 import (CHANNELS, REVISION, TILE_PIXELS, TARGET_MPP,
                             WEIGHTS_SHA256, atomic_json, digest, load_model)


def checked_plan(plan_dir: Path, source_image: Path, max_tiles: int) -> tuple[dict, list[dict]]:
    meta = json.loads((plan_dir / "PLAN_COMPLETE.json").read_text())
    if meta.get("format") != "wsiTools-virchow2-spot-plan-v1":
        raise ValueError("Expected a completed wsiTools spot tile plan")
    if Path(meta["image"]).resolve() != source_image.resolve():
        raise ValueError("Spot plan belongs to another source image")
    if meta["target_mpp"] != TARGET_MPP or meta["tile_pixels"] != TILE_PIXELS:
        raise ValueError("Spot plan Virchow2 target spacing differs")
    with (plan_dir / meta["tile_table"]).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not 1 <= len(rows) == meta["planned_spots"] <= max_tiles:
        raise ValueError("Spot plan has an invalid or excessive number of rows")
    seen = set()
    for i, row in enumerate(rows):
        if int(row["tile_index"]) != i or not row["spot_id"] or row["spot_id"] in seen:
            raise ValueError("Spot barcode or tile index changed")
        seen.add(row["spot_id"])
        x, y, w, h = (int(row[k]) for k in ("x", "y", "valid_width", "valid_height"))
        nw, nh = int(row["native_width"]), int(row["native_height"])
        left, top, right, bottom = (int(row[k]) for k in
                                    ("pad_left", "pad_top", "pad_right", "pad_bottom"))
        if (x < 0 or y < 0 or min(w, h) < 1 or
                left + w + right != nw or top + h + bottom != nh or
                min(left, top, right, bottom) < 0 or
                (int(row["output_width"]), int(row["output_height"])) !=
                (TILE_PIXELS, TILE_PIXELS)):
            raise ValueError(f"Invalid spot tile geometry at row {i}")
    return meta, rows


def crop_spot(image, r: dict):
    import numpy as np
    from PIL import Image

    x, y, w, h = (int(r[k]) for k in ("x", "y", "valid_width", "valid_height"))
    nw, nh = int(r["native_width"]), int(r["native_height"])
    left, top = int(r["pad_left"]), int(r["pad_top"])
    native = image.extract_area(x, y, w, h)
    pixels = np.ndarray(buffer=native.write_to_memory(), dtype=np.uint8,
                        shape=(h, w, native.bands)).copy()
    if native.bands == 4:
        alpha = pixels[..., 3:4].astype(np.float32) / 255
        pixels = np.rint(pixels[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    padded = Image.new("RGB", (nw, nh), "white")
    padded.paste(Image.fromarray(pixels, "RGB"), (left, top))
    return np.asarray(padded.resize((TILE_PIXELS, TILE_PIXELS), Image.Resampling.LANCZOS))


def run(args: argparse.Namespace) -> dict:
    import numpy as np
    import pyvips
    import torch

    if args.output.exists():
        raise FileExistsError("Use a new output directory for every spot unit")
    if not torch.cuda.is_available():
        raise RuntimeError("Authorized configured GPU is unavailable")
    source = (args.source_image or args.image).resolve()
    meta, rows = checked_plan(args.plan, source, args.max_tiles)
    last = min(args.last_tile, len(rows)) if args.last_tile is not None else len(rows)
    if (args.first_tile < 0 or args.first_tile >= last or last > len(rows) or
            last - args.first_tile > args.max_unit_tiles or
            not 1 <= args.chunk_size <= 64 or not 1 <= args.threads <= 8):
        raise ValueError("Spot unit bounds or resource limits are invalid")
    local = args.image.resolve()
    image_hash = digest(local)
    if image_hash != args.image_sha256:
        raise ValueError("Staged Visium source differs from pinned provider image")
    image = pyvips.Image.new_from_file(str(local), access="random")
    if image.format != "uchar" or image.bands not in (3, 4):
        raise ValueError("Expected 8-bit RGB/RGBA histology image")
    for r in rows[args.first_tile:last]:
        if int(r["x"]) + int(r["valid_width"]) > image.width or int(r["y"]) + int(r["valid_height"]) > image.height:
            raise ValueError("Spot crop extends beyond the source image")
    torch.set_num_threads(args.threads)
    torch.cuda.set_per_process_memory_fraction(.6)
    model = load_model(args.model_dir)
    mean = torch.tensor([.485, .456, .406], device="cuda")[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device="cuda")[None, :, None, None]
    args.output.mkdir(parents=True)
    identity = dict(format="bora-virchow2-spot-features-v1",
                    source_image=str(source), staged_image=str(local),
                    image_sha256=image_hash,
                    plan_sha256=digest(args.plan / "PLAN_COMPLETE.json"),
                    tile_table_sha256=digest(args.plan / meta["tile_table"]),
                    encoder_revision=REVISION, encoder_sha256=WEIGHTS_SHA256,
                    target_mpp=TARGET_MPP, tile_pixels=TILE_PIXELS,
                    source_mpp_xy=meta["source_mpp_xy"],
                    effective_mpp_xy=meta["effective_mpp_xy"],
                    source_upsampled=meta["source_upsampled"],
                    planned_spots=len(rows), first_tile=args.first_tile,
                    last_tile=last, tile_images_saved=False)
    atomic_json(args.output / "INPUTS.json", identity)
    deadline = time.monotonic() + args.max_seconds
    total = 0
    for first in range(args.first_tile, last, args.chunk_size):
        if time.monotonic() >= deadline:
            raise TimeoutError("Spot unit exceeded its finite inference deadline")
        subset = rows[first:min(first + args.chunk_size, last)]
        tokens, pooled, positions, barcodes = [], [], [], []
        started = time.monotonic()
        for r in subset:
            pixels = crop_spot(image, r)
            if float(np.mean(np.any(pixels < 250, axis=2))) < .002:
                continue
            tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy())[None].to("cuda").float() / 255
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                result = model((tensor - mean) / std)
            if result.shape != (1, 261, CHANNELS):
                raise ValueError("Unexpected Virchow2 spatial-token count")
            spatial = result[0, 5:].reshape(16, 16, CHANNELS).float().cpu().numpy()
            if not np.isfinite(spatial).all():
                raise FloatingPointError("Nonfinite Virchow2 spot features")
            tokens.append(spatial.astype(np.float16))
            pooled.append(spatial.mean((0, 1), dtype=np.float32))
            positions.append([int(r[k]) for k in ("tile_index", "x", "y", "valid_width", "valid_height")])
            barcodes.append(r["spot_id"])
        chunk = (first - args.first_tile) // args.chunk_size
        path = args.output / f"chunk-{chunk:05d}.npz"
        temp = path.with_name(path.name + ".tmp")
        with temp.open("wb") as f:
            np.savez_compressed(f,
                tokens=np.stack(tokens) if tokens else np.empty((0, 16, 16, CHANNELS), np.float16),
                pooled=np.stack(pooled) if pooled else np.empty((0, CHANNELS), np.float32),
                tile_table=np.asarray(positions, dtype=np.int64).reshape(-1, 5),
                barcodes=np.asarray(barcodes, dtype="U64"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        atomic_json(args.output / f"chunk-{chunk:05d}.json",
                    dict(first=first, last=first + len(subset), encoded_spots=len(tokens),
                         sha256=digest(path), seconds=time.monotonic() - started))
        total += len(tokens)
        print(json.dumps(dict(chunk=chunk, encoded_spots=len(tokens))), flush=True)
    if total == 0:
        raise ValueError("No tissue-containing spot tiles were encoded")
    result = dict(**identity, encoded_spots=total,
                  skipped_near_white_spots=last - args.first_tile - total,
                  chunks=math.ceil((last - args.first_tile) / args.chunk_size),
                  gene_expression_used_for_encoding=False)
    atomic_json(args.output / "FEATURES_COMPLETE.json", result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--source-image", type=Path)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--image-sha256", required=True)
    p.add_argument("--first-tile", type=int, default=0)
    p.add_argument("--last-tile", type=int)
    p.add_argument("--max-tiles", type=int, default=15000)
    p.add_argument("--max-unit-tiles", type=int, default=1024)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-seconds", type=int, default=3480)
    args = p.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
