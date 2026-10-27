#!/usr/bin/env python3
"""Encode a frozen wsiTools tile plan with the pinned Virchow2 checkpoint.

One invocation processes one image. Tiles are cropped and resized in memory;
only token/position chunks are written. A failed case is never retried here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

REVISION = "3158645804b69e3f3bc4439d4116edddf0840a72"
WEIGHTS_SHA256 = "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b"
TARGET_MPP = 0.5
TILE_PIXELS = 224
CHANNELS = 1280


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def checked_plan(plan_dir: Path, source_image: Path, limit: int) -> tuple[dict, list[dict]]:
    meta_path = plan_dir / "PLAN_COMPLETE.json"
    meta = json.loads(meta_path.read_text())
    table_path = plan_dir / meta["tile_table"]
    if meta.get("format") != "wsiTools-virchow2-plan-v1":
        raise ValueError("Unexpected wsiTools plan format")
    if Path(meta["image"]).resolve() != source_image.resolve():
        raise ValueError("Tile plan belongs to a different image")
    if (meta["target_mpp"], meta["tile_pixels"]) != (TARGET_MPP, TILE_PIXELS):
        raise ValueError("Tile plan does not match the pinned Virchow2 geometry")
    if not 1 <= meta["tiles"] <= limit:
        raise ValueError("WSI tile count exceeds the declared bounded scope")
    with table_path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if len(rows) != meta["tiles"]:
        raise ValueError("Tile table length disagrees with the completed plan")
    width, height = map(int, meta["image_size_wh"])
    nw, nh = map(int, meta["native_tile_wh"])
    nx, ny = math.ceil(width / nw), math.ceil(height / nh)
    if len(rows) != nx * ny:
        raise ValueError("Tile plan does not cover the complete image")
    for i, r in enumerate(rows):
        row, col = divmod(i, nx)
        x, y = col * nw, row * nh
        w, h = min(nw, width - x), min(nh, height - y)
        if (int(r["tile_index"]), int(r["row"]), int(r["col"]),
            int(r["x"]), int(r["y"]), int(r["valid_width"]),
            int(r["valid_height"]), int(r["native_width"]),
            int(r["native_height"]), int(r["output_width"]),
            int(r["output_height"])) != (i, row, col, x, y, w, h, nw, nh,
                                          TILE_PIXELS, TILE_PIXELS):
            raise ValueError(f"Tile table geometry differs at tile {i}")
    return meta, rows


def load_model(model_dir: Path):
    import torch
    import timm
    from timm.layers import SwiGLUPacked
    from safetensors.torch import load_file

    weights = model_dir / "model.safetensors"
    if digest(weights) != WEIGHTS_SHA256:
        raise ValueError("Virchow2 checkpoint differs from the pinned revision")
    config = json.loads((model_dir / "config.json").read_text())
    with torch.device("meta"):
        model = timm.create_model(config["architecture"], pretrained=False,
                                  **config["model_args"], mlp_layer=SwiGLUPacked,
                                  act_layer=torch.nn.SiLU)
    model.load_state_dict(load_file(str(weights)), strict=True, assign=True)
    return model.to("cuda").eval().requires_grad_(False)


def crop(image, r: dict):
    import numpy as np
    from PIL import Image

    x, y = int(r["x"]), int(r["y"])
    w, h = int(r["valid_width"]), int(r["valid_height"])
    nw, nh = int(r["native_width"]), int(r["native_height"])
    native = image.extract_area(x, y, w, h)
    pixels = np.ndarray(buffer=native.write_to_memory(), dtype=np.uint8,
                        shape=(h, w, native.bands)).copy()
    if native.bands == 4:
        alpha = pixels[..., 3:4].astype(np.float32) / 255
        pixels = np.rint(pixels[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    tile = Image.fromarray(pixels, "RGB")
    if (w, h) != (nw, nh):
        padded = Image.new("RGB", (nw, nh), "white")
        padded.paste(tile, (0, 0))
        tile = padded
    return np.asarray(tile.resize((TILE_PIXELS, TILE_PIXELS), Image.Resampling.LANCZOS))


def run(args: argparse.Namespace) -> dict:
    import numpy as np
    import pyvips
    import torch

    if args.output.exists():
        raise FileExistsError("Use a new case output directory; never retry into a failed bundle")
    if not 1 <= args.chunk_size <= 64 or not 1 <= args.threads <= 8:
        raise ValueError("Chunk size or thread count exceeds the bounded scope")
    if not torch.cuda.is_available():
        raise RuntimeError("This encoder requires the authorized configured GPU")
    image_path = args.image.resolve()
    source_image = (args.source_image or args.image).resolve()
    meta, rows = checked_plan(args.plan, source_image, args.max_tiles)
    if not 0 <= args.first_tile < len(rows):
        raise ValueError("First tile is outside the complete plan")
    last_tile = min(args.last_tile, len(rows)) if args.last_tile is not None else len(rows)
    if not args.first_tile < last_tile or last_tile - args.first_tile > args.max_unit_tiles:
        raise ValueError("One GPU unit must contain 1..max-unit-tiles planned tiles")
    image_hash = digest(image_path)
    if image_hash != args.image_sha256:
        raise ValueError("Source image SHA256 mismatch")
    image = pyvips.Image.new_from_file(str(image_path), access="random")
    if image.format != "uchar" or image.bands not in (3, 4):
        raise ValueError("Expected scanner RGB or RGBA uint8 image")
    if [image.width, image.height] != meta["image_size_wh"]:
        raise ValueError("Source image dimensions changed from the wsiTools plan")
    torch.set_num_threads(args.threads)
    torch.cuda.set_per_process_memory_fraction(0.6)
    model = load_model(args.model_dir)
    mean = torch.tensor([.485, .456, .406], device="cuda")[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device="cuda")[None, :, None, None]
    args.output.mkdir(parents=True)
    identity = dict(format="bora-virchow2-wsi-features-v1",
                    source_image=str(source_image), staged_image=str(image_path),
                    image_sha256=image_hash,
                    plan_sha256=digest(args.plan / "PLAN_COMPLETE.json"),
                    tile_table_sha256=digest(args.plan / meta["tile_table"]),
                    encoder_revision=REVISION, encoder_sha256=WEIGHTS_SHA256,
                    target_mpp=TARGET_MPP, tile_pixels=TILE_PIXELS,
                    native_tile_wh=meta["native_tile_wh"],
                    source_mpp_xy=meta["source_mpp_xy"],
                    effective_mpp_xy=meta["effective_mpp_xy"],
                    grid_tiles=len(rows), first_tile=args.first_tile,
                    last_tile=last_tile, tile_images_saved=False)
    atomic_json(args.output / "INPUTS.json", identity)
    deadline = time.monotonic() + args.max_seconds
    encoded_total = 0
    for first in range(args.first_tile, last_tile, args.chunk_size):
        if time.monotonic() >= deadline:
            raise TimeoutError("The declared per-case inference deadline expired")
        selected = rows[first:min(first + args.chunk_size, last_tile)]
        positions, features, pooled = [], [], []
        started = time.monotonic()
        for r in selected:
            pixels = crop(image, r)
            stain_fraction = float(np.mean(np.any(pixels < 250, axis=2)))
            if stain_fraction < .002:
                continue
            tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy())[None].to("cuda").float() / 255
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                result = model((tensor - mean) / std)
            if result.shape != (1, 261, CHANNELS):
                raise ValueError(f"Unexpected Virchow2 token shape: {tuple(result.shape)}")
            spatial = result[0, 5:].reshape(16, 16, CHANNELS).float().cpu().numpy()
            if not np.isfinite(spatial).all():
                raise FloatingPointError("Nonfinite Virchow2 feature")
            features.append(spatial.astype(np.float16))
            pooled.append(spatial.mean((0, 1), dtype=np.float32))
            positions.append([int(r[k]) for k in ("tile_index", "x", "y", "valid_width", "valid_height")])
        chunk_id = (first - args.first_tile) // args.chunk_size
        out = args.output / f"chunk-{chunk_id:05d}.npz"
        tmp = out.with_name(out.name + ".tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(f,
                tokens=np.stack(features) if features else np.empty((0, 16, 16, CHANNELS), np.float16),
                pooled=np.stack(pooled) if pooled else np.empty((0, CHANNELS), np.float32),
                tile_table=np.asarray(positions, dtype=np.int64).reshape(-1, 5))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out)
        atomic_json(args.output / f"chunk-{chunk_id:05d}.json",
                    dict(first=first, last=first + len(selected),
                         encoded_tiles=len(features), sha256=digest(out),
                         seconds=time.monotonic() - started))
        encoded_total += len(features)
        print(json.dumps(dict(chunk=chunk_id, of=math.ceil((last_tile - args.first_tile) / args.chunk_size),
                              encoded_tiles=len(features))), flush=True)
    if not encoded_total:
        raise ValueError("No tissue-containing Virchow2 tiles were encoded")
    result = dict(**identity, encoded_tissue_tiles=encoded_total,
                  skipped_near_white_tiles=last_tile - args.first_tile - encoded_total,
                  chunks=math.ceil((last_tile - args.first_tile) / args.chunk_size),
                  feature_shape="16x16x1280 float16 spatial tokens; 1280 float32 pooled",
                  expert_annotation_used=False)
    atomic_json(args.output / "FEATURES_COMPLETE.json", result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--source-image", type=Path,
                   help="Original source path when --image is a verified local staging copy")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--image-sha256", required=True)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-tiles", type=int, default=25000)
    p.add_argument("--first-tile", type=int, default=0)
    p.add_argument("--last-tile", type=int)
    p.add_argument("--max-unit-tiles", type=int, default=1024)
    p.add_argument("--max-seconds", type=int, default=3600)
    args = p.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
