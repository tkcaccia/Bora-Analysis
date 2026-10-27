#!/usr/bin/env python3
"""Encode 1 and 2 µm context views for Bora's unchanged RGB9+19 conversion.

Runs one bounded GPU case. Source images and the base feature receipt are
identity checked. RGB tiles are kept only in memory and never written to disk.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

from encode_virchow2 import (CHANNELS, REVISION, WEIGHTS_SHA256, atomic_json,
                             digest, load_model)

TILE_PIXELS = 224
SCALES = (("scale_1", 1.0), ("scale_2", 2.0))


def planned_shape(native_shape_yx: tuple[int, int], native_mpp_xy: list[float],
                  scale_mpp: float) -> tuple[int, int]:
    height, width = native_shape_yx
    mx, my = native_mpp_xy
    shape = (max(1, round(height * my / scale_mpp)),
             max(1, round(width * mx / scale_mpp)))
    return shape


def checked_source(source: Path, base_dir: Path, declared_mpp: list[float]) -> dict:
    base = base_dir / "FEATURES_COMPLETE.json"
    inputs = base_dir / "INPUTS.json"
    cleanup = base_dir / "GPU_CLEANUP.json"
    if not all(p.is_file() for p in (base, inputs, cleanup)):
        raise FileNotFoundError("Complete base features and GPU cleanup are required")
    receipt, identity = json.loads(base.read_text()), json.loads(inputs.read_text())
    if (receipt.get("model_revision") != REVISION or receipt.get("model_sha256") != WEIGHTS_SHA256 or
            identity.get("image_sha256") != digest(source) or
            any(abs(a - b) / b > .001 for a, b in zip(receipt["source_mpp_xy"], declared_mpp))):
        raise ValueError("Source, spacing or encoder differs from base features")
    if receipt.get("inputs_sha256") != digest(inputs):
        raise ValueError("Base input receipt changed")
    return dict(base_receipt_sha256=digest(base), base_inputs_sha256=digest(inputs),
                source_sha256=identity["image_sha256"],
                base_source_upsampled=any(x > .5 for x in declared_mpp))


def run(args: argparse.Namespace) -> dict:
    import numpy as np
    import pyvips
    from PIL import Image
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source.resolve()
    if source != Path(args.staged_source).resolve() and digest(source) != digest(args.staged_source):
        raise ValueError("Staged source image differs")
    proof = checked_source(source, args.base_dir, args.source_mpp_xy)
    image = pyvips.Image.new_from_file(str(args.staged_source), access="sequential")
    if image.format != "uchar" or image.bands not in (3, 4):
        raise ValueError("Expected 8-bit RGB/RGBA histology image")
    if image.width * image.height > args.max_native_pixels:
        raise ValueError("Field exceeds bounded CPU image size")
    native = np.ndarray(buffer=image.write_to_memory(), dtype=np.uint8,
                        shape=(image.height, image.width, image.bands)).copy()
    if image.bands == 4:
        alpha = native[..., 3:4].astype(np.float32) / 255
        native = np.rint(native[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    if not args.check_only:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device is unavailable")
        torch.set_num_threads(args.threads)
        torch.cuda.set_per_process_memory_fraction(.6)
        model = load_model(args.model_dir)
        mean = torch.tensor([.485, .456, .406], device="cuda")[None, :, None, None]
        std = torch.tensor([.229, .224, .225], device="cuda")[None, :, None, None]
    else:
        return dict(source=str(source), native_shape_yx=[image.height, image.width],
                    shapes={name: planned_shape((image.height, image.width),
                                                args.source_mpp_xy, scale)
                            for name, scale in SCALES}, **proof)
    args.output.mkdir(parents=True)
    identity = dict(format="bora-virchow2-multiscale-v1", case_id=args.case_id,
                    source_image=str(source), source_mpp_xy=args.source_mpp_xy,
                    encoder_revision=REVISION, encoder_sha256=WEIGHTS_SHA256,
                    tile_pixels=TILE_PIXELS, context_scales_mpp=[1.0, 2.0],
                    tile_images_saved=False, expert_annotation_read=False, **proof)
    atomic_json(args.output / "INPUTS.json", identity)
    deadline = time.monotonic() + args.max_seconds
    results = {}
    for name, scale in SCALES:
        shape_yx = planned_shape((image.height, image.width), args.source_mpp_xy, scale)
        h, w = shape_yx
        pixels = np.asarray(Image.fromarray(native, "RGB").resize((w, h), Image.Resampling.LANCZOS))
        grid_h, grid_w = math.ceil(h / TILE_PIXELS), math.ceil(w / TILE_PIXELS)
        tokens = np.zeros((grid_h * 16, grid_w * 16, CHANNELS), np.float16)
        encoded = 0
        for row in range(grid_h):
            for col in range(grid_w):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Multiscale field exceeded its finite deadline")
                y, x = row * TILE_PIXELS, col * TILE_PIXELS
                tile = np.full((TILE_PIXELS, TILE_PIXELS, 3), 255, np.uint8)
                valid = pixels[y:min(y + TILE_PIXELS, h), x:min(x + TILE_PIXELS, w)]
                tile[:valid.shape[0], :valid.shape[1]] = valid
                if float(np.mean(np.any(tile < 250, axis=2))) < .002:
                    continue
                tensor = torch.from_numpy(tile.transpose(2, 0, 1).copy())[None].to("cuda").float() / 255
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    result = model((tensor - mean) / std)
                if result.shape != (1, 261, CHANNELS):
                    raise ValueError("Unexpected Virchow2 spatial-token count")
                spatial = result[0, 5:].reshape(16, 16, CHANNELS).float().cpu().numpy()
                if not np.isfinite(spatial).all():
                    raise FloatingPointError("Nonfinite multiscale tokens")
                tokens[row * 16:(row + 1) * 16, col * 16:(col + 1) * 16] = spatial.astype(np.float16)
                encoded += 1
        trimmed = tokens[:math.ceil(h / 14), :math.ceil(w / 14)]
        file = args.output / f"{name}.npy"
        temp = args.output / f"{name}.tmp.npy"
        np.save(temp, trimmed)
        os.replace(temp, file)
        actual_spacing = [args.source_mpp_xy[0] * image.width / w,
                          args.source_mpp_xy[1] * image.height / h]
        results[name] = dict(file=file.name, sha256=digest(file),
                             shape_yx=[h, w], tokens_shape=list(trimmed.shape),
                             mpp_xy=actual_spacing, encoded_tiles=encoded,
                             grid_tiles=grid_h * grid_w)
        del pixels, tokens, trimmed
        print(json.dumps(dict(case=args.case_id, scale=name, encoded_tiles=encoded)), flush=True)
    complete = dict(**identity, scales=results)
    atomic_json(args.output / "FEATURES_COMPLETE.json", complete)
    return complete


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-id", required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--staged-source", type=Path, required=True)
    p.add_argument("--base-dir", type=Path, required=True)
    p.add_argument("--source-mpp-xy", type=float, nargs=2, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-native-pixels", type=int, default=80_000_000)
    p.add_argument("--max-seconds", type=int, default=1680)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    if not 1 <= a.threads <= 8 or not 1 <= a.max_seconds <= 3600:
        raise ValueError("Resource limits invalid")
    print(json.dumps(run(a), indent=2), flush=True)


if __name__ == "__main__":
    main()
