"""One isolated Virchow2 GPU process for one CAM image/scale, saving tissue tiles.

Numerical encoding is copied from the frozen Bora 0.3.4 worker. Only the
on-disk representation changes: every admitted 224-pixel tile keeps its
16×16×1,280 factor-specific token block and exact tile-column index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
import sys

import numpy as np

sys.path.insert(0, '/mnt/sata_ssd/ChatGPT/Bora-Data/worktrees/bora-r-package-2026-11-14-v1/src')
from bora.r_feature_preflight import SCALE_PAD, TILE, sha256

STRIDE = 112
PATCH = 14
CHANNELS = 1280


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('xb') as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _model(model_dir: Path, expected_sha256: str):
    import timm
    import torch
    from timm.layers import SwiGLUPacked
    from safetensors.torch import load_file

    checkpoint = model_dir / "model.safetensors"
    if sha256(checkpoint) != expected_sha256:
        raise ValueError("Virchow2 checkpoint SHA256 differs")
    cfg = json.loads((model_dir / "config.json").read_text())
    with torch.device("meta"):
        model = timm.create_model(cfg["architecture"], pretrained=False,
                                  **cfg["model_args"], mlp_layer=SwiGLUPacked,
                                  act_layer=torch.nn.SiLU)
    model.load_state_dict(load_file(str(checkpoint)), strict=True, assign=True)
    return model.to("cuda").eval().requires_grad_(False)


def _starts(length: int) -> list[int]:
    if length < TILE:
        raise ValueError("Padded image strip is smaller than a model window")
    last = length - TILE
    return sorted(set([*range(0, last + 1, STRIDE), last]))


def _scale_eligibility(base: np.ndarray, factor: int) -> np.ndarray:
    if base.ndim != 2 or base.dtype != np.bool_ or any(v % factor for v in base.shape):
        raise ValueError("Tissue tile grid cannot be scaled exactly")
    if factor == 1:
        return base
    return base.reshape(base.shape[0] // factor, factor,
                        base.shape[1] // factor, factor).any(axis=(1, 3))


def _encode_row(image, row: int, eligible: np.ndarray, model, mean, std,
                deadline: float) -> tuple[np.ndarray, int]:
    import torch

    height, width = image.height, image.width
    target0 = row * TILE
    target1 = min(target0 + TILE, height)
    origin = max(0, target0 - STRIDE)
    end = min(height, target1 + STRIDE)
    part = image.extract_area(0, origin, width, end - origin)
    pixels = np.ndarray(buffer=part.write_to_memory(), dtype=np.uint8,
                        shape=(end - origin, width, 3)).copy()
    padded = np.pad(pixels,
                    ((0, (-len(pixels)) % PATCH), (0, (-width) % PATCH), (0, 0)),
                    mode="reflect")
    grid_h, grid_w = padded.shape[0] // PATCH, padded.shape[1] // PATCH
    accum = np.zeros((grid_h, grid_w, CHANNELS), np.float32)
    mass = np.zeros((grid_h, grid_w), np.float32)
    axis = np.maximum(np.hanning(16), .05).astype(np.float32)
    blend = axis[:, None] * axis[None, :]
    coords: list[tuple[int, int]] = []
    for y in _starts(padded.shape[0]):
        global_y = origin + y
        r0, r1 = global_y // TILE, min((global_y + TILE - 1) // TILE,
                                     eligible.shape[0] - 1)
        for x in _starts(padded.shape[1]):
            c0, c1 = x // TILE, min((x + TILE - 1) // TILE,
                                  eligible.shape[1] - 1)
            if eligible[r0:r1 + 1, c0:c1 + 1].any():
                coords.append((y, x))
    for first in range(0, len(coords), 4):
        if time.monotonic() >= deadline:
            raise TimeoutError("Virchow2 row deadline reached")
        batch = np.stack([padded[y:y + TILE, x:x + TILE].transpose(2, 0, 1)
                          for y, x in coords[first:first + 4]])
        tensor = torch.from_numpy(batch.copy()).to("cuda").float() / 255
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            encoded = model((tensor - mean) / std)
        if encoded.ndim != 3 or encoded.shape[1:] != (261, CHANNELS):
            raise ValueError("Virchow2 spatial output differs from 261×1280")
        values = encoded[:, 5:, :].float().cpu().numpy().reshape(-1, 16, 16, CHANNELS)
        for (y, x), value in zip(coords[first:first + 4], values):
            ys, xs = y // PATCH, x // PATCH
            accum[ys:ys + 16, xs:xs + 16] += value * blend[..., None]
            mass[ys:ys + 16, xs:xs + 16] += blend
    offset = (target0 - origin) // PATCH
    count = (target1 - target0 + PATCH - 1) // PATCH
    result = accum[offset:offset + count]
    weights = mass[offset:offset + count]
    valid = weights > 0
    result[valid] /= weights[valid, None]
    result[~valid] = np.nan
    return result.astype(np.float16), len(coords)


def run(spec_path: Path) -> dict:
    import pyvips
    import torch

    spec = json.loads(spec_path.read_text())
    if spec.get("format") != "bora-cam-sparse-feature-worker-v1":
        raise ValueError("Feature worker contract differs")
    image_path = Path(spec["image"])
    eligible_path = Path(spec["eligible_path"])
    model_dir = Path(spec["model_dir"])
    output_dir = Path(spec["output_dir"])
    factor = int(spec["factor"])
    if factor not in (1, 2, 4) or output_dir.exists():
        raise ValueError("Scale or output path differs")
    if sha256(image_path) != spec["image_sha256"] or \
       sha256(eligible_path) != spec["eligible_sha256"]:
        raise ValueError("Source image or eligibility SHA256 differs")
    if sha256(model_dir / "config.json") != spec["model_config_sha256"]:
        raise ValueError("Model configuration SHA256 differs")
    base = np.load(eligible_path, mmap_mode="r", allow_pickle=False)
    if base.dtype != np.bool_ or list(base.shape) != spec["base_grid_shape_yx"]:
        raise ValueError("Eligible tile grid shape differs")
    grid = _scale_eligibility(base, factor)
    if not grid.any():
        raise ValueError("No eligible tiles at this scale")
    projected_payload = int(grid.sum()) * 16 * 16 * CHANNELS * 2
    max_feature_bytes = int(spec["max_feature_bytes"])
    reserve_bytes = int(spec["reserve_bytes"])
    if projected_payload + grid.shape[0] * 4096 > max_feature_bytes or \
       reserve_bytes < 100 * (1 << 30):
        raise OSError('Sparse scale exceeds its frozen byte allocation or volume reserve')
    disk = os.statvfs('/mnt/sata_ssd/ChatGPT/Bora-Data')
    if disk.f_bavail * disk.f_frsize - max_feature_bytes < reserve_bytes:
        raise OSError('Sparse feature output would breach the free-space reserve')
    source = pyvips.Image.new_from_file(str(image_path), access="random")
    if source.format != "uchar" or source.bands != 3:
        raise ValueError("Calibrated H&E image must be uint8 RGB")
    if [source.height, source.width] != spec["image_shape_yx"]:
        raise ValueError("Image dimensions differ from preflight")
    padded_h, padded_w = spec["padded_shape_yx"]
    if padded_h % SCALE_PAD or padded_w % SCALE_PAD or \
       padded_h < source.height or padded_w < source.width:
        raise ValueError("Invalid scale-compatible padding")
    padded = source.embed(0, 0, padded_w, padded_h, extend="white")
    image = padded if factor == 1 else padded.resize(1 / factor,
                                                     kernel="lanczos3")
    if (image.height, image.width) != (padded_h // factor, padded_w // factor):
        raise ValueError("Scaled image dimensions differ")
    rows, cols = image.height // TILE, image.width // TILE
    if grid.shape != (rows, cols):
        raise ValueError("Scaled tissue grid differs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for Virchow2 extraction")
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.65)
    started = time.monotonic()
    deadline = started + float(spec["max_seconds"])
    model = _model(model_dir, spec["model_checkpoint_sha256"])
    mean = torch.tensor([.485, .456, .406], device="cuda")[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device="cuda")[None, :, None, None]
    output_dir.mkdir(parents=True)
    shape = (image.height // PATCH, image.width // PATCH, CHANNELS)
    _atomic_json(output_dir / "INPUTS.json", spec)
    row_receipts = []
    selected_tiles = 0
    persisted_bytes = 0
    for row in range(rows):
        columns = np.flatnonzero(grid[row]).astype(np.int32)
        if len(columns):
            values, windows = _encode_row(image, row, grid, model, mean, std, deadline)
            if values.shape != (16, shape[1], CHANNELS):
                raise ValueError("Encoded token row geometry differs")
            blocks = np.ascontiguousarray(
                values.reshape(16, cols, 16, CHANNELS)[:, columns]
                .transpose(1, 0, 2, 3))
            if blocks.shape != (len(columns), 16, 16, CHANNELS) or \
               not np.isfinite(blocks).all():
                raise ValueError("Admitted tile has missing or nonfinite spatial tokens")
            tile_path = output_dir / f'row-{row:04d}-tiles.npy'
            column_path = output_dir / f'row-{row:04d}-columns.npy'
            _atomic_npy(tile_path, blocks)
            _atomic_npy(column_path, columns)
            tile_hash = sha256(tile_path)
            column_hash = sha256(column_path)
            persisted_bytes += tile_path.stat().st_size + column_path.stat().st_size
            selected_tiles += len(columns)
        else:
            windows = 0
            tile_path = column_path = None
            tile_hash = column_hash = None
        receipt = {"row": row, "tile_columns": int(len(columns)),
                   "block_shape_n_y_x_c": [int(len(columns)), 16, 16, CHANNELS],
                   "encoded_windows": windows,
                   "tile_path": str(tile_path) if tile_path else None,
                   "tile_sha256": tile_hash,
                   "column_path": str(column_path) if column_path else None,
                   "column_sha256": column_hash,
                   "elapsed_seconds": round(time.monotonic() - started, 3)}
        _atomic_json(output_dir / f"row-{row:04d}.json", receipt)
        row_receipts.append(receipt)
    torch.cuda.synchronize()
    result = {"format": "bora-r-sparse-feature-scale-v1", "status": "COMPLETE",
              "source_spec_sha256": sha256(spec_path), "shape_yxc": list(shape),
              "dtype": "float16", "channels": CHANNELS, "factor": factor,
              "eligible_tiles": int(grid.sum()), "persisted_tiles": selected_tiles,
              "persisted_bytes": persisted_bytes, "rows": rows,
              "row_receipts": row_receipts,
              "wall_seconds": round(time.monotonic() - started, 3),
              "peak_host_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()}
    if selected_tiles != int(grid.sum()):
        raise ValueError('Sparse tile count differs from eligibility grid')
    if persisted_bytes > max_feature_bytes:
        raise OSError('Persisted sparse features exceeded the frozen allocation')
    _atomic_json(output_dir / "FEATURES_COMPLETE.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.spec), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
