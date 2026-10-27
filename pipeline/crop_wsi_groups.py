#!/usr/bin/env python3
"""Register a whole-slide categorical tile grid to one cropped Bora image.

Only physical image coordinates are used here. The WSI cluster IDs must
already have been frozen; this program does not read expert annotations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile


def crop_grid(grid: np.ndarray, image_shape_yx: tuple[int, int],
              origin_xy: tuple[int, int], native_shape_yx: tuple[int, int],
              native_tile_wh: tuple[int, int]) -> np.ndarray:
    if grid.ndim != 2 or grid.dtype.kind not in 'ui' or grid.min() < 0 or grid.max() > 255:
        raise ValueError('Expected an integer whole-slide group grid in 0..255')
    height, width = image_shape_yx
    native_height, native_width = native_shape_yx
    tile_width, tile_height = native_tile_wh
    if min(height, width, native_height, native_width, tile_width, tile_height) <= 0:
        raise ValueError('Invalid image or tile dimensions')
    x = np.floor((origin_xy[0] + (np.arange(width) + .5) * native_width / width) / tile_width).astype(np.int64)
    y = np.floor((origin_xy[1] + (np.arange(height) + .5) * native_height / height) / tile_height).astype(np.int64)
    if x.min() < 0 or y.min() < 0 or x.max() >= grid.shape[1] or y.max() >= grid.shape[0]:
        raise ValueError('Crop extends beyond whole-slide tile grid')
    return grid[np.ix_(y, x)].astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid', type=Path, required=True)
    p.add_argument('--working-image', type=Path, required=True)
    p.add_argument('--origin-xy', nargs=2, type=int, required=True)
    p.add_argument('--native-shape-yx', nargs=2, type=int, required=True)
    p.add_argument('--native-tile-wh', nargs=2, type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    image = tifffile.imread(a.working_image)
    grid = np.load(a.grid, allow_pickle=False)
    coarse = crop_grid(grid, image.shape[:2], tuple(a.origin_xy),
                       tuple(a.native_shape_yx), tuple(a.native_tile_wh))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(a.output, coarse, compression='deflate')
    values, counts = np.unique(coarse, return_counts=True)
    print(json.dumps({'shape_yx': list(coarse.shape),
                      'groups': {str(int(v)): int(n) for v,n in zip(values,counts)}}, indent=2))


if __name__ == '__main__':
    main()
