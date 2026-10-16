"""Separate RINGS prostate tumour-region benchmark from verified inventory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import tifffile
from scipy.ndimage import binary_erosion, distance_transform_edt

from benchmark.run_bora_pilot import (DATA_ROOT, PYTHON, TILER, audit_repeat,
                                      vector_roundtrip)


DATASETS = ('rings-tumor',)
ROOT = DATA_ROOT / 'results/bora-additional-binary-2026-10-16'


def select(dataset: str, count: int = 3) -> list[dict]:
    path = DATA_ROOT / 'prepared' / f'{dataset}.jsonl'
    rows = [json.loads(line) for line in path.open()]
    chosen = []
    groups = set()
    for row in rows:
        if 0 not in row['raw_labels'] or not any(int(x) > 0 for x in row['raw_labels']):
            continue
        if row['group_id'] in groups:
            continue
        chosen.append(row)
        groups.add(row['group_id'])
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError(f'{dataset}: only {len(chosen)} eligible groups for {count} requested')
    return chosen


def score(pred: np.ndarray, truth: np.ndarray) -> dict:
    predicted = pred > 0
    reference = truth > 0
    tp = int(np.count_nonzero(predicted & reference))
    fp = int(np.count_nonzero(predicted & ~reference))
    fn = int(np.count_nonzero(~predicted & reference))
    union = tp + fp + fn
    pb = predicted ^ binary_erosion(predicted)
    tb = reference ^ binary_erosion(reference)
    if pb.any() and tb.any():
        dp = distance_transform_edt(~tb)[pb]
        dt = distance_transform_edt(~pb)[tb]
        f1 = {}
        for radius in (8, 16, 32):
            precision = float(np.mean(dp <= radius))
            recall = float(np.mean(dt <= radius))
            f1[str(radius)] = (2 * precision * recall / (precision + recall)
                               if precision + recall else 0.0)
        assd = float((dp.mean() + dt.mean()) / 2)
        hd95 = float(np.percentile(np.concatenate((dp, dt)), 95))
    else:
        f1 = {str(radius): None for radius in (8, 16, 32)}
        assd = hd95 = None
    return {'pixel_accuracy': float(np.mean(predicted == reference)),
            'foreground_iou': tp / union if union else None,
            'foreground_dice': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None,
            'boundary_f1': f1, 'assd_px': assd, 'hd95_px': hd95,
            'foreground_pixels': int(predicted.sum()),
            'boundary_pixels': int(pb.sum())}


def run_one(dataset: str, row: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(Image.open(DATA_ROOT / row['mask']))
    if raw.ndim != 2:
        raise ValueError(f'{dataset}/{row["id"]}: expected single-channel mask')
    truth = (raw > 0).astype(np.uint8)
    image = np.asarray(Image.open(DATA_ROOT / row['image']).convert('RGB'))
    if image.shape[:2] != truth.shape:
        raise ValueError(f'{dataset}/{row["id"]}: image/mask dimensions differ')
    image_path, truth_path = output / 'image.tif', output / 'binary_truth.tif'
    if not image_path.exists():
        tifffile.imwrite(image_path, image, photometric='rgb', compression='deflate')
    if not truth_path.exists():
        tifffile.imwrite(truth_path, truth, compression='deflate')
    (output / 'source.json').write_text(json.dumps({
        'inventory': row, 'target_rule': 'raw_mask_value_gt_zero',
        'task': 'binary_target_vs_supplied_outside',
        'interpretation': 'binary_prostate_tumour_region'}, indent=2) + '\n')
    tiled = output / 'tiling'
    tiled.mkdir(exist_ok=True)
    if not (tiled / 'metrics_cpp.json').exists():
        command = [str(TILER), '--mask', str(truth_path), '--output', str(tiled),
                   '--workers', '4', '--border-error-percent', '0']
        for size in (56, 90, 216):
            command += ['--tile-size', str(size)]
        began = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True)
        (tiled / 'tiling_run.json').write_text(json.dumps({
            'command': command, 'wall_seconds': time.perf_counter() - began,
            'returncode': result.returncode}, indent=2) + '\n')
        if result.returncode:
            (tiled / 'failure.log').write_text(result.stdout + '\n' + result.stderr)
            raise RuntimeError(f'{dataset}/{row["id"]}: tiling failed')
    for size in (56, 90, 216):
        case = output / f'tile_{size}'
        case.mkdir(exist_ok=True)
        record = case / 'result.json'
        if record.exists():
            continue
        grid_path = tiled / f'mask_tile_grid_{size}x{size}_cpp.tif'
        grid = tifffile.imread(grid_path)
        coarse = np.repeat(np.repeat(grid, size, axis=0), size, axis=1)[:truth.shape[0], :truth.shape[1]]
        coarse_path = case / 'coarse.tif'
        tifffile.imwrite(coarse_path, coarse, compression='deflate')
        refined_path = case / 'refined.ome.tif'
        geojson_path = case / 'refined.geojson'
        command = [str(PYTHON), '-m', 'bora.cli', 'refine', str(image_path), str(coarse_path),
                   '--output', str(refined_path), '--geojson', str(geojson_path),
                   '--backend', 'adaptive-wand', '--no-pyramid', '--geojson-workers', '2',
                   '--geojson-profile', 'custom', '--geojson-page', '0',
                   '--geojson-min-area', '0', '--geojson-simplify', '0',
                   '--source-tile-size', str(size), '--block-size', '512']
        env = os.environ.copy()
        from benchmark.runtime import require_bora
        env['PYTHONPATH'] = str(require_bora())
        began = time.perf_counter()
        process = subprocess.run(command, capture_output=True, text=True, env=env)
        elapsed = time.perf_counter() - began
        if process.returncode:
            (case / 'failure.log').write_text(process.stdout + '\n' + process.stderr)
            raise RuntimeError(f'{dataset}/{row["id"]}/tile_{size}: Bora failed')
        refined = tifffile.imread(refined_path).squeeze()
        report = json.loads(refined_path.with_suffix('.report.json').read_text())
        vector = vector_roundtrip(geojson_path, truth.shape, refined)
        repeat = audit_repeat(case, refined, vector)
        result = {'dataset': dataset, 'sample_id': row['id'], 'group_id': row['group_id'],
                  'tile_size': size, 'coarse': score(coarse, truth),
                  'bora': score(refined, truth), 'geojson': vector,
                  'geojson_repeat': repeat, 'wall_seconds': elapsed,
                  'geojson_seconds': report['geojson_conversion']['runtime_seconds'],
                  'geojson_bytes': geojson_path.stat().st_size, 'command': command}
        record.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'dataset': dataset, 'id': row['id'], 'tile': size,
                          'iou': result['bora']['foreground_iou'], 'seconds': round(elapsed, 2)}),
              flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--dataset', choices=DATASETS, action='append')
    parser.add_argument('--per-dataset', type=int, default=3)
    args = parser.parse_args()
    datasets = args.dataset or list(DATASETS)
    args.output.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        rows = select(dataset, args.per_dataset)
        selection = [{'id': row['id'], 'group_id': row['group_id'],
                      'image': row['image'], 'mask': row['mask']} for row in rows]
        folder = args.output / dataset
        folder.mkdir(exist_ok=True)
        manifest = folder / 'selection.json'
        if manifest.exists() and json.loads(manifest.read_text()) != selection:
            raise ValueError(f'{dataset}: existing selection differs')
        manifest.write_text(json.dumps(selection, indent=2) + '\n')
        for row in rows:
            run_one(dataset, row, folder / row['id'])


if __name__ == '__main__':
    main()
