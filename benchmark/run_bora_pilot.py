"""Resumable, source-aware pilot of the 3 x 3 Bora reconstruction benchmark.

This script deliberately selects a small, fixed set of ROIs. It does not
change the acquisition inventories or any original image/mask bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import rasterio
import tifffile
from PIL import Image
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from benchmark.runtime import BORA_ROOT, require_bora


DATA_ROOT = Path('/mnt/sata_ssd/Bora-benchmark')
PYTHON = Path('/home/chiamaka/miniconda3/envs/medsam/bin/python')
TILER = Path('/home/chiamaka/.local/bin/tiling-mask-cpp')
sys.path.insert(0, str(require_bora()))
SAMPLES = {
    'bcss': 'TCGA-OL-A5RW-DX1_xmin5772_ymin6839_MPP-0.2500',
    'tiger': 'tissue-cells--114S_[33188, 13944, 34412, 15110]',
    'wsss4luad': 'validation--25',
}


def load_record(dataset: str) -> dict:
    path = DATA_ROOT / 'prepared' / f'{dataset}.jsonl'
    for line in path.open():
        row = json.loads(line)
        if row['id'] == SAMPLES[dataset]:
            return row
    raise ValueError(f'Missing {dataset} pilot record')


def normalize_truth(raw: np.ndarray, ignore: list[int]) -> tuple[np.ndarray, dict[int, int]]:
    valid = sorted(int(x) for x in np.unique(raw) if int(x) not in ignore)
    if len(valid) < 2 or len(valid) > 255:
        raise ValueError(f'Expected 2..255 tissue classes, got {valid}')
    mapping = {value: i + 1 for i, value in enumerate(valid)}
    result = np.zeros(raw.shape, np.uint8)
    for original, encoded in mapping.items():
        result[raw == original] = encoded
    return result, mapping


def internal_border(labels: np.ndarray) -> np.ndarray:
    border = np.zeros(labels.shape, bool)
    a, b = labels[:-1], labels[1:]
    changed = (a != b) & (a > 0) & (b > 0)
    border[:-1] |= changed
    border[1:] |= changed
    a, b = labels[:, :-1], labels[:, 1:]
    changed = (a != b) & (a > 0) & (b > 0)
    border[:, :-1] |= changed
    border[:, 1:] |= changed
    return border


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    valid = truth > 0
    classes = np.unique(truth[valid])
    ious = []
    for value in classes:
        p = (pred == value) & valid
        t = truth == value
        union = np.count_nonzero(p | t)
        ious.append(float(np.count_nonzero(p & t) / union) if union else 1.0)
    pb, tb = internal_border(pred), internal_border(truth)
    # Ignore boundary detections outside reference annotations.
    pb &= valid
    if pb.any() and tb.any():
        to_truth = distance_transform_edt(~tb)
        to_pred = distance_transform_edt(~pb)
        f1 = {}
        for tolerance in (8, 16, 32):
            precision = float(np.mean(to_truth[pb] <= tolerance))
            recall = float(np.mean(to_pred[tb] <= tolerance))
            f1[str(tolerance)] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        assd = float((to_truth[pb].mean() + to_pred[tb].mean()) / 2)
        hd95 = float(np.percentile(np.r_[to_truth[pb], to_pred[tb]], 95))
    else:
        f1 = {str(x): None for x in (8, 16, 32)}
        assd = hd95 = None
    return {'pixel_accuracy': float(np.mean(pred[valid] == truth[valid])),
            'macro_iou': float(np.mean(ious)), 'per_class_iou': dict(zip(map(str, classes), ious)),
            'internal_boundary_f1': f1, 'internal_assd_px': assd,
            'internal_hd95_px': hd95, 'internal_boundary_pixels': int(pb.sum()),
            'retained_classes': int(sum(np.any(pred[valid] == x) for x in classes))}


def vector_roundtrip(path: Path, shape: tuple[int, int], expected: np.ndarray) -> dict:
    obj = json.loads(path.read_text())
    items = [(feature['geometry'], int(feature['properties']['value']))
             for feature in obj['features']]
    restored = rasterize(items, out_shape=shape, fill=0, dtype='uint8',
                         transform=rasterio.Affine.identity(), all_touched=False)
    return {'feature_count': len(items), 'pixel_agreement': float(np.mean(restored == expected)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def audit_repeat(case: Path, refined: np.ndarray, original: dict) -> dict:
    from bora.multicpu_geojson import convert
    repeated = case / 'refined.repeat.geojson'
    began = time.perf_counter()
    report = convert(case / 'refined.ome.tif', repeated, page=0, min_area=0,
                     simplify=0, workers=2)
    elapsed = time.perf_counter() - began
    second = vector_roundtrip(repeated, refined.shape, refined)
    return {'repeat_seconds': elapsed, 'repeat_converter_seconds': report['runtime_seconds'],
            'byte_reproducible': original['sha256'] == second['sha256'],
            'repeat_pixel_agreement': second['pixel_agreement'],
            'repeat_sha256': second['sha256']}


def audit_other_stages(case: Path, compact: Path, size: int,
                       truth: np.ndarray) -> dict:
    """Measure repeat execution of stages omitted by early pilot records."""
    began = time.perf_counter()
    grid = tifffile.imread(compact)
    expanded = np.repeat(np.repeat(grid, size, axis=0), size, axis=1)[:truth.shape[0], :truth.shape[1]]
    expansion = time.perf_counter() - began
    refined = np.squeeze(tifffile.imread(case / 'refined.ome.tif'))
    began = time.perf_counter()
    metrics(expanded, truth)
    metrics(refined, truth)
    scoring = time.perf_counter() - began
    began = time.perf_counter()
    vector_roundtrip(case / 'refined.geojson', truth.shape, refined)
    rasterize_time = time.perf_counter() - began
    return {'measurement': 'repeat_after_original_run',
            'grid_expansion_seconds': expansion,
            'two_raster_scores_seconds': scoring,
            'geojson_roundtrip_score_seconds': rasterize_time}


def run_sample(dataset: str, output: Path, backend: str,
               sizes: tuple[int, ...] = (56, 90, 216),
               errors: tuple[int, ...] = (0, 1, 5)) -> None:
    prep_started = time.perf_counter()
    row = load_record(dataset)
    target = output / dataset
    target.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(Image.open(DATA_ROOT / row['mask']))
    if raw.ndim != 2:
        raise ValueError(f'{dataset}: expected integer label plane, got {raw.shape}')
    truth, mapping = normalize_truth(raw, row['ignore_labels'])
    image = np.asarray(Image.open(DATA_ROOT / row['image']).convert('RGB'))
    if image.shape[:2] != truth.shape:
        raise ValueError(f'{dataset}: image/mask mismatch')
    source = target / 'normalized_truth.tif'
    input_image = target / 'image.tif'
    repeated_preparation = source.exists() and input_image.exists()
    if not source.exists():
        tifffile.imwrite(source, truth, compression='deflate')
    if not input_image.exists():
        tifffile.imwrite(input_image, image, photometric='rgb', compression='deflate')
    if repeated_preparation:
        # Include the same TIFF encoding work when auditing a resumed run.
        with tempfile.TemporaryDirectory(dir=target) as scratch:
            tifffile.imwrite(Path(scratch) / 'truth.tif', truth, compression='deflate')
            tifffile.imwrite(Path(scratch) / 'image.tif', image, photometric='rgb', compression='deflate')
    git_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=BORA_ROOT, text=True).strip()
    tiler_sha = hashlib.sha256(TILER.read_bytes()).hexdigest()
    (target / 'sample.json').write_text(json.dumps({'inventory': row, 'label_remap': mapping,
        'preparation_wall_seconds': time.perf_counter() - prep_started,
        'preparation_timing_basis': 'repeat_after_original_run' if repeated_preparation else 'original_run',
        'bora_git_revision': git_revision, 'tiling_mask_executable_sha256': tiler_sha}, indent=2))
    for error in errors:
        tiled = target / f'error_{error}'
        tiled.mkdir(exist_ok=True)
        report_path = tiled / 'metrics_cpp.json'
        if not report_path.exists():
            command = [str(TILER), '--mask', str(source), '--output', str(tiled),
                       '--workers', '4', '--border-error-percent', str(error),
                       '--border-error-seed', '20261016']
            for size in sizes:
                command += ['--tile-size', str(size)]
            started = time.perf_counter()
            subprocess.run(command, check=True, capture_output=True, text=True)
            (tiled / 'orchestration.json').write_text(json.dumps({'tiling_wall_seconds': time.perf_counter()-started,
                'command': command}, indent=2))
        for size in sizes:
            case = tiled / f'tile_{size}'
            case.mkdir(exist_ok=True)
            record_path = case / 'result.json'
            if record_path.exists():
                previous = json.loads(record_path.read_text())
                if 'geojson_repeat' not in previous:
                    refined_existing = np.squeeze(tifffile.imread(case / 'refined.ome.tif'))
                    previous['geojson_repeat'] = audit_repeat(case, refined_existing, previous['geojson'])
                if 'other_stage_timings' not in previous:
                    slug = f'{size}x{size}'
                    suffix = '_border_error' if error else ''
                    compact = tiled / f'mask_tile_grid_{slug}{suffix}_cpp.tif'
                    previous['other_stage_timings'] = audit_other_stages(case, compact, size, truth)
                record_path.write_text(json.dumps(previous, indent=2))
                continue
            slug = f'{size}x{size}'
            suffix = '_border_error' if error else ''
            compact = tiled / f'mask_tile_grid_{slug}{suffix}_cpp.tif'
            expansion_started = time.perf_counter()
            grid = tifffile.imread(compact)
            expanded = np.repeat(np.repeat(grid, size, axis=0), size, axis=1)[:truth.shape[0], :truth.shape[1]]
            expansion_seconds = time.perf_counter() - expansion_started
            coarse = case / 'coarse.tif'
            tifffile.imwrite(coarse, expanded, compression='deflate')
            refined_path = case / 'refined.ome.tif'
            geojson_path = case / 'refined.geojson'
            command = [str(PYTHON), '-m', 'bora.cli', 'refine', str(input_image), str(coarse),
                       '--output', str(refined_path), '--geojson', str(geojson_path),
                       '--backend', backend, '--no-pyramid', '--geojson-workers', '2',
                       '--geojson-profile', 'custom', '--geojson-page', '0',
                       '--geojson-min-area', '0', '--geojson-simplify', '0']
            if backend == 'adaptive-wand':
                command += ['--source-tile-size', str(size), '--block-size', '512']
            started = time.perf_counter()
            env = __import__('os').environ.copy()
            env['PYTHONPATH'] = str(require_bora())
            process = subprocess.run(command, capture_output=True, text=True, env=env)
            total = time.perf_counter()-started
            if process.returncode:
                (case / 'failure.log').write_text(process.stdout + '\n' + process.stderr)
                raise RuntimeError(f'{dataset} error={error} tile={size}: {process.stderr[-2000:]}')
            report = json.loads(refined_path.with_suffix('.report.json').read_text())
            refined = tifffile.imread(refined_path)
            if refined.ndim != 2:
                refined = np.squeeze(refined)
            scoring_started = time.perf_counter()
            coarse_metrics, refined_metrics = metrics(expanded, truth), metrics(refined, truth)
            raster_scoring_seconds = time.perf_counter() - scoring_started
            roundtrip_started = time.perf_counter()
            roundtrip = vector_roundtrip(geojson_path, truth.shape, refined)
            roundtrip_seconds = time.perf_counter() - roundtrip_started
            result = {'dataset': dataset, 'sample_id': row['id'], 'tile_size': size,
                'error_percent': error, 'backend': backend,
                'tiling': json.loads(report_path.read_text()),
                'coarse': coarse_metrics, 'refined': refined_metrics,
                'geojson': roundtrip,
                'other_stage_timings': {'measurement': 'original_run',
                    'grid_expansion_seconds': expansion_seconds,
                    'two_raster_scores_seconds': raster_scoring_seconds,
                    'geojson_roundtrip_score_seconds': roundtrip_seconds},
                'bora_wall_seconds_including_vectorization': total,
                'geojson_conversion_seconds': report['geojson_conversion']['runtime_seconds'],
                'refinement_estimated_seconds': total - report['geojson_conversion']['runtime_seconds'],
                'output_bytes': {'raster': refined_path.stat().st_size, 'geojson': geojson_path.stat().st_size},
                'command': command}
            result['geojson_repeat'] = audit_repeat(case, refined, result['geojson'])
            record_path.write_text(json.dumps(result, indent=2))
            print(json.dumps({'dataset': dataset, 'tile': size, 'error': error,
                'macro_iou': result['refined']['macro_iou'], 'seconds': round(total, 2)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DATA_ROOT / 'results' / 'bora-pilot-independent-2026-10-16')
    parser.add_argument('--dataset', choices=tuple(SAMPLES), action='append')
    parser.add_argument('--backend', choices=('watershed','adaptive-wand'), default='adaptive-wand')
    parser.add_argument('--wsss-validation-cohort', action='store_true',
                        help='run 0%% error on all multiclass WSSS4LUAD validation ROIs')
    args = parser.parse_args()
    if args.wsss_validation_cohort:
        rows = [json.loads(line) for line in (DATA_ROOT / 'prepared/wsss4luad.jsonl').open()]
        for row in rows:
            raw = np.asarray(Image.open(DATA_ROOT / row['mask']))
            valid_classes = [int(x) for x in np.unique(raw) if int(x) not in row['ignore_labels']]
            if len(valid_classes) < 2:
                print(json.dumps({'sample_id': row['id'], 'skipped': 'one_valid_class'}), flush=True)
                continue
            SAMPLES['wsss4luad'] = row['id']
            run_sample('wsss4luad', args.output / row['id'], args.backend,
                       errors=(0,))
        raise SystemExit(0)
    for name in args.dataset or tuple(SAMPLES):
        run_sample(name, args.output, args.backend)
