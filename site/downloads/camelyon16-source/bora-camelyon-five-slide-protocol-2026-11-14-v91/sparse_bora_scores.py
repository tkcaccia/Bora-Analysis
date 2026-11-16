"""Bounded source-only Bora score projection from frozen sparse 1280-channel caches.

This adapter reads every present full-channel feature tile and projects only the
channels selected by the released Bora.training model. Missing tokens are filled
by the same nearest-token rule as the released dense predictor. It never reads
CAMELYON16 expert reference masks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from bora.r_prediction import load_model


ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
FEATURES = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v41/TEST_FEATURES_FROZEN.json'
FEATURE_AUDIT = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v44/TEST_FEATURES_AUDIT_VERIFIED.json'
MODEL = Path('/mnt/sata_ssd/ChatGPT/Bora-Data/external-camelyon16-five-slide-2026-11-14-v69/bora-pair-binary-head')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build_scores(slide: str, output: Path) -> dict:
    started = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    features = json.loads(FEATURES.read_text())
    audit = json.loads(FEATURE_AUDIT.read_text())
    if (features['status'] != 'SOURCE_ONLY_FEATURES_FROZEN' or
            audit['status'] != 'VERIFIED' or
            audit['frozen_handoff_sha256'] != sha(FEATURES)):
        raise ValueError('Frozen full-channel test source receipt differs')
    case = next(x for x in features['cases'] if x['slide_id'] == slide)
    scale = next(x for x in case['scales'] if x['factor'] == 1)
    receipt_path = Path(scale['feature_receipt_path'])
    if sha(receipt_path) != scale['feature_receipt_sha256']:
        raise ValueError('Frozen factor-1 feature receipt changed')
    receipt = json.loads(receipt_path.read_text())
    spec_path = Path(scale['source_spec_path'])
    if (sha(spec_path) != scale['source_spec_sha256'] or
            receipt['channels'] != 1280 or receipt['factor'] != 1 or
            receipt['eligible_tiles'] != scale['eligible_tiles']):
        raise ValueError('Source image, grid or feature contract differs')
    spec = json.loads(spec_path.read_text())
    if (spec['slide_id'] != slide or spec['image_sha256'] != case['image_sha256'] or
            spec['tissue_support_sha256'] != case['support_sha256']):
        raise ValueError('Test image or H&E-only support changed')
    eligible_path = Path(spec['eligible_path'])
    if sha(eligible_path) != spec['eligible_sha256']:
        raise ValueError('Frozen strict tile admission changed')
    eligible = np.load(eligible_path, mmap_mode='r', allow_pickle=False)
    model = load_model(MODEL)
    if model['classes'].tolist() != [1, 2] or model['factors'].tolist() != [1]:
        raise ValueError('Only the frozen two-class factor-1 head is admitted')
    selected = np.asarray(model['selected'], np.int64)
    weights = model['coef'].T / model['std'][:, None]
    offset = model['intercept'] - (model['mean'] / model['std']) @ model['coef'].T
    height, width, channels = receipt['shape_yxc']
    assert channels == 1280 and eligible.shape == (height // 16, width // 16)
    output.mkdir(parents=True)
    try:
        raw_path = output / 'raw-scores.npy'
        raw = np.lib.format.open_memmap(raw_path, mode='w+', dtype=np.float32,
                                         shape=(2, height, width))
        raw[:] = np.nan
        finite_count = 0
        for entry in receipt['row_receipts']:
            row = int(entry['row'])
            count = int(entry['tile_columns'])
            if not count:
                assert entry['tile_path'] is None and entry['column_path'] is None
                continue
            tile_path = Path(entry['tile_path'])
            column_path = Path(entry['column_path'])
            if sha(tile_path) != entry['tile_sha256'] or sha(column_path) != entry['column_sha256']:
                raise ValueError(f'Feature row {row} differs from audited cache')
            tiles = np.load(tile_path, mmap_mode='r', allow_pickle=False)
            columns = np.load(column_path, mmap_mode='r', allow_pickle=False)
            if (tiles.shape != (count, 16, 16, 1280) or columns.shape != (count,) or
                    np.any(columns < 0) or np.any(columns >= eligible.shape[1]) or
                    len(np.unique(columns)) != count or not eligible[row, columns].all()):
                raise ValueError(f'Sparse feature row {row} is not registered')
            for index, column in enumerate(columns):
                chosen = np.asarray(tiles[index, :, :, selected], np.float32)
                # NumPy's advanced indexing may lead with the selected axis.
                if chosen.shape == (len(selected), 16, 16):
                    chosen = np.moveaxis(chosen, 0, -1)
                if chosen.shape != (16, 16, len(selected)):
                    raise ValueError('Selected-channel tile axis order differs')
                finite = np.isfinite(chosen).all(axis=-1)
                values = np.full((16, 16), np.nan, np.float32)
                if finite.any():
                    values[finite] = (chosen[finite].astype(np.float64) @ weights + offset).reshape(-1)
                y = row * 16
                x = int(column) * 16
                raw[0, y:y + 16, x:x + 16] = -values
                raw[1, y:y + 16, x:x + 16] = values
                finite_count += int(finite.sum())
        if finite_count < 1:
            raise ValueError('No finite selected features in admitted tissue')
        raw.flush()
        present = np.isfinite(np.asarray(raw[1]))
        nearest = distance_transform_edt(~present, return_distances=False,
                                         return_indices=True)
        for channel in range(2):
            raw[channel] = np.asarray(raw[channel])[nearest[0], nearest[1]]
        raw.flush()
        if not np.isfinite(raw).all():
            raise ValueError('Nearest projection left nonfinite scores')
        smooth_path = output / 'smoothed-scores.npy'
        smooth = np.lib.format.open_memmap(smooth_path, mode='w+', dtype=np.float32,
                                            shape=raw.shape)
        for channel in range(2):
            smooth[channel] = gaussian_filter(np.asarray(raw[channel]), sigma=.75,
                                              mode='nearest')
        smooth.flush()
        result = {'status': 'SPARSE_BORA_SCORES_COMPLETE', 'slide_id': slide,
                  'image_sha256': case['image_sha256'],
                  'support_sha256': case['support_sha256'],
                  'full1280_feature_receipt_sha256': sha(receipt_path),
                  'full1280_feature_audit_sha256': scale['audit_sha256'],
                  'model_sha256': model['model_sha256'],
                  'classes': model['classes'].tolist(),
                  'selected_channels': selected.tolist(),
                  'score_shape_class_yx': list(raw.shape),
                  'finite_selected_tokens': finite_count,
                  'nearest_imputed_tokens': int(present.size - present.sum()),
                  'raw_scores_path': str(raw_path), 'raw_scores_sha256': sha(raw_path),
                  'smoothed_scores_path': str(smooth_path),
                  'smoothed_scores_sha256': sha(smooth_path),
                  'wall_seconds': time.monotonic() - started,
                  'peak_host_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'test_reference_opened': False, 'online_publication': False}
        atomic(output / 'SCORES_COMPLETE.json', result)
        return result
    except BaseException as error:
        atomic(output / 'FAILED.json', {'status': 'FAILED_LATCHED', 'slide_id': slide,
               'error': repr(error), 'partial_bytes_preserved': True,
               'test_reference_opened': False, 'online_publication': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide', choices=['tumor_084', 'tumor_091', 'normal_019'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_scores(args.slide, args.output)), flush=True)
