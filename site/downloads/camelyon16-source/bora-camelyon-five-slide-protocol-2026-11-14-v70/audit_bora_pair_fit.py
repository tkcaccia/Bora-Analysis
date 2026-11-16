"""Reconstruct the two-slide Bora classwise ridge fit from frozen training data."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.linear_model import RidgeClassifier

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
PAIR = ROOT / 'external-benchmark-wsss-camelyon-2026-11-13-v85/PAIR_FEATURES_FROZEN.json'
PREP = DATA / 'external-camelyon16-five-slide-2026-11-14-v65/bora-pair-preparation/PREPARATION_COMPLETE.json'
MODEL = DATA / 'external-camelyon16-five-slide-2026-11-14-v69/bora-pair-binary-head'
SETTINGS = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v67/METHOD_SETTINGS_FROZEN.json'
HERE = Path(__file__).parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def labels_at_token_centres(path, shape):
    height, width = shape
    ys = np.arange(7, height, 14)
    xs = np.arange(7, width, 14)
    out = np.empty((len(ys), len(xs)), np.uint8)
    with rasterio.open(path) as src:
        if (src.height, src.width, src.count) != (height, width, 1):
            raise ValueError('Training label grid differs')
        for row, y in enumerate(ys):
            out[row] = src.read(1, window=Window(0, int(y), width, 1))[0, xs]
    return out


def main():
    target = HERE / 'BORA_PAIR_FIT_AUDIT_VERIFIED.json'
    if target.exists():
        raise FileExistsError(target)
    setting = json.loads(SETTINGS.read_text())
    complete_path = MODEL / 'TRAINING_COMPLETE.json'
    complete = json.loads(complete_path.read_text())
    input_path = MODEL / 'TRAINING_INPUTS.json'
    inputs = json.loads(input_path.read_text())
    prep = json.loads(PREP.read_text())
    pair = json.loads(PAIR.read_text())
    if complete['status'] != 'COMPLETE' or complete['preparation_sha256'] != sha(PREP):
        raise ValueError('Model receipt/preparation differ')
    if (inputs['preparation_sha256'] != sha(PREP) or
            inputs['max_tokens_per_class'] != setting['max_tokens_per_class'] or
            inputs['channels_per_class'] != setting['channels_per_class'] or
            inputs['alpha'] != setting['alpha'] or inputs['seed'] != setting['seed']):
        raise ValueError('Training input/settings differ')
    records = {(r['slide'], r['factor']): r for r in pair['records']}
    rng = np.random.default_rng(setting['seed'])
    matrix, targets, source_rows = [], [], []
    quota = (setting['max_tokens_per_class'] + 1) // 2
    for index, slide in enumerate(('tumor_093', 'normal_093')):
        label_path = DATA / f'external-camelyon16-five-slide-2026-11-14-v58/training-labels/{slide}/training-classes-in-support.tif'
        audit = json.loads((ROOT / f'bora-camelyon-five-slide-protocol-2026-11-14-v60/{slide.upper()}_TRAINING_LABELS_AUDIT_VERIFIED.json').read_text())
        if sha(label_path) != audit['clipped_mask_sha256'] or inputs['images'][index]['annotation_sha256'] != audit['clipped_mask_sha256']:
            raise ValueError('Training annotation differs')
        case = prep['cases'][index]['source']
        if case['image_sha256'] != inputs['images'][index]['image_sha256']:
            raise ValueError('Training image identity differs')
        labels = labels_at_token_centres(label_path, case['image_shape_yx'])
        grid = np.load(records[(slide, 1)]['canonical_path'], mmap_mode='r', allow_pickle=False)
        for cls in np.unique(labels):
            cls = int(cls)
            if cls == 0:
                continue
            flat = np.flatnonzero(labels.ravel() == cls)
            chosen = rng.choice(flat, size=min(len(flat), quota), replace=False)
            ys, xs = np.unravel_index(chosen, labels.shape)
            values = np.asarray(grid[ys, xs], np.float32)
            finite = np.isfinite(values).all(axis=1)
            matrix.append(values[finite])
            targets.append(np.full(int(finite.sum()), cls, np.uint8))
            source_rows.append({'slide': slide, 'class_id': cls,
                                'available_centres': int(len(flat)),
                                'sampled_finite': int(finite.sum())})
    x = np.concatenate(matrix)
    y = np.concatenate(targets)
    classes, counts = np.unique(y, return_counts=True)
    if classes.tolist() != [1, 2] or int(len(y)) != complete['sampled_tokens']:
        raise ValueError('Training sample/classes differ')
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(x.std(axis=0, dtype=np.float64).astype(np.float32), 1e-4)
    z = (x - mean) / std
    picked = set()
    for cls in classes:
        positive = y == cls
        contrast = np.abs(z[positive].mean(axis=0) - z[~positive].mean(axis=0))
        picked.update(np.argsort(-contrast, kind='stable')[:setting['channels_per_class']].tolist())
    selected = np.asarray(sorted(picked), np.int32)
    weights = counts[np.searchsorted(classes, y)].astype(np.float64) ** -.5
    weights /= weights.mean()
    head = RidgeClassifier(alpha=setting['alpha'], solver='cholesky')
    head.fit(np.ascontiguousarray(z[:, selected]), y, sample_weight=weights)
    predictor = MODEL / 'predictor.npz'
    if sha(predictor) != complete['model_sha256']:
        raise ValueError('Saved predictor hash differs')
    with np.load(predictor, allow_pickle=False) as saved:
        if not np.array_equal(saved['classes'], classes) or not np.array_equal(saved['selected'], selected):
            raise ValueError('Classwise channel selection differs')
        checks = {'mean': (saved['mean'], mean[selected]),
                  'std': (saved['std'], std[selected]),
                  'coef': (saved['coef'], np.atleast_2d(head.coef_)),
                  'intercept': (saved['intercept'], np.atleast_1d(head.intercept_))}
        maximum = {}
        for name, (actual, expected) in checks.items():
            maximum[name] = float(np.max(np.abs(actual - expected)))
            if not np.allclose(actual, expected, atol=2e-5, rtol=2e-5):
                raise ValueError(f'Reconstructed {name} differs')
    service = subprocess.run(['systemctl', '--user', 'show', 'bora-cam16-bora-binary-fit-v69.service',
                              '-p', 'ActiveState', '-p', 'MainPID', '-p', 'ControlGroup',
                              '-p', 'Result', '-p', 'ExecMainStatus'],
                             capture_output=True, text=True, check=True).stdout
    if 'ActiveState=inactive' not in service or 'MainPID=0' not in service or 'ControlGroup=\n' not in service or 'Result=success' not in service:
        raise ValueError('Fit service is not terminally clean')
    result = {'status': 'VERIFIED', 'format': 'bora-cam-pair-fit-independent-audit-v1',
              'fit_slides': ['tumor_093', 'normal_093'], 'classes': classes.tolist(),
              'class_counts': counts.tolist(), 'sampled_tokens': int(len(y)),
              'source_rows': source_rows, 'selected_channels': selected.tolist(),
              'predictor_sha256': sha(predictor), 'receipt_sha256': sha(complete_path),
              'max_abs_reconstruction_difference': maximum,
              'service': service.strip().splitlines(),
              'test_reference_opened': False, 'online_publication': False}
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(target)


if __name__ == '__main__':
    main()
