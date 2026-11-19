#!/usr/bin/env python3
"""Prepare a bounded tumor_093 training-tissue tile sample from frozen features."""

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
OUT = DATA / 'external-camelyon16-training-kodama-2026-11-19-v2'
BASE = DATA / 'external-camelyon16-five-slide-2026-11-14-v65/bora-pair-preparation/image-000'
RECEIPT = BASE / 'factor-1/FEATURES_COMPLETE.json'
PREP = DATA / 'external-camelyon16-one-slide-2026-11-16-v3/preparation/PREPARATION_COMPLETE.json'
TRAIN = DATA / 'external-camelyon16-threshold-tumor093-tumor087-2026-11-18-v1/training/TRAINING_COMPLETE.json'
SEED = 20261119
N = 900


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError('Use a new version for a rerun')
    feature_receipt = json.loads(RECEIPT.read_text())
    prep = json.loads(PREP.read_text())
    if (feature_receipt['status'] != 'COMPLETE' or
            feature_receipt['channels'] != 1280 or
            feature_receipt['factor'] != 1 or
            feature_receipt['shape_yxc'] != [2944, 3200, 1280] or
            len(prep['cases']) != 1 or
            'tumor_093' not in prep['cases'][0]['source']['image']):
        raise ValueError('Frozen tumor_093 preparation differs')
    feature_path = Path(feature_receipt['features_path'])
    if sha(feature_path) != feature_receipt['features_sha256']:
        raise ValueError('Full-channel feature identity differs')
    eligible = np.load(BASE / 'eligible.npy', allow_pickle=False)
    counts = np.load(BASE / 'tissue-pixel-counts.npy', allow_pickle=False)
    if (eligible.shape != (184, 200) or counts.shape != eligible.shape or
            sha(BASE / 'eligible.npy') != prep['cases'][0]['source']['eligible_sha256'] or
            sha(BASE / 'tissue-pixel-counts.npy') !=
            prep['cases'][0]['source']['tissue_pixel_counts_sha256']):
        raise ValueError('Reviewed tissue selection differs')
    good = eligible & (counts >= int(np.ceil(0.9 * 224 * 224)))
    train = json.loads(TRAIN.read_text())
    annotation = Path(train['annotations'][0]['annotation'])
    if (train['status'] != 'COMPLETE' or len(train['annotations']) != 1 or
            'tumor_093' not in str(annotation) or
            sha(annotation) != train['annotations'][0]['annotation_sha256']):
        raise ValueError('Fitting annotation identity differs')
    with rasterio.open(annotation) as source:
        if source.shape != (41216, 44800):
            raise ValueError('Annotation registration differs')
        labels = source.read(1, out_shape=(184, 200), resampling=Resampling.nearest)
    positive = np.argwhere(good & (labels == 2))
    negative = np.argwhere(good & (labels == 1))
    if not 8 <= len(positive) <= 100 or len(negative) < N:
        raise ValueError('Fitting slide training-label counts differ')
    rng = np.random.default_rng(SEED)
    negative = negative[rng.choice(len(negative), N-len(positive), replace=False)]
    sample = np.concatenate((positive, negative))
    sample = sample[rng.permutation(N)]
    sample_labels = labels[sample[:, 0], sample[:, 1]]
    feature = np.load(feature_path, mmap_mode='r', allow_pickle=False)
    if feature.shape != (184 * 16, 200 * 16, 1280):
        raise ValueError('Feature/token grid mismatch')
    means = np.empty((N, 1280), np.float32)
    for i, (y, x) in enumerate(sample):
        block = np.asarray(feature[y*16:(y+1)*16, x*16:(x+1)*16], np.float32)
        if not np.isfinite(block).all():
            raise ValueError('Nonfinite tile features')
        means[i] = block.mean(axis=(0, 1), dtype=np.float64)
    OUT.mkdir(parents=True)
    np.save(OUT / 'tile-yx.npy', sample.astype(np.int16))
    np.save(OUT / 'tile-expert-center-label.npy', sample_labels.astype(np.uint8))
    np.savetxt(OUT / 'tile-expert-center-label.csv', sample_labels,
               delimiter=',', fmt='%d')
    np.save(OUT / 'tile-means-1280.npy', means)
    np.savetxt(OUT / 'tile-means-1280.csv', means, delimiter=',', fmt='%.8g')
    result = {
        'format': 'camelyon16-tumor093-kodama-input-v1',
        'status': 'COMPLETE',
        'slide': 'tumor_093',
        'scope': 'exploratory training-slide-only; no test data or scores',
        'feature_receipt_sha256': sha(RECEIPT),
        'preparation_sha256': sha(PREP),
        'feature_sha256': feature_receipt['features_sha256'],
        'tissue_eligible_tiles': int(good.sum()),
        'sampled_tiles': N,
        'minimum_tissue_fraction': 0.9,
        'selection_seed': SEED,
        'sampling': 'all tissue-rich metastatic-center tiles plus seeded nonmetastatic-center sample',
        'metastatic_center_tiles': int((sample_labels == 2).sum()),
        'nonmetastatic_center_tiles': int((sample_labels == 1).sum()),
        'annotation_sha256': train['annotations'][0]['annotation_sha256'],
        'channels': 1280,
        'tile_yx_sha256': sha(OUT / 'tile-yx.npy'),
        'tile_expert_center_label_sha256': sha(OUT / 'tile-expert-center-label.npy'),
        'tile_expert_center_label_csv_sha256': sha(OUT / 'tile-expert-center-label.csv'),
        'tile_means_sha256': sha(OUT / 'tile-means-1280.npy'),
        'tile_means_csv_sha256': sha(OUT / 'tile-means-1280.csv'),
    }
    (OUT / 'INPUT_COMPLETE.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
