#!/usr/bin/env python3
"""Fit the training-selected strategy on all 397 images before opening validation masks."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import time

os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'

import joblib
import numpy as np

from search import FROZEN, HARM, HERE, DATA, SEED, model_for_class


def sha(path: Path) -> str:
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    receipt_path = HERE / 'FINAL_MODEL_FROZEN.json'
    if receipt_path.exists():
        raise FileExistsError('Training-only model already frozen')
    search_path = HERE / 'SEARCH_COMPLETE.json'
    search = json.loads(search_path.read_text())
    if search['status'] != 'TRAINING_ONLY_STRATEGY_SELECTED' or search['validation_cases_opened'] != 0:
        raise ValueError('Training-only search not complete')
    sample = json.loads((HERE / 'SAMPLE_COMPLETE.json').read_text())
    x = np.load(sample['feature_path'], mmap_mode='r', allow_pickle=False)
    idx = np.load(sample['index_path'], allow_pickle=False)
    y = idx['label']; dataset = idx['dataset_id']
    train_idx = np.arange(len(y))
    assert x.shape == (139736, 1280) and len(y) == len(dataset) == len(x)
    harmony = json.loads(HARM.read_text())['mapping']
    classes = json.loads(FROZEN.read_text())['classes']
    models = {}
    supports = {}
    selected = search['selected']
    family = selected['family']
    started = time.monotonic()
    for cid in range(1, 25):
        role = classes[str(cid)]
        names = sorted(d for d, mapping in harmony.items() if any(
            v.get('use') == 'score' and v.get('canonical_role') == role
            for v in mapping.values()) and d in sample['dataset_names'])
        supported = np.asarray([sample['dataset_names'].index(d) for d in names], dtype=np.uint8)
        models[cid] = model_for_class(x, y, train_idx, dataset, supported, cid, family, SEED + 99000)
        supports[cid] = names
        print('FINAL_FIT', cid, role, flush=True)
    model_path = DATA / 'selected-model.joblib'
    joblib.dump(models, model_path, compress=3)
    receipt = {'status': 'TRAINING_ONLY_WINNER_FROZEN', 'gpu_used': False,
               'validation_cases_opened': 0, 'training_cases': 397,
               'training_tokens': len(y), 'family': family,
               'prior_weight': selected['prior_weight'],
               'selected_training_heldout_macro_iou': selected['mean_case_macro_iou'],
               'model_path': str(model_path), 'model_sha256': sha(model_path),
               'search_receipt_sha256': sha(search_path),
               'sample_receipt_sha256': sha(HERE / 'SAMPLE_COMPLETE.json'),
               'selected_original_channel_indices_by_class': {
                   str(cid): models[cid][0].tolist() for cid in models},
               'supported_datasets_by_class': {str(cid): supports[cid] for cid in supports},
               'training_fit_seconds': time.monotonic() - started,
               'source_sha256': sha(Path(__file__))}
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print('MODEL_FROZEN', family, selected['prior_weight'], flush=True)


if __name__ == '__main__':
    main()
