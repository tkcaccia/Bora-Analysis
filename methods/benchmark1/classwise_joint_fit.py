#!/usr/bin/env python3
"""Freeze a class-balanced feature union and joint softmax model on 397 images."""
import hashlib
import json
import os
from pathlib import Path
from time import monotonic

os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
HERE = Path(__file__).resolve().parent
V4 = ROOT / 'migration/benchmark1-joint-supervised-2026-11-11-v4'
OLD = ROOT / 'migration/benchmark1-supervised-strategy-search-2026-11-10-v1'
OUT = Path('/mnt/sata_ssd/ChatGPT/Bora-Data/benchmark1-joint-supervised-2026-11-11-v5')

def supports(sample, class_names):
    mapping = json.loads((ROOT / 'benchmark/methods/benchmark1/tissue_label_harmonization.json').read_text())['mapping']
    result = {}
    for cid in range(1, 25):
        role = class_names[str(cid)]
        names = [d for d, labels in mapping.items() if any(
            entry.get('use') == 'score' and entry.get('canonical_role') == role
            for entry in labels.values())]
        result[cid] = np.asarray([sample['dataset_names'].index(d) for d in names
                                  if d in sample['dataset_names']], dtype=np.uint8)
    return result


def classwise_union(x, y, train_idx, dataset, supported, k, seed):
    all_selected = []
    for cid in range(1, 25):
        eligible = train_idx[np.isin(dataset[train_idx], supported[cid])]
        pos = eligible[y[eligible] == cid]
        neg = eligible[y[eligible] != cid]
        if len(pos) < 20 or len(neg) < 20:
            raise ValueError(f'Class {cid} insufficient in training')
        if len(neg) > 4 * len(pos):
            neg = np.random.default_rng(seed + cid).choice(neg, 4 * len(pos), replace=False)
        z = np.asarray(x[np.concatenate((pos, neg))], np.float32)
        scale = np.maximum(z.std(axis=0), 1e-4)
        rank = np.abs((z[:len(pos)].mean(axis=0) - z[len(pos):].mean(axis=0)) / scale)
        all_selected.extend(np.argsort(-rank, kind='stable')[:k].tolist())
    return np.unique(np.asarray(all_selected, np.int32))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    receipt = HERE / 'FINAL_MODEL_FROZEN.json'
    path = OUT / 'model.joblib'
    if receipt.exists() or path.exists():
        raise FileExistsError('No in-place refit')
    search = json.loads((V4 / 'SEARCH_COMPLETE.json').read_text())
    sample = json.loads((OLD / 'SAMPLE_COMPLETE.json').read_text())
    frozen = json.loads((ROOT / 'migration/benchmark1-supervised-virchow2-response-2026-11-10-v1/INPUTS_FROZEN.json').read_text())
    assert search['validation_opened'] == sample['validation_cases_opened'] == 0
    by_name = {r['family']: r for r in search['aggregate']}
    best = by_name['classwise_union64_joint_logistic']
    chosen = by_name['classwise_union32_joint_logistic']
    folds = {family: [r['metrics']['four_multiclass_dataset_macro_iou'] for r in search['rows'] if r['family'] == family]
             for family in ('classwise_union64_joint_logistic', 'classwise_union32_joint_logistic')}
    difference = np.asarray(folds['classwise_union64_joint_logistic']) - np.asarray(folds['classwise_union32_joint_logistic'])
    one_se = float(np.std(difference, ddof=1) / np.sqrt(len(difference)))
    assert best['four_multiclass_dataset_macro_iou'] - chosen['four_multiclass_dataset_macro_iou'] <= one_se
    assert chosen['mean_seconds'] < best['mean_seconds']
    assert chosen['mean_muscle_iou'] > best['mean_muscle_iou']
    assert sha(sample['feature_path']) == sample['feature_sha256']
    assert sha(sample['index_path']) == sample['index_sha256']
    x = np.load(sample['feature_path'], mmap_mode='r')
    idx = np.load(sample['index_path'])
    y, datasets = idx['label'], idx['dataset_id']
    assert x.shape == (139736, 1280) and len(np.unique(idx['case_index'])) == 397
    tic = monotonic()
    train_idx = np.arange(len(y))
    selected = classwise_union(x, y, train_idx, datasets, supports(sample, frozen['classes']),
                               32, 20261111)
    a = np.asarray(x[:, selected], np.float32)
    mean = a.mean(axis=0)
    std = np.maximum(a.std(axis=0), 1e-4)
    a = (a - mean) / std
    count = np.bincount(y, minlength=25).astype(np.float64)
    weight = np.maximum(count[y], 1) ** (-.5)
    weight /= weight.mean()
    model = LogisticRegression(C=.1, solver='lbfgs', max_iter=150, tol=1e-3)
    model.fit(np.ascontiguousarray(a), y, sample_weight=weight)
    assert np.array_equal(model.classes_, np.arange(1, 25))
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump({'selected': selected, 'mean': mean, 'std': std,
                 'classifier': model}, path, compress=3)
    result = {'status': 'TRAINING_ONLY_CLASS_BALANCED_UNION_FROZEN',
              'validation_cases_opened': 0, 'training_images': 397,
              'training_tokens': len(y), 'input_channels': 1280,
              'selection': '32 classwise channels per class, deduplicated union; joint multinomial logistic head',
              'selected_union_channels': len(selected),
              'selected_original_channel_indices': selected.tolist(),
              'training_four_dataset_macro_iou': chosen['four_multiclass_dataset_macro_iou'],
              'training_all_case_macro_iou': chosen['mean_case_macro_iou'],
              'training_gcss_macro_iou': chosen['gcss_macro_iou'],
              'training_muscle_iou': chosen['mean_muscle_iou'],
              'one_standard_error_of_fold_difference': one_se,
              'difference_from_larger_model': best['four_multiclass_dataset_macro_iou'] - chosen['four_multiclass_dataset_macro_iou'],
              'selection_rule': 'Choose faster 32-per-class union when its four-dataset score is within one paired-fold SE of the 64-per-class leader; muscle IoU secondary.',
              'search_receipt_sha256': sha(V4 / 'SEARCH_COMPLETE.json'),
              'sample_receipt_sha256': sha(OLD / 'SAMPLE_COMPLETE.json'),
              'source_sha256': sha(Path(__file__)),
              'fit_seconds': monotonic() - tic,
              'iterations': model.n_iter_.tolist(),
              'model_path': str(path), 'model_sha256': sha(path)}
    receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('FINAL_MODEL_FROZEN', len(selected), result['model_sha256'], flush=True)


if __name__ == '__main__':
    main()
