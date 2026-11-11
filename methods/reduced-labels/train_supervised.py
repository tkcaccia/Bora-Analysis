#!/usr/bin/env python3
"""Training-only reduced-label Virchow2 head search and final fit."""
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
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
HERE = Path(__file__).resolve().parent
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data/benchmark-label-reduction-2026-11-11-v1')
SAMPLE = ROOT / 'migration/benchmark1-supervised-strategy-search-2026-11-10-v1/SAMPLE_COMPLETE.json'
SPLITS = ROOT / 'migration/benchmark1-supervised-strategy-search-2026-11-10-v1/SEARCH_COMPLETE.json'
ONTOLOGY = HERE / 'ONTOLOGY_FROZEN.json'
HARM = ROOT / 'benchmark/methods/benchmark1/tissue_label_harmonization.json'
CONFIGS = ((128, 0.1), (256, 0.1), (256, 1.0))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def fit(x, y, n_channels, penalty):
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-4)
    z = (x - mean) / std
    ranks, _ = f_classif(z, y)
    order = np.argsort(-np.nan_to_num(ranks, nan=-np.inf), kind='stable')[:n_channels]
    counts = np.bincount(y, minlength=9).astype(np.float64)
    weights = np.maximum(counts[y], 1) ** -0.5
    weights /= weights.mean()
    classifier = LogisticRegression(C=penalty, solver='lbfgs', max_iter=150, tol=1e-3)
    classifier.fit(np.ascontiguousarray(z[:, order]), y, sample_weight=weights)
    return {'selected': order, 'mean': mean[order], 'std': std[order],
            'classifier': classifier, 'channels': n_channels, 'C': penalty}


def predict_scores(model, x):
    z = (x[:, model['selected']] - model['mean']) / model['std']
    return model['classifier'].decision_function(np.ascontiguousarray(z))


def score(pred, truth, cases, datasets, dataset_names):
    rows = []
    for case in np.unique(cases):
        m = cases == case
        t, p = truth[m], pred[m]
        ious = []
        for cid in np.unique(t):
            ious.append(float(np.count_nonzero((t == cid) & (p == cid)) /
                              np.count_nonzero((t == cid) | (p == cid))))
        rows.append({'case_index': int(case), 'dataset': dataset_names[int(datasets[m][0])],
                     'macro_iou': float(np.mean(ious)), 'accuracy': float(np.mean(t == p))})
    by = {name: float(np.mean([r['macro_iou'] for r in rows if r['dataset'] == name]))
          for name in dataset_names if any(r['dataset'] == name for r in rows)}
    return {'case_macro_iou': float(np.mean([r['macro_iou'] for r in rows])),
            'four_multiclass_dataset_macro_iou': float(np.mean([v for k, v in by.items() if k != 'peso'])),
            'dataset_mean': by, 'cases': len(rows)}


def main():
    receipt = HERE / 'SUPERVISED_MODEL_FROZEN.json'
    if receipt.exists():
        raise FileExistsError(receipt)
    ontology = json.loads(ONTOLOGY.read_text())
    sample = json.loads(SAMPLE.read_text())
    splits = json.loads(SPLITS.read_text())
    assert ontology['status'] == 'TRAINING_ONLY_EIGHT_CLASS_ONTOLOGY_FROZEN'
    assert sample['training_cases'] == 397 and sample['validation_cases_opened'] == 0
    assert splits['validation_cases_opened'] == 0
    x = np.load(sample['feature_path'], mmap_mode='r')
    idx = np.load(sample['index_path'])
    old_y, cases, datasets = [idx[k] for k in ('label', 'case_index', 'dataset_id')]
    lut = np.zeros(256, np.uint8)
    for old_id, new_id in ontology['old_to_new'].items():
        if new_id is not None:
            lut[int(old_id)] = new_id
    y = lut[old_y]
    valid = y > 0
    harmony = json.loads(HARM.read_text())['mapping']
    name_to_new = {c['name']: int(cid) for cid, c in ontology['classes'].items()}
    old_names = json.loads((ROOT / 'migration/benchmark1-supervised-virchow2-response-2026-11-10-v1/INPUTS_FROZEN.json').read_text())['classes']
    role_to_new = {role: ontology['old_to_new'][old_id] for old_id, role in old_names.items()}
    eligible = {}
    for dataset in sample['dataset_names']:
        eligible[dataset] = sorted({role_to_new[v['canonical_role']] for v in harmony[dataset].values()
                                    if v['use'] == 'score' and v.get('canonical_role') in role_to_new
                                    and role_to_new[v['canonical_role']] is not None})
    DATA.mkdir(parents=True, exist_ok=True)
    state = HERE / 'SUPERVISED_TRAIN_STATE.json'
    rows = []
    state.write_text(json.dumps({'status': 'RUNNING', 'completed': 0}) + '\n')
    try:
        for fold, split in enumerate(splits['folds'], 1):
            tr = valid & np.isin(cases, split['training_case_indices'])
            te = valid & np.isin(cases, split['heldout_case_indices'])
            xtr, xte = np.asarray(x[tr], np.float32), np.asarray(x[te], np.float32)
            for n_channels, C in CONFIGS:
                tic = monotonic()
                model = fit(xtr, y[tr], n_channels, C)
                fit_seconds = monotonic() - tic
                scores = predict_scores(model, xte)
                for did, name in enumerate(sample['dataset_names']):
                    subset = datasets[te] == did
                    if subset.any():
                        allowed = np.isin(model['classifier'].classes_, eligible[name])
                        assert allowed.any()
                        scores[np.ix_(subset, ~allowed)] = -np.inf
                pred = model['classifier'].classes_[np.argmax(scores, axis=1)]
                metrics = score(pred, y[te], cases[te], datasets[te], sample['dataset_names'])
                rows.append({'fold': fold, 'channels': n_channels, 'C': C,
                             'fit_seconds': fit_seconds, 'metrics': metrics})
                state.write_text(json.dumps({'status': 'RUNNING', 'completed': len(rows),
                                             'last': [fold, n_channels, C]}) + '\n')
                print('TRAINING_FOLD', fold, n_channels, C,
                      metrics['four_multiclass_dataset_macro_iou'], flush=True)
        aggregates = []
        for n_channels, C in CONFIGS:
            part = [r for r in rows if r['channels'] == n_channels and r['C'] == C]
            aggregates.append({'channels': n_channels, 'C': C,
                               'four_multiclass_dataset_macro_iou': float(np.mean([r['metrics']['four_multiclass_dataset_macro_iou'] for r in part])),
                               'mean_fit_seconds': float(np.mean([r['fit_seconds'] for r in part]))})
        aggregates.sort(key=lambda r: (-r['four_multiclass_dataset_macro_iou'], r['mean_fit_seconds']))
        winner = aggregates[0]
        fit_mask = valid
        final = fit(np.asarray(x[fit_mask], np.float32), y[fit_mask], winner['channels'], winner['C'])
        model_path = DATA / 'supervised-eight-class.joblib'
        joblib.dump(final, model_path, compress=3)
        result = {'status': 'EIGHT_CLASS_TRAINING_ONLY_MODEL_FROZEN', 'training_images': 397,
                  'validation_masks_opened': 0, 'model_path': str(model_path),
                  'model_sha256': sha(model_path), 'ontology_sha256': sha(ONTOLOGY),
                  'sample_sha256': sha(SAMPLE), 'splits_sha256': sha(SPLITS),
                  'source_sha256': sha(Path(__file__)), 'selected': winner,
                  'eligible_classes_by_dataset': eligible, 'search': aggregates,
                  'training_tokens_total': int(len(y)), 'training_tokens_used': int(valid.sum()),
                  'training_tokens_excluded_rare': int((~valid).sum())}
        receipt.write_text(json.dumps(result, indent=2) + '\n')
        state.write_text(json.dumps({'status': 'COMPLETE', 'completed': len(rows),
                                     'receipt_sha256': sha(receipt)}) + '\n')
        print('MODEL_FROZEN', winner, flush=True)
    except BaseException as exc:
        state.write_text(json.dumps({'status': 'FAILED', 'completed': len(rows), 'error': repr(exc)}) + '\n')
        raise


if __name__ == '__main__':
    main()
