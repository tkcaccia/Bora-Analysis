#!/usr/bin/env python3
"""Compare class-balanced feature unions and jointly calibrated PLS scores."""
import json
import os
from pathlib import Path
import sys
from time import monotonic

os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'migration/benchmark1-supervised-strategy-search-2026-11-10-v1'
sys.path.insert(0, str(OLD))
from search import model_for_class, predict_class  # noqa: E402


def measure(pred, truth, cases, datasets, names):
    rows = []
    for case in np.unique(cases):
        take = cases == case
        y, p = truth[take], pred[take]
        ious = [np.count_nonzero((y == cid) & (p == cid)) /
                np.count_nonzero((y == cid) | (p == cid)) for cid in np.unique(y)]
        muscle = None
        if 23 in y:
            muscle = float(np.count_nonzero((y == 23) & (p == 23)) /
                           np.count_nonzero((y == 23) | (p == 23)))
        rows.append({'case': int(case), 'dataset': names[int(datasets[take][0])],
                     'macro_iou': float(np.mean(ious)), 'muscle_iou': muscle})
    by_dataset = {name: float(np.mean([r['macro_iou'] for r in rows if r['dataset'] == name]))
                  for name in names if any(r['dataset'] == name for r in rows)}
    muscle_values = [r['muscle_iou'] for r in rows if r['muscle_iou'] is not None]
    return {'mean_case_macro_iou': float(np.mean([r['macro_iou'] for r in rows])),
            'four_multiclass_dataset_macro_iou': float(np.mean([v for k, v in by_dataset.items() if k != 'peso'])),
            'gcss_macro_iou': by_dataset['gcss'],
            'mean_muscle_iou': float(np.mean(muscle_values)) if muscle_values else None,
            'muscle_cases': len(muscle_values), 'by_dataset': by_dataset, 'case_rows': rows}


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
            raise ValueError(f'Class {cid} insufficient in fold')
        if len(neg) > 4 * len(pos):
            neg = np.random.default_rng(seed + cid).choice(neg, 4 * len(pos), replace=False)
        z = np.asarray(x[np.concatenate((pos, neg))], np.float32)
        scale = np.maximum(z.std(axis=0), 1e-4)
        rank = np.abs((z[:len(pos)].mean(axis=0) - z[len(pos):].mean(axis=0)) / scale)
        all_selected.extend(np.argsort(-rank, kind='stable')[:k].tolist())
    return np.unique(np.asarray(all_selected, np.int32))


def fit_joint(xtrain, ytrain, xheld, selected):
    mean = xtrain[:, selected].mean(axis=0)
    std = np.maximum(xtrain[:, selected].std(axis=0), 1e-4)
    a = (np.asarray(xtrain[:, selected], np.float32) - mean) / std
    b = (np.asarray(xheld[:, selected], np.float32) - mean) / std
    count = np.bincount(ytrain, minlength=25).astype(np.float64)
    weight = np.maximum(count[ytrain], 1) ** (-.5)
    weight /= weight.mean()
    clf = LogisticRegression(C=.1, solver='lbfgs', max_iter=150, tol=1e-3)
    clf.fit(np.ascontiguousarray(a), ytrain, sample_weight=weight)
    return clf.predict(np.ascontiguousarray(b))


def main():
    receipt = HERE / 'SEARCH_COMPLETE.json'
    if receipt.exists():
        raise FileExistsError('No in-place retry')
    sample = json.loads((OLD / 'SAMPLE_COMPLETE.json').read_text())
    prior = json.loads((OLD / 'SEARCH_COMPLETE.json').read_text())
    frozen = json.loads((ROOT / 'migration/benchmark1-supervised-virchow2-response-2026-11-10-v1/INPUTS_FROZEN.json').read_text())
    assert sample['validation_cases_opened'] == prior['validation_cases_opened'] == 0
    support = supports(sample, frozen['classes'])
    x = np.load(sample['feature_path'], mmap_mode='r')
    idx = np.load(sample['index_path'])
    y, cases, datasets = [idx[k] for k in ('label', 'case_index', 'dataset_id')]
    assert x.shape == (139736, 1280)
    state = HERE / 'SEARCH_STATE.json'
    state.write_text(json.dumps({'status': 'RUNNING', 'complete': 0}) + '\n')
    rows = []
    try:
        for fold, split in enumerate(prior['folds'], 1):
            train_idx = np.flatnonzero(np.isin(cases, split['training_case_indices']))
            held_idx = np.flatnonzero(np.isin(cases, split['heldout_case_indices']))
            xt = np.asarray(x[train_idx], np.float32)
            xv = np.asarray(x[held_idx], np.float32)
            for k in (32, 64):
                tic = monotonic()
                selected = classwise_union(x, y, train_idx, datasets, support, k, 20261111 + fold * 100)
                pred = fit_joint(xt, y[train_idx], xv, selected)
                elapsed = monotonic() - tic
                m = measure(pred, y[held_idx], cases[held_idx], datasets[held_idx], sample['dataset_names'])
                rows.append({'fold': fold, 'family': f'classwise_union{k}_joint_logistic',
                             'selected_union_channels': len(selected), 'seconds': elapsed,
                             'metrics': m})
                state.write_text(json.dumps({'status': 'RUNNING', 'complete': len(rows)}) + '\n')
                print('UNION', fold, k, len(selected), m['four_multiclass_dataset_macro_iou'],
                      'GCSS', m['gcss_macro_iou'], 'muscle', m['mean_muscle_iou'], flush=True)
            tic = monotonic()
            tr_scores = np.empty((len(train_idx), 24), np.float32)
            te_scores = np.empty((len(held_idx), 24), np.float32)
            for cid in range(1, 25):
                fitted = model_for_class(x, y, train_idx, datasets, support[cid], cid,
                                         'effect64_pls8_logistic', 20261110 + fold * 1000)
                tr_scores[:, cid - 1] = predict_class(xt, fitted) + fitted[-1]
                te_scores[:, cid - 1] = predict_class(xv, fitted) + fitted[-1]
            count = np.bincount(y[train_idx], minlength=25).astype(np.float64)
            weight = np.maximum(count[y[train_idx]], 1) ** (-.5)
            weight /= weight.mean()
            clf = LogisticRegression(C=.1, solver='lbfgs', max_iter=150, tol=1e-3)
            clf.fit(tr_scores, y[train_idx], sample_weight=weight)
            pred = clf.predict(te_scores)
            elapsed = monotonic() - tic
            m = measure(pred, y[held_idx], cases[held_idx], datasets[held_idx], sample['dataset_names'])
            rows.append({'fold': fold, 'family': 'classwise_pls8_joint_calibration',
                         'selected_union_channels': None, 'seconds': elapsed, 'metrics': m})
            state.write_text(json.dumps({'status': 'RUNNING', 'complete': len(rows)}) + '\n')
            print('PLS_CAL', fold, m['four_multiclass_dataset_macro_iou'],
                  'GCSS', m['gcss_macro_iou'], 'muscle', m['mean_muscle_iou'], flush=True)
        aggregate = []
        for family in sorted({r['family'] for r in rows}):
            part = [r for r in rows if r['family'] == family]
            aggregate.append({'family': family,
                              'four_multiclass_dataset_macro_iou': float(np.mean([r['metrics']['four_multiclass_dataset_macro_iou'] for r in part])),
                              'mean_case_macro_iou': float(np.mean([r['metrics']['mean_case_macro_iou'] for r in part])),
                              'gcss_macro_iou': float(np.mean([r['metrics']['gcss_macro_iou'] for r in part])),
                              'mean_muscle_iou': float(np.mean([r['metrics']['mean_muscle_iou'] for r in part if r['metrics']['mean_muscle_iou'] is not None])),
                              'mean_seconds': float(np.mean([r['seconds'] for r in part]))})
        aggregate.sort(key=lambda v: (-v['four_multiclass_dataset_macro_iou'], v['mean_seconds']))
        result = {'status': 'TRAINING_ONLY_CLASSWISE_JOINT_SEARCH', 'validation_opened': 0,
                  'selection_rule': 'Highest four-multiclass-dataset mean case macro IoU across two frozen source-group-disjoint training splits; speed tie-breaker',
                  'aggregate': aggregate, 'rows': rows}
        receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        state.write_text(json.dumps({'status': 'COMPLETE', 'complete': len(rows)}) + '\n')
    except BaseException as error:
        state.write_text(json.dumps({'status': 'FAILED', 'complete': len(rows), 'error': repr(error)}) + '\n')
        raise


if __name__ == '__main__':
    main()
