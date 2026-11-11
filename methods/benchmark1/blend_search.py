#!/usr/bin/env python3
"""Training-only blend of classwise PLS and class-balanced joint scores."""
import json
import os
from pathlib import Path
import sys
from time import monotonic

os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'migration/benchmark1-supervised-strategy-search-2026-11-10-v1'
V5 = ROOT / 'migration/benchmark1-joint-supervised-2026-11-11-v5'
sys.path.insert(0, str(OLD))
from search import model_for_class, predict_class  # noqa: E402

ALPHAS = (0., .1, .25, .5, .75, 1.)


def support_by_class(sample, classes):
    harmonization = json.loads((ROOT / 'benchmark/methods/benchmark1/tissue_label_harmonization.json').read_text())['mapping']
    out = {}
    for cid in range(1, 25):
        names = [dataset for dataset, mapping in harmonization.items()
                 if any(v.get('use') == 'score' and v.get('canonical_role') == classes[str(cid)]
                        for v in mapping.values())]
        out[cid] = np.asarray([sample['dataset_names'].index(name) for name in names if name in sample['dataset_names']], np.uint8)
    return out


def selected_union(x, y, train_idx, dataset, support, seed):
    selected = []
    for cid in range(1, 25):
        eligible = train_idx[np.isin(dataset[train_idx], support[cid])]
        pos = eligible[y[eligible] == cid]
        neg = eligible[y[eligible] != cid]
        if len(neg) > 4 * len(pos):
            neg = np.random.default_rng(seed + cid).choice(neg, 4 * len(pos), replace=False)
        z = np.asarray(x[np.concatenate((pos, neg))], np.float32)
        rank = np.abs((z[:len(pos)].mean(axis=0) - z[len(pos):].mean(axis=0)) /
                      np.maximum(z.std(axis=0), 1e-4))
        selected.extend(np.argsort(-rank, kind='stable')[:32].tolist())
    return np.unique(np.asarray(selected, np.int32))


def score(pred, actual, cases, dataset, names):
    rows = []
    for case in np.unique(cases):
        take = cases == case
        t, p = actual[take], pred[take]
        iou = [np.count_nonzero((t == cid) & (p == cid)) /
               np.count_nonzero((t == cid) | (p == cid)) for cid in np.unique(t)]
        muscle = None
        if 23 in t:
            muscle = float(np.count_nonzero((t == 23) & (p == 23)) /
                           np.count_nonzero((t == 23) | (p == 23)))
        rows.append({'case': int(case), 'dataset': names[int(dataset[take][0])],
                     'macro_iou': float(np.mean(iou)), 'muscle_iou': muscle})
    means = {name: float(np.mean([r['macro_iou'] for r in rows if r['dataset'] == name]))
             for name in names if any(r['dataset'] == name for r in rows)}
    muscle_values = [r['muscle_iou'] for r in rows if r['muscle_iou'] is not None]
    return {'all_case_macro_iou': float(np.mean([r['macro_iou'] for r in rows])),
            'four_dataset_macro_iou': float(np.mean([v for k, v in means.items() if k != 'peso'])),
            'gcss_macro_iou': means['gcss'],
            'muscle_iou': float(np.mean(muscle_values)) if muscle_values else None,
            'dataset_means': means, 'case_rows': rows}


def main():
    output = HERE / 'SEARCH_COMPLETE.json'
    if output.exists():
        raise FileExistsError(output)
    sample = json.loads((OLD / 'SAMPLE_COMPLETE.json').read_text())
    old_search = json.loads((OLD / 'SEARCH_COMPLETE.json').read_text())
    freeze = json.loads((ROOT / 'migration/benchmark1-supervised-virchow2-response-2026-11-10-v1/INPUTS_FROZEN.json').read_text())
    assert sample['validation_cases_opened'] == old_search['validation_cases_opened'] == 0
    support = support_by_class(sample, freeze['classes'])
    x = np.load(sample['feature_path'], mmap_mode='r')
    idx = np.load(sample['index_path'])
    y, cases, dataset = [idx[k] for k in ('label', 'case_index', 'dataset_id')]
    state = HERE / 'SEARCH_STATE.json'
    state.write_text(json.dumps({'status': 'RUNNING', 'folds_complete': 0}) + '\n')
    rows = []
    try:
        for fold, split in enumerate(old_search['folds'], 1):
            tr = np.flatnonzero(np.isin(cases, split['training_case_indices']))
            te = np.flatnonzero(np.isin(cases, split['heldout_case_indices']))
            xt, xv = np.asarray(x[tr], np.float32), np.asarray(x[te], np.float32)
            tic = monotonic()
            selected = selected_union(x, y, tr, dataset, support, 20261111 + fold * 100)
            mean = xt[:, selected].mean(axis=0)
            std = np.maximum(xt[:, selected].std(axis=0), 1e-4)
            a = (xt[:, selected] - mean) / std
            b = (xv[:, selected] - mean) / std
            count = np.bincount(y[tr], minlength=25).astype(np.float64)
            weight = np.maximum(count[y[tr]], 1) ** (-.5)
            weight /= weight.mean()
            model = LogisticRegression(C=.1, solver='lbfgs', max_iter=150, tol=1e-3)
            model.fit(np.ascontiguousarray(a), y[tr], sample_weight=weight)
            joint = softmax(model.decision_function(np.ascontiguousarray(b)), axis=1)
            pls_scores = np.empty((len(te), 24), np.float32)
            for cid in range(1, 25):
                fitted = model_for_class(x, y, tr, dataset, support[cid], cid,
                                         'effect64_pls8_logistic', 20261110 + fold * 1000)
                pls_scores[:, cid - 1] = predict_class(xv, fitted) + fitted[-1]
            pls = softmax(pls_scores, axis=1)
            elapsed = monotonic() - tic
            for alpha in ALPHAS:
                pred = np.argmax((1 - alpha) * joint + alpha * pls, axis=1).astype(np.uint8) + 1
                metric = score(pred, y[te], cases[te], dataset[te], sample['dataset_names'])
                rows.append({'fold': fold, 'alpha_pls': alpha, 'metrics': metric,
                             'fit_and_predict_seconds': elapsed,
                             'selected_joint_channels': len(selected)})
                print('FOLD', fold, 'alpha', alpha,
                      'four_dataset', metric['four_dataset_macro_iou'],
                      'gcss', metric['gcss_macro_iou'], 'muscle', metric['muscle_iou'], flush=True)
            state.write_text(json.dumps({'status': 'RUNNING', 'folds_complete': fold}) + '\n')
        aggregate = []
        for alpha in ALPHAS:
            part = [r for r in rows if r['alpha_pls'] == alpha]
            aggregate.append({'alpha_pls': alpha,
                              'four_dataset_macro_iou': float(np.mean([r['metrics']['four_dataset_macro_iou'] for r in part])),
                              'all_case_macro_iou': float(np.mean([r['metrics']['all_case_macro_iou'] for r in part])),
                              'gcss_macro_iou': float(np.mean([r['metrics']['gcss_macro_iou'] for r in part])),
                              'muscle_iou': float(np.mean([r['metrics']['muscle_iou'] for r in part]))})
        aggregate.sort(key=lambda r: -r['four_dataset_macro_iou'])
        output.write_text(json.dumps({'status': 'TRAINING_ONLY_BLEND_SEARCH',
                                      'validation_opened': 0, 'aggregate': aggregate,
                                      'rows': rows}, indent=2, allow_nan=False) + '\n')
        state.write_text(json.dumps({'status': 'COMPLETE', 'folds_complete': 2}) + '\n')
    except BaseException as error:
        state.write_text(json.dumps({'status': 'FAILED', 'folds_complete': len(rows) // len(ALPHAS),
                                     'error': repr(error)}) + '\n')
        raise


if __name__ == '__main__':
    main()
