"""Matched CPU-only whole-group controls for the fixed Bora development cohort.

Inputs are frozen per-group descriptors and source-derived labels. This does
not change the Bora application or run encoder/segmentation inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from bora._vendor.fastpls import PLS


def group_bootstrap(groups, values, repeats=10000, seed=19017):
    unique = sorted(set(groups))
    blocks = [np.asarray(values)[np.asarray(groups) == key] for key in unique]
    rng = np.random.default_rng(seed)
    out = np.empty(repeats)
    for j in range(repeats):
        selected = rng.integers(0, len(blocks), len(blocks))
        out[j] = sum(blocks[i].sum() for i in selected) / sum(len(blocks[i]) for i in selected)
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.inputs, allow_pickle=False) as d:
        xtr, ytr = d['Xtrain'].astype(np.float64), d['ytrain'].astype(np.int64)
        xte, yte = d['Xeval'].astype(np.float64), d['yeval'].astype(np.int64)
        groups, cases = d['source_group'].astype(str), d['case'].astype(str)
    assert xtr.shape == (967, 2560) and xte.shape == (398, 2560)
    assert len(groups) == len(yte) and len(set(cases)) == 116
    results = {}
    with threadpool_limits(limits=2):
        for name, train, test, model in (
            ('neighbor_fastpls', xtr, xte,
             PLS(n_components=24, method='simpls', classifier='lda',
                 scaling='autoscaling', backend='cpu', seed=314159)),
            ('neighbor_ridge', xtr, xte, make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))),
            ('contrast_only_fastpls', xtr[:, 1280:], xte[:, 1280:],
             PLS(n_components=24, method='simpls', classifier='lda',
                 scaling='autoscaling', backend='cpu', seed=314159))):
            t0 = perf_counter(); model.fit(train, ytr); fit_seconds = perf_counter()-t0
            for _ in range(2): model.predict(test)
            times = []
            for _ in range(10):
                t0 = perf_counter(); prediction = np.asarray(model.predict(test), dtype=np.int64); times.append(perf_counter()-t0)
            assert prediction.shape == yte.shape
            correct = prediction == yte
            results[name] = {'groups': len(yte), 'correct_groups': int(correct.sum()),
                             'accuracy': float(correct.mean()),
                             'balanced_accuracy': float(balanced_accuracy_score(yte, prediction)),
                             'fit_seconds': fit_seconds,
                             'warm_predict_ms_per_image': float(np.median(times))*1000/116,
                             'source_group_bootstrap_accuracy_95': group_bootstrap(groups, correct),
                             'prediction_by_case': {case:prediction[cases==case].tolist() for case in sorted(set(cases))}}
    assert results['neighbor_fastpls']['correct_groups'] == 350
    # Compare directly through case masks to preserve registry row alignment.
    for name in ('neighbor_ridge', 'contrast_only_fastpls'):
        differences = []
        current = {case: np.asarray(pred) for case, pred in results[name]['prediction_by_case'].items()}
        reference = {case: np.asarray(pred) for case, pred in results['neighbor_fastpls']['prediction_by_case'].items()}
        cursor = {case: 0 for case in reference}
        for i, case in enumerate(cases):
            j = cursor[case]; cursor[case] += 1
            differences.append(float(reference[case][j] == yte[i]) - float(current[case][j] == yte[i]))
        results[name]['neighbor_minus_control_accuracy'] = float(np.mean(differences))
        results[name]['neighbor_minus_control_source_group_bootstrap_95'] = group_bootstrap(groups, differences)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'cohort':'116 previously inspected development images',
                                       'experiment':'matched whole-group classifier controls',
                                       'trained_with_reference_labels_only_on_training_groups':True,
                                       'results':results},indent=2)+'\n')


if __name__ == '__main__':
    main()
