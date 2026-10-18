"""Reanalyse frozen whole-group and mask summaries for the CMPB article.

This script performs no inference. Inputs are immutable per-method RESULTS.json
files, the published viewer data.json and the source registry. Uncertainty is
descriptive because these images were repeatedly inspected during development.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


def read(path: Path):
    return json.loads(path.read_text())


def cluster_interval(cases, values, *, seed=19017, repeats=10000):
    """Equal-image mean with a source-group resampling interval."""
    assert len(cases) == len(values) and cases
    groups = defaultdict(list)
    for case, value in zip(cases, values):
        groups[case['source_group']].append(float(value))
    blocks = [np.asarray(v, dtype=float) for _, v in sorted(groups.items())]
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats, dtype=float)
    for i in range(repeats):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        estimates[i] = sum(blocks[j].sum() for j in selected) / sum(len(blocks[j]) for j in selected)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewer', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--neighbor', type=Path, required=True)
    parser.add_argument('--mean', type=Path, required=True)
    parser.add_argument('--spatial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    viewer, registry = read(args.viewer), read(args.registry)
    names = ('neighbor', 'mean', 'spatial')
    results = {name: read(getattr(args, name)) for name in names}
    selected = {c['case'] for c in viewer['cases']}
    index = {name: {c['case']: c for c in result['cases']} for name, result in results.items()}
    assert len(selected) == 116 and all(set(cases) == selected for cases in index.values())
    reg = {c['case']: c for c in registry['cases']}
    assert selected <= set(reg)
    assert all(index['neighbor'][case]['source_group'] == reg[case]['group'] for case in selected)
    assert all(index['neighbor'][case]['split'] == reg[case]['split'] for case in selected)
    assert not ({reg[c]['group'] for c in selected} & {c['group'] for c in registry['cases'] if c['split'] == 'train'})
    assert not ({reg[c]['group'] for c in selected if reg[c]['split'] == 'test'} &
                {reg[c]['group'] for c in selected if reg[c]['split'] == 'validation'})

    out = {'provenance': {'images': len(selected), 'registry_version': registry['version'],
                          'resampling': '10000 source-group bootstrap draws, equal image mean, seed 19017',
                          'interpretation': 'descriptive development-cohort interval, not external generalization'},
           'cohort': {}, 'comparators': {}, 'per_role': {}, 'failure_cases': []}
    for split in ('all', 'test', 'validation'):
        cases = [index['neighbor'][case] for case in sorted(selected)
                 if split == 'all' or reg[case]['split'] == split]
        scores = [c['final']['macro_iou'] for c in cases]
        out['cohort'][split] = {'images': len(cases), 'source_groups': len({c['source_group'] for c in cases}),
                                'mean_final_iou': float(np.mean(scores)),
                                'source_group_bootstrap_95': cluster_interval(cases, scores),
                                'annotated_unassigned_fraction': sum(c['final']['annotated_unassigned_pixels'] for c in cases) /
                                sum(c['final']['valid_pixels'] for c in cases)}
    for dataset in sorted(viewer['dataset_counts']):
        cases = [index['neighbor'][case] for case in sorted(selected) if reg[case]['dataset'] == dataset]
        scores = [c['final']['macro_iou'] for c in cases]
        out['cohort'][dataset] = {'images': len(cases), 'source_groups': len({c['source_group'] for c in cases}),
                                   'mean_final_iou': float(np.mean(scores)),
                                   'source_group_bootstrap_95': cluster_interval(cases, scores)}
    reference = [index['neighbor'][case] for case in sorted(selected)]
    for name in names:
        cases = [index[name][case] for case in sorted(selected)]
        correct = sum(c['group']['top1'] * c['group']['units'] for c in cases)
        units = sum(c['group']['units'] for c in cases)
        scores = [c['final']['macro_iou'] for c in cases]
        out['comparators'][name] = {'groups': units, 'group_correct': round(correct),
                                    'group_accuracy': correct / units,
                                    'balanced_group_accuracy': results[name]['balanced_accuracy'],
                                    'mean_final_iou': float(np.mean(scores)),
                                    'mean_cached_deployment_seconds': float(np.mean([c['cached_deployment_seconds'] for c in cases])),
                                    'mean_warm_head_ms': 1000 * float(np.mean([c['warm_head_seconds'] for c in cases]))}
        if name != 'neighbor':
            diffs = [a['final']['macro_iou'] - b['final']['macro_iou'] for a, b in zip(reference, cases)]
            out['comparators'][name]['neighbor_minus_control_iou'] = float(np.mean(diffs))
            out['comparators'][name]['source_group_bootstrap_95'] = cluster_interval(reference, diffs)
    for role in range(1, len(viewer['roles']) + 1):
        records = [(c, c['final']['per_role'].get(str(role))) for c in reference]
        records = [(c, r) for c, r in records if r and r['truth_pixels'] > 0]
        if records:
            out['per_role'][str(role)] = {'name': viewer['roles'][role - 1],
                                          'annotated_images': len(records),
                                          'total_truth_pixels': sum(r['truth_pixels'] for _, r in records),
                                          'mean_iou_on_annotated_images': float(np.mean([r['iou'] for _, r in records])),
                                          'source_group_bootstrap_95': cluster_interval([c for c, _ in records], [r['iou'] for _, r in records])}
    out['failure_cases'] = [{'case': c['case'], 'dataset': c['dataset'],
                             'mean_final_iou': c['final']['macro_iou'],
                             'unassigned_fraction': c['final']['annotated_unassigned_pixels'] / c['final']['valid_pixels']}
                            for c in sorted(reference, key=lambda c: c['final']['macro_iou'])[:5]]
    out['source_group_audit'] = {'registry_train_groups': len({c['group'] for c in registry['cases'] if c['split'] == 'train'}),
                                  'registry_validation_groups': len({c['group'] for c in registry['cases'] if c['split'] == 'validation'}),
                                  'registry_test_groups': len({c['group'] for c in registry['cases'] if c['split'] == 'test'}),
                                  'known_group_overlap_train_evaluation': 0,
                                  'patient_linkage_complete': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
