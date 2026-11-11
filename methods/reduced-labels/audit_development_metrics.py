#!/usr/bin/env python3
"""Deterministic audit of frozen eight-class development case receipts.

Provide the six frozen input receipts; no raw image or model files are loaded.
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('supervised', 'prior_24_head', 'kodama_bora', 'external', 'ontology', 'original_supervised'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    return parser.parse_args()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def interval(values, seed=21112, n=10000):
    rng = random.Random(seed)
    size = len(values)
    boot = sorted(mean(values[rng.randrange(size)] for _ in range(size)) for _ in range(n))
    return [boot[int(n * .025)], boot[int(n * .975)]]


def stratified_interval(by_dataset, seed=21113, n=10000):
    rng = random.Random(seed)
    groups = list(by_dataset.values())
    boot = sorted(mean(mean(v[rng.randrange(len(v))] for _ in v) for v in groups) for _ in range(n))
    return [boot[int(n * .025)], boot[int(n * .975)]]


def main():
    args = parse_args()
    sources = {name: getattr(args, name) for name in ('supervised', 'prior_24_head', 'kodama_bora', 'external', 'ontology', 'original_supervised')}
    d = {name: json.loads(path.read_text()) for name, path in sources.items()}
    ids = [r['id'] for r in d['supervised']['cases']]
    assert len(ids) == len(set(ids)) == 87
    indexed = {name: {r['id']: r for r in obj['cases']} for name, obj in d.items() if 'cases' in obj}
    assert all(set(rows) == set(ids) for rows in indexed.values())
    all_datasets = defaultdict(list)
    case_scores = []
    class_rows = defaultdict(list)
    scored_pixels = 0
    image_pixels = 0
    single_class = []
    for id_ in ids:
        sup = indexed['supervised'][id_]
        prior = indexed['prior_24_head'][id_]
        old = indexed['original_supervised'][id_]
        bora = indexed['kodama_bora'][id_]
        external = indexed['external'][id_]
        dataset = sup['dataset']
        assert all(r['dataset'] == dataset for r in (prior, old, bora, external))
        val = sup['after']['expert_present_macro_iou']
        assert abs(val - prior['new_8_head_iou']) < 1e-12
        assert abs(val - prior['prior_24_head_remapped_iou'] - prior['delta']) < 1e-12
        all_datasets[dataset].append(val)
        scored_pixels += old['metrics']['scored_expert_pixels']
        image_pixels += old['metrics']['all_pixels']
        if dataset == 'peso':
            single_class.append(val)
        case_scores.append({
            'id': id_, 'dataset': dataset,
            'supervised': val,
            'prior_24_head': prior['prior_24_head_remapped_iou'],
            'kodama_bora': bora['expert_present_macro_iou'],
            **{m: external['metrics'][m]['expert_present_macro_iou'] for m in ('unet', 'deeplabv3plus', 'vista_path', 'medsam')},
        })
        for cid, c in sup['after']['per_present_class'].items():
            class_rows[cid].append({'case': id_, 'iou': c['iou'],
                                    'expert_pixels': old['metrics']['per_present_class'][cid]['expert_pixels']})
    assert len(single_class) == 19
    assert abs(mean(r['supervised'] for r in case_scores) - d['supervised']['mean_macro_iou_after']) < 1e-12
    assert abs(mean(r['kodama_bora'] for r in case_scores) - d['kodama_bora']['mean_case_macro_iou']) < 1e-12
    paired = [r['supervised'] - r['prior_24_head'] for r in case_scores]
    class_summary = {}
    for cid, vals in sorted(class_rows.items(), key=lambda x: int(x[0])):
        scores = [v['iou'] for v in vals]
        class_summary[d['ontology']['classes'][cid]['name']] = {
            'class_id': int(cid), 'expert_present_cases': len(vals),
            'expert_pixels': sum(v['expert_pixels'] for v in vals),
            'mean_case_iou': mean(scores), 'case_bootstrap_95pct': interval(scores, 21112 + int(cid)),
        }
    multi = {k: v for k, v in all_datasets.items() if k != 'peso'}
    out = {
        'status': 'FROZEN_BENCHMARK1_CASE_RECEIPTS_REANALYZED',
        'source_sha256': {k: sha(v) for k, v in sources.items()},
        'case_count': len(ids), 'dataset_counts': {k: len(v) for k, v in all_datasets.items()},
        'scored_pixels_total': scored_pixels, 'image_pixels_total': image_pixels,
        'scored_pixel_fraction_pooled': scored_pixels / image_pixels,
        'supervised_all_87_case_macro_iou': mean(r['supervised'] for r in case_scores),
        'supervised_all_87_case_bootstrap_95pct': interval([r['supervised'] for r in case_scores]),
        'supervised_four_multiclass_equal_dataset_macro_iou': mean(mean(v) for v in multi.values()),
        'supervised_four_multiclass_stratified_case_bootstrap_95pct': stratified_interval(multi),
        'prior_24_head_remapped_all_87_case_macro_iou': mean(r['prior_24_head'] for r in case_scores),
        'supervised_minus_prior_24_head_paired_delta': mean(paired),
        'supervised_minus_prior_24_head_case_bootstrap_95pct': interval(paired),
        'paired_improved_tied_worsened': [sum(v > 1e-12 for v in paired), sum(abs(v) <= 1e-12 for v in paired), sum(v < -1e-12 for v in paired)],
        'single_class_peso': {'case_count': len(single_class), 'scores_equal_one': sum(v == 1 for v in single_class), 'mean_iou': mean(single_class)},
        'per_class': class_summary,
        'by_dataset': {k: {'cases': len(v), 'mean_iou': mean(v)} for k, v in all_datasets.items()},
        'method_case_means': {m: mean(r[m] for r in case_scores) for m in ('supervised', 'prior_24_head', 'kodama_bora', 'unet', 'deeplabv3plus', 'vista_path', 'medsam')},
        'interpretation': 'Development-only reanalysis. No method was refitted; method tasks differ. These intervals are case resampling summaries, not independent test confidence intervals.',
    }
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k in ('status', 'supervised_all_87_case_macro_iou', 'supervised_all_87_case_bootstrap_95pct', 'supervised_four_multiclass_equal_dataset_macro_iou', 'supervised_four_multiclass_stratified_case_bootstrap_95pct', 'supervised_minus_prior_24_head_paired_delta', 'supervised_minus_prior_24_head_case_bootstrap_95pct', 'single_class_peso', 'scored_pixel_fraction_pooled')}, indent=2))


if __name__ == '__main__':
    main()
