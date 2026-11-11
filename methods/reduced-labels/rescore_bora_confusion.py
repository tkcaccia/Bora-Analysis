#!/usr/bin/env python3
"""Exact reduced-label rescoring of frozen KODAMA–Bora confusion counts."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
HERE = Path(__file__).resolve().parent
SITE = Path('/tmp/bora-analysis-bw106-v2/site/benchmark1-comparators/cases.json')
STRICT = ROOT / 'migration/benchmark1-strict-eval-2026-11-08-v1'
ONTOLOGY = ROOT / 'migration/benchmark-label-reduction-2026-11-11-v1/ONTOLOGY_FROZEN.json'
NEW = ROOT / 'migration/benchmark-label-reduction-2026-11-11-v3/SUPERVISED_CONTOUR_COMPLETE.json'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    target = HERE / 'BORA_REDUCED_CONFUSION_REANALYSIS.json'
    if target.exists():
        raise FileExistsError(target)
    ontology = json.loads(ONTOLOGY.read_text())
    lut = {int(old): (int(new) if new is not None else 0)
           for old, new in ontology['old_to_new'].items()}
    lut[0] = 0
    cases = json.loads(SITE.read_text())['cases']
    supervised = {r['id']: r for r in json.loads(NEW.read_text())['cases']}
    assert len(cases) == len(supervised) == 87
    rows = []
    for site in cases:
        case, cid = site['case'], site['id']
        assert site['dataset'] != 'beetle' and 'pm016' not in case.lower()
        path = STRICT / f'{case}_PILOT_DIAGNOSTIC.json'
        prior = json.loads(path.read_text())
        assert prior['case'] == case and prior['dataset'] == site['dataset']
        matrix = np.zeros((9, 9), dtype=np.int64)
        for old_actual, predictions in prior['confusion'].items():
            actual = lut[int(old_actual)]
            if actual == 0:
                continue
            for old_pred, pixels in predictions.items():
                matrix[actual, lut[int(old_pred)]] += int(pixels)
        per = {}
        for cls in range(1, 9):
            true = int(matrix[cls].sum())
            if not true:
                continue
            tp = int(matrix[cls, cls])
            fp = int(matrix[1:, cls].sum() - tp)
            fn = true - tp
            per[str(cls)] = {'expert_pixels': true, 'intersection': tp,
                             'false_positive': fp, 'false_negative': fn,
                             'iou': tp / (tp + fp + fn),
                             'dice': 2 * tp / (2 * tp + fp + fn)}
        assert per
        scorable = int(matrix[1:].sum())
        expected = supervised[cid]['after']['scored_expert_pixels']
        if scorable != expected:
            raise ValueError(f'Expert denominator differs for {cid}: Bora {scorable}, supervised {expected}')
        rows.append({'id': cid, 'case': case, 'dataset': site['dataset'],
                     'source_confusion_receipt_sha256': sha(path),
                     'scored_expert_pixels': scorable,
                     'expert_present_macro_iou': float(np.mean([v['iou'] for v in per.values()])),
                     'expert_present_macro_dice': float(np.mean([v['dice'] for v in per.values()])),
                     'per_present_class': per})
        print('BORA_RESCORED', len(rows), '/87', cid, flush=True)
    result = {'status': 'EIGHT_CLASS_87_CASE_BORA_CONFUSION_EXACTLY_REANALYZED',
              'method_model_refit': False,
              'method_note': 'KODAMA geometry and the frozen 24-class fastPLS group predictions were not refit. The full expert-by-prediction pixel confusion was mapped to eight classes before computing intersection and union, preserving noncoverage as prediction 0.',
              'ontology_sha256': sha(ONTOLOGY), 'site_case_manifest_sha256': sha(SITE),
              'supervised_contour_sha256': sha(NEW), 'case_count': 87,
              'mean_case_macro_iou': float(np.mean([r['expert_present_macro_iou'] for r in rows])),
              'four_multiclass_dataset_mean_macro_iou': float(np.mean([
                  np.mean([r['expert_present_macro_iou'] for r in rows if r['dataset'] == dataset])
                  for dataset in ('bcss', 'gcss', 'ignite', 'tiger')])),
              'cases': rows, 'source_sha256': sha(Path(__file__))}
    target.write_text(json.dumps(result, indent=2) + '\n')
    print('COMPLETE', result['mean_case_macro_iou'])


if __name__ == '__main__':
    main()
