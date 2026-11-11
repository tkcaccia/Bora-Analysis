#!/usr/bin/env python3
"""Compare supervised Virchow2 heads on source-group-held-out training cases."""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import time

os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
HERE = Path(__file__).resolve().parent
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data/benchmark1-supervised-strategy-search-2026-11-10-v1')
HARM = HERE / 'tissue_label_harmonization.json'
FROZEN = ROOT / 'migration/benchmark1-supervised-virchow2-response-2026-11-10-v1/INPUTS_FROZEN.json'
FAMILIES = ('effect32_logistic', 'effect64_logistic', 'effect128_logistic',
            'anova64_logistic', 'effect64_pls8_logistic', 'effect64_rbf128_logistic')
PRIOR_WEIGHTS = (0.0, 0.5, 1.0)
SEED = 20261110


def sha(path: Path) -> str:
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def model_for_class(x: np.ndarray, y: np.ndarray, train_indices: np.ndarray,
                    dataset: np.ndarray, supported: np.ndarray, cid: int,
                    family: str, seed: int):
    eligible = train_indices[np.isin(dataset[train_indices], supported)]
    pos = eligible[y[eligible] == cid]
    neg = eligible[y[eligible] != cid]
    if len(pos) < 20 or len(neg) < 20:
        raise ValueError(f'Class {cid} has insufficient fitting samples: {len(pos)}, {len(neg)}')
    rng = np.random.default_rng(seed + cid)
    if len(neg) > 4 * len(pos):
        neg = rng.choice(neg, 4 * len(pos), replace=False)
    fit_idx = np.concatenate((pos, neg))
    z = np.asarray(x[fit_idx], dtype=np.float32)
    target = (y[fit_idx] == cid).astype(np.uint8)
    mean = z.mean(axis=0)
    scale = np.maximum(z.std(axis=0), 1e-4)
    diff = z[target == 1].mean(axis=0) - z[target == 0].mean(axis=0)
    if family.startswith('anova'):
        pvar = z[target == 1].var(axis=0)
        nvar = z[target == 0].var(axis=0)
        ranking = np.abs(diff) / np.sqrt(pvar / len(pos) + nvar / len(neg) + 1e-8)
    else:
        ranking = np.abs(diff / scale)
    count = 32 if family.startswith('effect32') else 128 if family.startswith('effect128') else 64
    selected = np.argsort(-ranking, kind='stable')[:count]
    zs = (z[:, selected] - mean[selected]) / scale[selected]
    transformer = None
    if 'pls8' in family:
        transformer = PLSRegression(n_components=min(8, count, len(pos) - 1), scale=False)
        transformer.fit(zs, target)
        zs = transformer.transform(zs)
    elif 'rbf128' in family:
        transformer = RBFSampler(gamma=1 / count, n_components=128, random_state=seed + cid)
        zs = transformer.fit_transform(zs)
    head = LogisticRegression(C=0.3, class_weight='balanced', solver='liblinear',
                              max_iter=500, random_state=seed)
    head.fit(zs, target)
    prior = np.log(len(pos) / len(neg))
    return selected, mean[selected], scale[selected], transformer, head, prior


def predict_class(x: np.ndarray, model) -> np.ndarray:
    selected, mean, scale, transformer, head, _ = model
    z = (np.asarray(x[:, selected], dtype=np.float32) - mean) / scale
    if transformer is not None:
        z = transformer.transform(z)
    return head.decision_function(z).astype(np.float32)


def score_prediction(pred: np.ndarray, actual: np.ndarray,
                     cases: np.ndarray, datasets: np.ndarray, dataset_names: list[str]) -> dict:
    case_rows = []
    for case in np.unique(cases):
        take = cases == case
        truth = actual[take]
        guess = pred[take]
        per = []
        dice = []
        for cid in np.unique(truth):
            tp = int(np.count_nonzero((truth == cid) & (guess == cid)))
            fp = int(np.count_nonzero((truth != cid) & (guess == cid)))
            fn = int(np.count_nonzero((truth == cid) & (guess != cid)))
            per.append(tp / (tp + fp + fn))
            dice.append(2 * tp / (2 * tp + fp + fn))
        dataset_name = dataset_names[int(datasets[take][0])]
        case_rows.append({'case_index': int(case), 'dataset': dataset_name,
                          'tokens': int(take.sum()),
                          'macro_iou': float(np.mean(per)),
                          'macro_dice': float(np.mean(dice)),
                          'accuracy': float(np.mean(guess == truth))})
    by_dataset = defaultdict(list)
    for row in case_rows:
        by_dataset[row['dataset']].append(row['macro_iou'])
    return {'mean_case_macro_iou': float(np.mean([r['macro_iou'] for r in case_rows])),
            'mean_case_macro_dice': float(np.mean([r['macro_dice'] for r in case_rows])),
            'mean_case_accuracy': float(np.mean([r['accuracy'] for r in case_rows])),
            'dataset_mean_case_macro_iou': {k: float(np.mean(v)) for k, v in by_dataset.items()},
            'case_rows': case_rows}


def main() -> None:
    complete = HERE / 'SEARCH_COMPLETE.json'
    if complete.exists():
        raise FileExistsError('Versioned strategy search already complete')
    sample = json.loads((HERE / 'SAMPLE_COMPLETE.json').read_text())
    if sample['status'] != 'TRAINING_SAMPLE_COMPLETE' or sample['validation_cases_opened']:
        raise ValueError('Training sample receipt invalid')
    assert sha(Path(sample['feature_path'])) == sample['feature_sha256']
    assert sha(Path(sample['index_path'])) == sample['index_sha256']
    assert sha(FROZEN) == sample['frozen_inputs_sha256']
    x = np.load(sample['feature_path'], mmap_mode='r', allow_pickle=False)
    idx = np.load(sample['index_path'], allow_pickle=False)
    y = idx['label']; case_index = idx['case_index']; dataset = idx['dataset_id']
    assert x.shape == (139736, 1280) and len(y) == len(case_index) == len(dataset) == len(x)
    harmony = json.loads(HARM.read_text())['mapping']
    classes = json.loads(FROZEN.read_text())['classes']
    supports = {}
    for cid in range(1, 25):
        role = classes[str(cid)]
        datasets = [d for d, mapping in harmony.items() if any(
            v.get('use') == 'score' and v.get('canonical_role') == role
            for v in mapping.values())]
        supports[cid] = np.asarray([sample['dataset_names'].index(d) for d in datasets if d in sample['dataset_names']],
                                   dtype=np.uint8)
    groups = np.asarray(sample['source_groups'])
    assert len(groups) == 397
    splitter = GroupShuffleSplit(n_splits=30, test_size=0.2, random_state=SEED)
    folds = []
    for train_cases, held_cases in splitter.split(np.arange(397), groups=groups):
        train_take = np.isin(case_index, train_cases)
        training_classes = set(np.unique(y[train_take]))
        if training_classes == set(range(1, 25)) and all(np.count_nonzero(y[train_take] == cid) >= 20 for cid in range(1, 25)):
            folds.append((train_cases, held_cases))
        if len(folds) == 2:
            break
    if len(folds) != 2:
        raise ValueError('Cannot make two group-disjoint training development splits')
    result_rows = []
    state = HERE / 'SEARCH_STATE.json'
    state.write_text(json.dumps({'status': 'RUNNING', 'gpu_used': False}) + '\n')
    try:
        for fold_no, (train_cases, held_cases) in enumerate(folds, 1):
            train_idx = np.flatnonzero(np.isin(case_index, train_cases))
            held_idx = np.flatnonzero(np.isin(case_index, held_cases))
            xv = np.asarray(x[held_idx], dtype=np.float32)
            for family in FAMILIES:
                fit_seconds = 0.0
                inference_seconds = 0.0
                logits = np.empty((len(held_idx), 24), np.float32)
                priors = np.empty(24, np.float32)
                for cid in range(1, 25):
                    tick = time.monotonic()
                    fitted = model_for_class(x, y, train_idx, dataset, supports[cid],
                                             cid, family, SEED + fold_no * 1000)
                    fit_seconds += time.monotonic() - tick
                    tick = time.monotonic()
                    logits[:, cid - 1] = predict_class(xv, fitted)
                    inference_seconds += time.monotonic() - tick
                    priors[cid - 1] = fitted[-1]
                elapsed = fit_seconds + inference_seconds
                for weight in PRIOR_WEIGHTS:
                    pred = np.argmax(logits + weight * priors[None, :], axis=1).astype(np.uint8) + 1
                    metrics = score_prediction(pred, y[held_idx], case_index[held_idx],
                                               dataset[held_idx], sample['dataset_names'])
                    result_rows.append({'fold': fold_no, 'family': family, 'prior_weight': weight,
                                        'training_cases': len(train_cases), 'heldout_cases': len(held_cases),
                                        'training_tokens': len(train_idx), 'heldout_tokens': len(held_idx),
                                        'fit_seconds': fit_seconds,
                                        'inference_seconds': inference_seconds,
                                        'inference_ms_per_1000_tokens': inference_seconds * 1000 / len(held_idx),
                                        'metrics': metrics})
                state.write_text(json.dumps({'status': 'RUNNING', 'last_fold': fold_no,
                                             'last_family': family, 'rows': len(result_rows)}) + '\n')
                print('SEARCH', fold_no, family, 'seconds', round(elapsed, 2),
                      'IoU', [round(r['metrics']['mean_case_macro_iou'], 4)
                              for r in result_rows[-3:]], flush=True)
        aggregate = []
        for family in FAMILIES:
            for weight in PRIOR_WEIGHTS:
                parts = [r for r in result_rows if r['family'] == family and r['prior_weight'] == weight]
                assert len(parts) == 2
                aggregate.append({'family': family, 'prior_weight': weight,
                                  'mean_case_macro_iou': float(np.mean([p['metrics']['mean_case_macro_iou'] for p in parts])),
                                  'mean_case_macro_dice': float(np.mean([p['metrics']['mean_case_macro_dice'] for p in parts])),
                                  'fit_seconds': float(np.mean([p['fit_seconds'] for p in parts])),
                                  'inference_seconds': float(np.mean([p['inference_seconds'] for p in parts]))})
        aggregate.sort(key=lambda r: (-r['mean_case_macro_iou'], r['inference_seconds']))
        winner = aggregate[0]
        result = {'status': 'TRAINING_ONLY_STRATEGY_SELECTED', 'gpu_used': False,
                  'validation_cases_opened': 0, 'sample_receipt_sha256': sha(HERE / 'SAMPLE_COMPLETE.json'),
                  'source_sha256': sha(Path(__file__)),
                  'selection_rule': 'Highest mean held-out case macro IoU across two source-group-disjoint 20% training splits; speed breaks exact ties.',
                  'primary_metric': 'case macro IoU over expert-present classes at sampled Virchow2 token centers',
                  'secondary_metrics': ['case macro Dice', 'case accuracy', 'per-dataset case macro IoU', 'fit plus prediction seconds'],
                  'folds': [{'fold': i + 1, 'training_case_indices': tr.tolist(), 'heldout_case_indices': te.tolist()}
                            for i, (tr, te) in enumerate(folds)],
                  'aggregate': aggregate, 'selected': winner, 'rows': result_rows}
        complete.write_text(json.dumps(result, indent=2) + '\n')
        state.write_text(json.dumps({'status': 'COMPLETE', 'receipt_sha256': sha(complete)}) + '\n')
    except BaseException as error:
        state.write_text(json.dumps({'status': 'FAILED', 'error': repr(error),
                                     'completed_rows': len(result_rows)}) + '\n')
        raise


if __name__ == '__main__':
    main()
