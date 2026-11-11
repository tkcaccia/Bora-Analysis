#!/usr/bin/env python3
"""One locked 87-case evaluation of the frozen joint Virchow2 classifier."""
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
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.special import softmax

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
HERE = Path(__file__).resolve().parent
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data/benchmark1-joint-supervised-2026-11-11-v7')
OUT = DATA / 'validation'
FROZEN = ROOT / 'migration/benchmark1-supervised-virchow2-response-2026-11-10-v1/INPUTS_FROZEN.json'
HARM = ROOT / 'benchmark/methods/benchmark1/tissue_label_harmonization.json'
OLD_RESPONSES = Path('/mnt/sata_ssd/ChatGPT/Bora-Data/benchmark1-supervised-virchow2-response-2026-11-10-v1')
BASE = Path('/mnt/sata_ssd/ChatGPT/Bora-Data/benchmark1-supervised-mask-overlay-score-2026-11-10-v1/SCORES_AND_FIGURES.json')
PLS = ROOT / 'migration/benchmark1-supervised-strategy-search-2026-11-10-v2/VALIDATION_COMPLETE.json'
JOINT = ROOT / 'migration/benchmark1-joint-supervised-2026-11-11-v5/VALIDATION_COMPLETE.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def score(pred, truth):
    valid = truth > 0
    per = {}
    for cid in np.unique(truth[valid]):
        actual, guessed = truth == cid, pred == cid
        tp = int(np.count_nonzero(valid & actual & guessed))
        fp = int(np.count_nonzero(valid & ~actual & guessed))
        fn = int(np.count_nonzero(valid & actual & ~guessed))
        per[str(int(cid))] = {'expert_pixels': tp + fn, 'intersection': tp,
                              'false_positive': fp, 'false_negative': fn,
                              'iou': tp / (tp + fp + fn),
                              'dice': 2 * tp / (2 * tp + fp + fn)}
    return {'expert_present_macro_iou': float(np.mean([v['iou'] for v in per.values()])),
            'expert_present_macro_dice': float(np.mean([v['dice'] for v in per.values()])),
            'scored_pixel_accuracy': float(np.mean(pred[valid] == truth[valid])),
            'scored_expert_pixels': int(valid.sum()), 'per_present_class': per}


def main():
    receipt_path = HERE / 'VALIDATION_COMPLETE.json'
    if receipt_path.exists() or OUT.exists():
        raise FileExistsError('No in-place validation retry')
    fitted = json.loads((HERE / 'FINAL_BLEND_FROZEN.json').read_text())
    assert fitted['status'] == 'TRAINING_SELECTED_JOINT_PLS_BLEND_FROZEN'
    assert fitted['validation_cases_opened'] == 0
    assert sha(fitted['joint_model_path']) == fitted['joint_model_sha256']
    assert sha(fitted['pls_model_path']) == fitted['pls_model_sha256']
    model = joblib.load(fitted['joint_model_path'])
    pls_models = joblib.load(fitted['pls_model_path'])
    selected, mean, std, classifier = [model[k] for k in ('selected', 'mean', 'std', 'classifier')]
    assert np.array_equal(classifier.classes_, np.arange(1, 25))
    assert set(pls_models) == set(range(1, 25))
    selected_union = np.unique(np.concatenate([selected] + [m[0] for m in pls_models.values()]))
    freeze = json.loads(FROZEN.read_text())
    validation = {r['case']: r for r in freeze['rows'] if r['split'] == 'validation'}
    baseline = json.loads(BASE.read_text())
    old_pls = json.loads(PLS.read_text())
    old_pls_cases = {r['case']: r for r in old_pls['cases']}
    old_joint = json.loads(JOINT.read_text())
    old_joint_cases = {r['case']: r for r in old_joint['cases']}
    assert len(validation) == len(old_pls_cases) == len(baseline['cases']) == 87
    assert set(validation) == set(old_pls_cases) == set(old_joint_cases) == {r['case'] for r in baseline['cases']}
    harmony = json.loads(HARM.read_text())['mapping']
    class_ids = {name: int(cid) for cid, name in freeze['classes'].items()}
    OUT.mkdir(parents=True)
    state = HERE / 'VALIDATION_STATE.json'
    state.write_text(json.dumps({'status': 'RUNNING', 'complete': 0}) + '\n')
    records = []
    try:
        for old in baseline['cases']:
            case, case_id = old['case'], old['id']
            source = validation[case]
            geometry = json.loads((OLD_RESPONSES / case_id / 'RESPONSE_COMPLETE.json').read_text())
            working_wh = tuple(geometry['working_size_wh'])
            display_wh = tuple(geometry['combined_mask']['display_size_wh'])
            feature_path = Path(source['feature_path'])
            if sha(feature_path) != source['feature_sha256']:
                raise ValueError('Feature SHA mismatch: ' + case_id)
            grid = np.load(feature_path, mmap_mode='r')
            if tuple(grid.shape) != tuple(source['feature_shape']):
                raise ValueError('Feature shape mismatch: ' + case_id)
            finite = np.isfinite(np.asarray(grid[:, :, selected_union])).all(axis=2)
            if not finite.any():
                raise ValueError('No finite features: ' + case_id)
            if finite.all():
                nearest = np.arange(grid.shape[0] * grid.shape[1])
            else:
                _, near = distance_transform_edt(~finite, return_indices=True)
                nearest = np.ravel_multi_index(near, grid.shape[:2]).ravel()
            flat = grid.reshape(-1, 1280)
            started = monotonic()
            x = np.asarray(flat[nearest[:, None], selected[None, :]], np.float32)
            x = (x - mean) / std
            joint_logits = classifier.decision_function(x).reshape(*grid.shape[:2], 24)
            joint_display = np.empty((*display_wh[::-1], 24), np.float32)
            pls_display = np.empty_like(joint_display)
            for cid in range(1, 25):
                token = gaussian_filter(joint_logits[:, :, cid - 1], sigma=0.75, mode='nearest')
                full = Image.fromarray(token.astype(np.float32), mode='F').resize(
                    (grid.shape[1] * 14, grid.shape[0] * 14), Image.Resampling.BICUBIC)
                joint_display[:, :, cid - 1] = np.asarray(full.crop((0, 0, *working_wh)).resize(
                    display_wh, Image.Resampling.BICUBIC), np.float32)
                channels, pmean, pstd, transformer, head, prior = pls_models[cid]
                z = np.asarray(flat[nearest[:, None], channels[None, :]], np.float32)
                z = (z - pmean) / pstd
                if transformer is not None:
                    z = transformer.transform(z)
                token = head.decision_function(z).reshape(grid.shape[:2]).astype(np.float32)
                token += fitted['pls_prior_weight'] * prior
                token = gaussian_filter(token, sigma=.75, mode='nearest')
                full = Image.fromarray(token, 'F').resize(
                    (grid.shape[1] * 14, grid.shape[0] * 14), Image.Resampling.BICUBIC)
                pls_display[:, :, cid - 1] = np.asarray(full.crop((0, 0, *working_wh)).resize(
                    display_wh, Image.Resampling.BICUBIC), np.float32)
            blended = fitted['alpha_joint'] * softmax(joint_display, axis=2)
            blended += fitted['alpha_pls'] * softmax(pls_display, axis=2)
            winner = np.argmax(blended, axis=2).astype(np.uint8) + 1
            inference_seconds = monotonic() - started
            path = OUT / f'{case_id}-class-mask.png'
            Image.fromarray(winner, 'L').save(path, optimize=True)
            frame_wh = tuple(old['scoring_shape_yx'][::-1])
            pred = np.asarray(Image.fromarray(winner, 'L').resize(frame_wh, Image.Resampling.NEAREST))
            mask_path = Path(old['source_mask_path'])
            if sha(mask_path) != old['source_mask_sha256']:
                raise ValueError('Expert SHA mismatch: ' + case_id)
            with Image.open(mask_path) as image:
                raw = np.asarray(image.resize(frame_wh, Image.Resampling.NEAREST))
            truth = np.zeros(raw.shape, np.uint8)
            for raw_code in np.unique(raw):
                spec = harmony[old['dataset']].get(str(int(raw_code)))
                if spec is None and raw_code != 0:
                    raise ValueError('Unmapped expert code: ' + case_id)
                if spec and spec['use'] == 'score' and spec.get('canonical_role') in class_ids:
                    truth[raw == raw_code] = class_ids[spec['canonical_role']]
            result = score(pred, truth)
            if result['scored_expert_pixels'] != old['scored_expert_pixels']:
                raise ValueError('Denominator mismatch: ' + case_id)
            records.append({'id': case_id, 'case': case, 'dataset': old['dataset'],
                            'prediction_path': str(path), 'prediction_sha256': sha(path),
                            'feature_sha256': source['feature_sha256'],
                            'expert_sha256': old['source_mask_sha256'],
                            'inference_seconds': inference_seconds,
                            'finite_feature_tokens': int(finite.sum()),
                            'nearest_imputed_tokens': int((~finite).sum()),
                            'baseline_32_macro_iou': old['expert_present_macro_iou'],
                            'baseline_pls8_macro_iou': old_pls_cases[case]['new']['expert_present_macro_iou'],
                            'baseline_joint_macro_iou': old_joint_cases[case]['new']['expert_present_macro_iou'],
                            'new': result})
            state.write_text(json.dumps({'status': 'RUNNING', 'complete': len(records),
                                         'last': case_id}) + '\n')
            print('VALIDATED', len(records), '/87', case_id,
                  round(result['expert_present_macro_iou'], 4), flush=True)
        new = np.asarray([r['new']['expert_present_macro_iou'] for r in records])
        old = np.asarray([r['baseline_pls8_macro_iou'] for r in records])
        rng = np.random.default_rng(20261111)
        boot = rng.integers(0, len(records), size=(10000, len(records)))
        delta = np.mean((new - old)[boot], axis=1)
        per_dataset = {name: float(np.mean([r['new']['expert_present_macro_iou'] for r in records if r['dataset'] == name]))
                       for name in sorted({r['dataset'] for r in records})}
        receipt = {'status': '87_CASE_JOINT_PLS_BLEND_VALIDATION_SCORED',
                   'model_receipt_sha256': sha(HERE / 'FINAL_BLEND_FROZEN.json'),
                   'input_freeze_sha256': sha(FROZEN), 'baseline_pls8_receipt_sha256': sha(PLS),
                   'baseline_joint_receipt_sha256': sha(JOINT),
                   'validation_cases_used_for_selection': 0, 'validation_cases_scored': 87,
                   'mean_case_macro_iou': float(new.mean()),
                   'baseline_pls8_mean_case_macro_iou': float(old.mean()),
                   'paired_mean_macro_iou_difference': float(np.mean(new - old)),
                   'paired_difference_case_bootstrap_95pct': np.percentile(delta, [2.5, 97.5]).tolist(),
                   'mean_case_macro_dice': float(np.mean([r['new']['expert_present_macro_dice'] for r in records])),
                   'mean_case_pixel_accuracy': float(np.mean([r['new']['scored_pixel_accuracy'] for r in records])),
                   'dataset_mean_case_macro_iou': per_dataset,
                   'median_inference_seconds_per_case': float(np.median([r['inference_seconds'] for r in records])),
                   'total_inference_seconds': float(sum(r['inference_seconds'] for r in records)),
                   'imputation_rule': 'Copy the nearest finite feature token within the same image before model inference; no expert label used.',
                   'cases': records}
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        state.write_text(json.dumps({'status': 'COMPLETE', 'complete': 87,
                                     'receipt_sha256': sha(receipt_path)}) + '\n')
        print('COMPLETE', receipt['mean_case_macro_iou'], flush=True)
    except BaseException as error:
        state.write_text(json.dumps({'status': 'FAILED', 'complete': len(records),
                                     'error': repr(error)}) + '\n')
        raise


if __name__ == '__main__':
    main()
