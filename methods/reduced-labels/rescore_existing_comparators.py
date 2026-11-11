#!/usr/bin/env python3
"""Re-evaluate immutable comparator predictions under the frozen broad ontology.

This is an output-space reanalysis, not a claim that external models were refit.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from validate_supervised import sha, score

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE = Path(__file__).resolve().parent
SITE = Path('/tmp/bora-analysis-bw106-v2/site/benchmark1-comparators')
BASE = DATA / 'benchmark1-supervised-mask-overlay-score-2026-11-10-v1/SCORES_AND_FIGURES.json'
HARM = ROOT / 'benchmark/methods/benchmark1/tissue_label_harmonization.json'
OLD = ROOT / 'migration/benchmark1-supervised-virchow2-response-2026-11-10-v1/INPUTS_FROZEN.json'
VISTA = ROOT / 'benchmark/comparators/VISTA_BRCA_CLASS_MAP_PREDECLARED_2026-11-07.json'
VISTA_NAMES = ('tumor', 'stroma', 'lymphocytic_infiltrate', 'necrosis_or_debris',
               'blood', 'metaplasia_NOS', 'fat', 'plasma_cells',
               'other_immune_infiltrate', 'lymphatics', 'blood_vessel',
               'normal_acinus_or_duct', 'dcis')


def resize_nn(mask, shape):
    if tuple(mask.shape) == tuple(shape):
        return mask
    return np.asarray(Image.fromarray(mask).resize(shape[::-1], Image.Resampling.NEAREST))


def main():
    output = HERE / 'EXISTING_COMPARATORS_REDUCED_REANALYSIS.json'
    if output.exists():
        raise FileExistsError(output)
    ontology_path = HERE / 'ONTOLOGY_FROZEN.json'
    ontology = json.loads(ontology_path.read_text())
    old_names = json.loads(OLD.read_text())['classes']
    role_to_new = {name: ontology['old_to_new'][cid] for cid, name in old_names.items()}
    lut = np.zeros(256, dtype=np.uint8)
    for old_id, new_id in ontology['old_to_new'].items():
        if new_id is not None:
            lut[int(old_id)] = new_id
    harmony = json.loads(HARM.read_text())['mapping']
    vista_map = {r['vista_name']: r for r in json.loads(VISTA.read_text())['matched']}
    site_rows = {r['case']: r for r in json.loads((SITE / 'cases.json').read_text())['cases']}
    baselines = json.loads(BASE.read_text())['cases']
    assert len(site_rows) == len(baselines) == 87
    rows = []
    for base in baselines:
        case, cid, dataset = base['case'], base['id'], base['dataset']
        site = site_rows[case]
        shape = tuple(base['scoring_shape_yx'])
        expert_path = Path(base['source_mask_path'])
        assert sha(expert_path) == base['source_mask_sha256']
        with Image.open(expert_path) as im:
            raw = np.asarray(im.resize(shape[::-1], Image.Resampling.NEAREST))
        truth = np.zeros(shape, np.uint8)
        for raw_id in np.unique(raw):
            spec = harmony[dataset].get(str(int(raw_id)))
            if spec and spec['use'] == 'score':
                new_id = role_to_new.get(spec.get('canonical_role'))
                if new_id is not None:
                    truth[raw == raw_id] = new_id
        support_path = SITE / site['he_background_support']['image']
        assert sha(support_path) == site['he_background_support']['sha256']
        with Image.open(support_path) as im:
            support = resize_nn(np.asarray(im.getchannel('A')), shape) > 0
        preds = {}
        for method, path in (
            ('unet', DATA / 'comparators-2026-11-07-v3/results/unet-balanced-v3/predictions' / f'{case}.npz'),
            ('deeplabv3plus', DATA / 'comparators-deeplabv3plus-2026-11-08-v1/results/deeplabv3plus-v1/predictions' / f'{case}.npz')):
            assert sha(path) == site['source_sha256'][method]
            with np.load(path, allow_pickle=False) as z:
                mask = z['prediction']
            pred = lut[resize_nn(mask.astype(np.uint8), shape)]
            pred[~support] = 0
            preds[method] = pred
        vista_path = DATA / 'comparators-2026-11-07-v1/results/vista-v2/cropped-to-true-frame' / f'{case}.png'
        assert sha(vista_path) == site['source_sha256']['vista']
        with Image.open(vista_path) as im:
            raw_vista = resize_nn(np.asarray(im), shape)
        vista_pred = np.zeros(shape, np.uint8)
        for code in np.unique(raw_vista):
            if not (1 <= code <= len(VISTA_NAMES)):
                continue
            mapping = vista_map[VISTA_NAMES[code - 1]]
            if 'applies_to' in mapping and dataset not in mapping['applies_to']:
                continue
            new_id = role_to_new.get(mapping['bora_role'])
            if new_id is not None:
                vista_pred[raw_vista == code] = new_id
        vista_pred[~support] = 0
        preds['vista_path'] = vista_pred
        med_path = DATA / 'comparators-2026-11-07-v1/results/medsam-guided-v1' / f'{case}.npz'
        assert sha(med_path) == site['source_sha256']['medsam']
        with np.load(med_path, allow_pickle=False) as z:
            components = {name: resize_nn(z[name].astype(np.uint8), shape) > 0
                          for name in ('tumor', 'stroma', 'necrosis')}
        overlap = sum(component.astype(np.uint8) for component in components.values())
        med_pred = np.zeros(shape, np.uint8)
        for name, new_id in [('tumor', 1), ('stroma', 2), ('necrosis', 3)]:
            med_pred[components[name] & (overlap == 1)] = new_id
        med_pred[~support] = 0
        preds['medsam'] = med_pred
        metrics = {method: score(pred, truth) for method, pred in preds.items()}
        med_truth = truth.copy()
        med_truth[~np.isin(med_truth, [1, 2, 3])] = 0
        metrics['medsam_three_class_reference'] = score(med_pred, med_truth)
        rows.append({'id': cid, 'case': case, 'dataset': dataset,
                     'expert_sha256': base['source_mask_sha256'],
                     'support_sha256': site['he_background_support']['sha256'],
                     'input_prediction_sha256': {k: site['source_sha256'][k] for k in ('unet', 'deeplabv3plus', 'vista', 'medsam')},
                     'metrics': metrics})
        print('RESCORED', len(rows), '/87', cid, flush=True)
    methods = list(rows[0]['metrics'])
    result = {'status': 'FOUR_EXISTING_COMPARATORS_EIGHT_CLASS_OUTPUT_SPACE_REANALYZED',
              'ontology_sha256': sha(ontology_path), 'case_count': 87,
              'method_model_refit': False,
              'method_note': 'U-Net, DeepLabV3+, VISTA-PATH and MedSAM prior verified predictions were mapped to the frozen eight-class ontology and rescored against the same expert pixels and Bora display support. This is not retraining.',
              'mean_case_macro_iou': {method: float(np.mean([r['metrics'][method]['expert_present_macro_iou'] for r in rows]))
                                      for method in methods},
              'four_multiclass_dataset_mean_case_macro_iou': {
                  method: float(np.mean([np.mean([r['metrics'][method]['expert_present_macro_iou']
                                                  for r in rows if r['dataset'] == dataset])
                                         for dataset in sorted({r['dataset'] for r in rows} - {'peso'})]))
                  for method in methods},
              'cases': rows, 'source_sha256': sha(Path(__file__))}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('COMPLETE', result['mean_case_macro_iou'], flush=True)


if __name__ == '__main__':
    main()
