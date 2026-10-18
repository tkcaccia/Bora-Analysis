"""CPU-only native-resolution tumor–stroma interface audit.

Run where the provider masks and final Bora TIFFs are retained. No encoder,
classifier or segmentation inference is performed. Interface F1 treats stroma
and inflamed stroma as one boundary partner, using an 8 µm tolerance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import tifffile


def tumor_stroma_edge(labels, valid):
    tumor = labels == 1
    stroma = (labels == 3) | (labels == 4)
    edge = np.zeros(labels.shape, bool)
    h = ((tumor[:, :-1] & stroma[:, 1:]) | (stroma[:, :-1] & tumor[:, 1:])) & valid[:, :-1] & valid[:, 1:]
    v = ((tumor[:-1, :] & stroma[1:, :]) | (stroma[:-1, :] & tumor[1:, :])) & valid[:-1, :] & valid[1:, :]
    edge[:, :-1] |= h; edge[:, 1:] |= h
    edge[:-1, :] |= v; edge[1:, :] |= v
    return edge


def score(prediction, truth, mpp_xy, tolerance_um=8.0):
    valid = truth > 0
    p = tumor_stroma_edge(prediction, valid)
    t = tumor_stroma_edge(truth, valid)
    npred, ntruth = int(p.sum()), int(t.sum())
    result = {'predicted_boundary_pixels': npred, 'reference_boundary_pixels': ntruth,
              'f1_8um': None, 'assd_um': None}
    if not ntruth:
        return result
    if not npred:
        result['f1_8um'] = 0.0
        return result
    spacing = (float(mpp_xy[1]), float(mpp_xy[0]))
    pred_distance = distance_transform_edt(~t, sampling=spacing)[p]
    truth_distance = distance_transform_edt(~p, sampling=spacing)[t]
    precision = float(np.mean(pred_distance <= tolerance_um))
    recall = float(np.mean(truth_distance <= tolerance_um))
    result['f1_8um'] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result['assd_um'] = float((pred_distance.sum() + truth_distance.sum()) / (npred + ntruth))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--geometry-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    registry = {c['case']: c for c in json.loads(args.registry.read_text())['cases']}
    data = json.loads(args.results.read_text())
    records = []
    for case in data['cases']:
        row = registry[case['case']]
        raw = tifffile.imread(row['reference_path'])
        truth = np.zeros(raw.shape, np.uint8)
        for value, role in row['ontology_raw_to_role'].items():
            truth[raw == int(value)] = int(role)
        final = tifffile.imread(Path(case['vector']).with_name('mask.tif'))
        assert final.shape == truth.shape
        result = score(final, truth, row['sample']['native_mpp_xy'])
        coarse = tifffile.imread(args.geometry_root/'cases'/case['case']/'coarse.working.tif')
        key = json.loads((args.geometry_root/'evaluation_keys'/(case['case']+'.json')).read_text())
        identities = dict(zip(key['input_anonymous_ids'], key['input_source_ids']))
        unmapped = set(np.unique(coarse)) - {0} - set(identities)
        assert not unmapped, (case['case'], unmapped)
        coarse_roles = np.zeros(coarse.shape, np.uint8)
        for anonymous, raw_value in identities.items():
            coarse_roles[coarse == anonymous] = int(row['ontology_raw_to_role'].get(str(raw_value), 0))
        if coarse_roles.shape != truth.shape:
            coarse_roles = np.asarray(Image.fromarray(coarse_roles).resize((truth.shape[1],truth.shape[0]), Image.Resampling.NEAREST))
        coarse_result = score(coarse_roles, truth, row['sample']['native_mpp_xy'])
        records.append({'case':case['case'],'dataset':case['dataset'],'source_group':case['source_group'],
                        'mpp_xy':row['sample']['native_mpp_xy'],**result,
                        'expert_decoded_coarse':coarse_result})
    evaluated = [c for c in records if c['f1_8um'] is not None]
    summary = {'cases':len(records),'cases_with_reference_interface':len(evaluated),
               'cases_with_missing_predicted_interface':sum(c['predicted_boundary_pixels']==0 for c in evaluated),
               'mean_f1_8um':float(np.mean([c['f1_8um'] for c in evaluated])),
               'median_f1_8um':float(np.median([c['f1_8um'] for c in evaluated])),
               'mean_assd_um_when_both_present':float(np.mean([c['assd_um'] for c in evaluated if c['assd_um'] is not None])),
               'definition':'Direct 4-neighbor tumor (role 1) versus stroma or inflamed stroma (roles 3,4), evaluated on annotated pixels; 8 um physical tolerance.'}
    coarse_values=[c['expert_decoded_coarse']['f1_8um'] for c in evaluated]
    paired=np.asarray([c['f1_8um']-c['expert_decoded_coarse']['f1_8um'] for c in evaluated])
    summary['expert_decoded_coarse_mean_f1_8um']=float(np.mean(coarse_values))
    summary['final_minus_expert_decoded_coarse_mean_f1_8um']=float(np.mean(paired))
    summary['coarse_comparison_caveat']='Coarse tile identities were decoded from the expert-origin permutation; this is a privileged diagnostic, not a deployable class predictor.'
    units={k:[] for k in sorted({c['source_group'] for c in evaluated})}
    for c,value in zip(evaluated,paired):units[c['source_group']].append(float(value))
    blocks=[np.asarray(v) for v in units.values()]
    rng=np.random.default_rng(19017);draws=np.empty(10000)
    for i in range(len(draws)):
        chosen=rng.integers(0,len(blocks),len(blocks))
        draws[i]=sum(blocks[j].sum() for j in chosen)/sum(len(blocks[j]) for j in chosen)
    summary['final_minus_coarse_source_group_bootstrap_95']=[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'summary':summary,'records':records},indent=2)+'\n')


if __name__ == '__main__':
    main()
