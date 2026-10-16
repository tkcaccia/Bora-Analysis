"""Independent boundary and comparator audit of the frozen Bora pilot outputs.

This analysis does not rerun Bora or modify the acquisition inventories.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, median_filter

from benchmark.run_bora_pilot import DATA_ROOT, SAMPLES, metrics

ROOT = DATA_ROOT / 'results' / 'bora-pilot-independent-2026-10-16'
OUT = ROOT / 'reviewer_reanalysis'


def pair_borders(labels: np.ndarray, valid: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    pairs: dict[tuple[int, int], np.ndarray] = {}
    for axis in (0, 1):
        a = labels[:-1, :] if axis == 0 else labels[:, :-1]
        b = labels[1:, :] if axis == 0 else labels[:, 1:]
        va = valid[:-1, :] if axis == 0 else valid[:, :-1]
        vb = valid[1:, :] if axis == 0 else valid[:, 1:]
        edge = (a != b) & (a > 0) & (b > 0) & va & vb
        if not edge.any():
            continue
        low = np.minimum(a, b).astype(np.uint32)
        high = np.maximum(a, b).astype(np.uint32)
        codes = (low << 16) | high
        for code in np.unique(codes[edge]):
            key = (int(code >> 16), int(code & 65535))
            layer = pairs.setdefault(key, np.zeros(labels.shape, bool))
            selected = edge & (codes == code)
            if axis == 0:
                layer[:-1] |= selected
                layer[1:] |= selected
            else:
                layer[:, :-1] |= selected
                layer[:, 1:] |= selected
    return pairs


def pair_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    valid = truth > 0
    pm, tm = pair_borders(pred, valid), pair_borders(truth, valid)
    tolerances = (8, 16, 32)
    matched_pred = {radius: 0 for radius in tolerances}
    matched_truth = {radius: 0 for radius in tolerances}
    total_pred = total_truth = 0
    distances: list[np.ndarray] = []
    absent_pairs = []
    per_pair = {}
    for key in sorted(pm.keys() | tm.keys()):
        pb = pm.get(key, np.zeros(truth.shape, bool))
        tb = tm.get(key, np.zeros(truth.shape, bool))
        npred, ntruth = int(pb.sum()), int(tb.sum())
        total_pred += npred
        total_truth += ntruth
        if npred and ntruth:
            dp = distance_transform_edt(~tb)[pb]
            dt = distance_transform_edt(~pb)[tb]
            for radius in tolerances:
                matched_pred[radius] += int(np.count_nonzero(dp <= radius))
                matched_truth[radius] += int(np.count_nonzero(dt <= radius))
            distances.extend([dp, dt])
        else:
            absent_pairs.append(key)
        per_pair[f'{key[0]}-{key[1]}'] = {'prediction_pixels': npred,
                                         'reference_pixels': ntruth}
    boundary_scores = {}
    for radius in tolerances:
        precision = matched_pred[radius] / total_pred if total_pred else None
        recall = matched_truth[radius] / total_truth if total_truth else None
        f1 = (2*precision*recall/(precision+recall)
              if precision is not None and recall is not None and precision+recall
              else 0.0 if precision is not None and recall is not None else None)
        boundary_scores[f'f1_{radius}'] = f1
        boundary_scores[f'precision_{radius}'] = precision
        boundary_scores[f'recall_{radius}'] = recall
    return {**boundary_scores,
            'predicted_boundary_pixels': total_pred, 'reference_boundary_pixels': total_truth,
            'missing_or_extra_pairs': [f'{a}-{b}' for a,b in absent_pairs],
            'assd_px_observed_pairs': float(np.mean(np.concatenate(distances))) if distances else None,
            'hd95_px_observed_pairs': float(np.percentile(np.concatenate(distances), 95)) if distances else None,
            'per_pair': per_pair}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    records = []
    for dataset in SAMPLES:
        sample = ROOT / dataset
        truth = tifffile.imread(sample / 'normalized_truth.tif')
        for error in (0, 1, 5):
            for tile in (56, 90, 216):
                case = sample / f'error_{error}' / f'tile_{tile}'
                coarse = tifffile.imread(case / 'coarse.tif')
                refined = np.squeeze(tifffile.imread(case / 'refined.ome.tif'))
                cm, rm = pair_metrics(coarse, truth), pair_metrics(refined, truth)
                comps = {}
                for width in (5, 9):
                    began = time.perf_counter()
                    smoothed = median_filter(coarse, size=width)
                    seconds = time.perf_counter()-began
                    comps[str(width)] = {'macro_iou': metrics(smoothed, truth)['macro_iou'],
                                         'pair_boundary_f1_8': pair_metrics(smoothed, truth)['f1_8'],
                                         'runtime_seconds': seconds}
                record = {'dataset':dataset, 'sample_id':SAMPLES[dataset], 'tile':tile,
                          'error':error, 'coarse':cm, 'bora':rm, 'median':comps,
                          'coarse_macro_iou':metrics(coarse, truth)['macro_iou'],
                          'bora_macro_iou':metrics(refined, truth)['macro_iou']}
                records.append(record)
    (OUT/'reviewer_reanalysis.json').write_text(json.dumps(records, indent=2))
    with (OUT/'reviewer_reanalysis.csv').open('w', newline='') as f:
        keys = ['dataset','tile','error','coarse_macro_iou','bora_macro_iou',
                'coarse_pair_f1_8','bora_pair_f1_8','coarse_pair_f1_16','bora_pair_f1_16',
                'coarse_pair_f1_32','bora_pair_f1_32','median5_macro_iou','median5_pair_f1_8',
                'median9_macro_iou','median9_pair_f1_8']
        writer=csv.DictWriter(f,keys);writer.writeheader()
        for r in records:
            writer.writerow({'dataset':r['dataset'],'tile':r['tile'],'error':r['error'],
              'coarse_macro_iou':r['coarse_macro_iou'],'bora_macro_iou':r['bora_macro_iou'],
              'coarse_pair_f1_8':r['coarse']['f1_8'],'bora_pair_f1_8':r['bora']['f1_8'],
              'coarse_pair_f1_16':r['coarse']['f1_16'],'bora_pair_f1_16':r['bora']['f1_16'],
              'coarse_pair_f1_32':r['coarse']['f1_32'],'bora_pair_f1_32':r['bora']['f1_32'],
              'median5_macro_iou':r['median']['5']['macro_iou'],
              'median5_pair_f1_8':r['median']['5']['pair_boundary_f1_8'],
              'median9_macro_iou':r['median']['9']['macro_iou'],
              'median9_pair_f1_8':r['median']['9']['pair_boundary_f1_8']})
    for dataset,tile in [('bcss',56),('tiger',56),('wsss4luad',90)]:
        case=ROOT/dataset/'error_0'/f'tile_{tile}'
        image=tifffile.imread(ROOT/dataset/'image.tif')
        truth=tifffile.imread(ROOT/dataset/'normalized_truth.tif')
        coarse=tifffile.imread(case/'coarse.tif')
        refined=np.squeeze(tifffile.imread(case/'refined.ome.tif'))
        fig,axes=plt.subplots(1,4,figsize=(16,4),layout='constrained')
        axes[0].imshow(image);axes[0].set_title('H&E')
        for ax,labels,name in zip(axes[1:],(truth,coarse,refined),('Reference','Tile baseline','Bora')):
            ax.imshow(image)
            border=np.zeros(labels.shape,bool)
            border[:-1] |= (labels[:-1]!=labels[1:]) & (labels[:-1]>0) & (labels[1:]>0)
            border[:,:-1] |= (labels[:,:-1]!=labels[:,1:]) & (labels[:,:-1]>0) & (labels[:,1:]>0)
            overlay=np.zeros((*labels.shape,4),dtype=float)
            overlay[border]=[1,0.85,0,0.95]
            ax.imshow(overlay)
            ax.set_title(name)
        for ax in axes:ax.axis('off')
        fig.savefig(OUT/f'{dataset}_{tile}_overlay.png',dpi=180)
        plt.close(fig)
    print(OUT)


if __name__ == '__main__':
    main()
