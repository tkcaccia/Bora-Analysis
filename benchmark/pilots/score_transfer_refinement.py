"""Score frozen coarse and Bora masks only after their prediction files exist."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile


def scores(truth: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> dict:
    roles=sorted(set(np.unique(truth[valid])) | set(np.unique(pred[valid])))
    out={}
    for role in roles:
        if not role:
            continue
        intersection=np.count_nonzero(valid & (truth==role) & (pred==role))
        union=np.count_nonzero(valid & ((truth==role) | (pred==role)))
        denom=np.count_nonzero(valid & (truth==role))+np.count_nonzero(valid & (pred==role))
        out[str(int(role))]=dict(iou=intersection/union if union else 0,
                                 dice=2*intersection/denom if denom else 0)
    return dict(macro_iou=float(np.mean([r['iou'] for r in out.values()])),
                macro_dice=float(np.mean([r['dice'] for r in out.values()])),
                accuracy=float(np.mean(pred[valid]==truth[valid])),by_role=out)


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--registry',type=Path,required=True)
    args=p.parse_args()
    root=args.root
    frozen=json.loads((root/'tiles/PREDICTIONS_FROZEN.json').read_text())
    result=root/'refined/tiger_15'
    report=json.loads((result/'REFINEMENT_COMPLETE.json').read_text())
    assert report['reference_read'] is False and report['source_case']==frozen['source_case']
    row=next(r for r in json.loads(args.registry.read_text())['cases'] if r['case']=='tiger_15')
    raw=tifffile.imread(row['reference_path'])
    truth=np.zeros(raw.shape,np.uint8)
    for label,role in row['ontology_raw_to_role'].items():
        truth[raw==int(label)]=int(role)
    coarse=tifffile.imread(root/'inputs/tiger_15/cases/tiger_15/coarse.tif')
    baseline=np.where(coarse==1,1,np.where(coarse==2,3,0)).astype(np.uint8)
    pred=tifffile.imread(result/'mask.ome.tif')
    assert truth.shape==baseline.shape==pred.shape
    valid=truth>0
    payload=dict(case='tiger_15',source_training_case=frozen['source_case'],
                 source_training_group=frozen['source_group'],target_group=row['group'],
                 annotated_pixels=int(valid.sum()),coarse=scores(truth,baseline,valid),
                 bora=scores(truth,pred,valid),
                 delta_macro_iou=float(scores(truth,pred,valid)['macro_iou']-scores(truth,baseline,valid)['macro_iou']),
                 scope='One exploratory target ROI; source-trained tile head plus Bora geometry and area-majority segment readout; not independent validation.')
    (result/'SCORES.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload))


if __name__=='__main__':
    main()
