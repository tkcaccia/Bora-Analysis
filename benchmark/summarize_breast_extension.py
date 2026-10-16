"""Summarize the fixed seven group breast extension."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import tifffile

from benchmark.reanalyze_bora_pilot import pair_metrics
from benchmark.run_bora_pilot import DATA_ROOT

ROOT = DATA_ROOT/'results'/'bora-breast-extension-2026-10-16'


def main() -> None:
    selection=json.loads((ROOT/'selection.json').read_text())
    records=[]
    for item in selection:
        sample=ROOT/item['group_id']/item['dataset']
        case=sample/'error_0'/'tile_56'
        result=json.loads((case/'result.json').read_text())
        truth=tifffile.imread(sample/'normalized_truth.tif')
        coarse=tifffile.imread(case/'coarse.tif')
        refined=np.squeeze(tifffile.imread(case/'refined.ome.tif'))
        c=pair_metrics(coarse,truth);r=pair_metrics(refined,truth)
        records.append({'dataset':item['dataset'],'group_id':item['group_id'],
            'sample_id':item['id'],'width':item['width'],'height':item['height'],
            'coarse_macro_iou':result['coarse']['macro_iou'],
            'bora_macro_iou':result['refined']['macro_iou'],
            'coarse_pair_f1_8':c['f1_8'],'bora_pair_f1_8':r['f1_8'],
            'coarse_pair_f1_16':c['f1_16'],'bora_pair_f1_16':r['f1_16'],
            'coarse_pair_f1_32':c['f1_32'],'bora_pair_f1_32':r['f1_32'],
            'retained_classes':result['refined']['retained_classes'],
            'reference_classes':len(result['refined']['per_class_iou']),
            'bora_wall_seconds':result['bora_wall_seconds_including_vectorization'],
            'geojson_roundtrip_exact':result['geojson']['pixel_agreement']==1,
            'geojson_repeat_identical':result['geojson_repeat']['byte_reproducible']})
    with (ROOT/'reviewer_extension.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summary={}
    for dataset in ('bcss','tiger'):
        rows=[r for r in records if r['dataset']==dataset]
        delta=np.array([r['bora_macro_iou']-r['coarse_macro_iou'] for r in rows])
        fdelta=np.array([r['bora_pair_f1_8']-r['coarse_pair_f1_8'] for r in rows
                         if r['bora_pair_f1_8'] is not None and r['coarse_pair_f1_8'] is not None])
        summary[dataset]={'n_source_groups':len(rows),'macro_delta_median':float(np.median(delta)),
            'macro_improved':int((delta>1e-9).sum()),'macro_worsened':int((delta<-1e-9).sum()),
            'pair_f1_delta_median':float(np.median(fdelta)) if len(fdelta) else None,
            'pair_f1_improved':int((fdelta>1e-9).sum()),
            'pair_f1_worsened':int((fdelta<-1e-9).sum()),
            'all_classes_retained':all(r['retained_classes']==r['reference_classes'] for r in rows),
            'all_roundtrip_exact':all(r['geojson_roundtrip_exact'] for r in rows)}
    (ROOT/'reviewer_extension_summary.json').write_text(json.dumps(summary,indent=2))
    print(ROOT/'reviewer_extension_summary.json')

if __name__=='__main__':main()
