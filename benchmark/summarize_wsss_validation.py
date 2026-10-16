"""Summarize the WSSS4LUAD validation ROI cohort without patient-level inference."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import tifffile

from benchmark.reanalyze_bora_pilot import pair_metrics

ROOT = Path('/mnt/sata_ssd/Bora-benchmark/results/bora-wsss-validation-2026-10-16')
ROWS = [json.loads(line) for line in
        Path('/mnt/sata_ssd/Bora-benchmark/prepared/wsss4luad.jsonl').open()]


def main() -> None:
    data=[]; omitted=[]
    for source in ROWS:
        sample=ROOT/source['id']/'wsss4luad'
        if not sample.exists():
            omitted.append(source['id'])
            continue
        truth=tifffile.imread(sample/'normalized_truth.tif')
        for tile in (56,90,216):
            case=sample/'error_0'/f'tile_{tile}'
            record=case/'result.json'
            if not record.exists():
                omitted.append(f"{source['id']} tile {tile}")
                continue
            old=json.loads(record.read_text())
            coarse=tifffile.imread(case/'coarse.tif')
            refined=np.squeeze(tifffile.imread(case/'refined.ome.tif'))
            c=pair_metrics(coarse,truth);r=pair_metrics(refined,truth)
            data.append({'sample_id':source['id'],'tile':tile,
                'height':int(truth.shape[0]),'width':int(truth.shape[1]),
                'n_reference_classes':int(np.unique(truth[truth>0]).size),
                'baseline_macro_iou':old['coarse']['macro_iou'],
                'bora_macro_iou':old['refined']['macro_iou'],
                'baseline_pair_f1_8':c['f1_8'],'bora_pair_f1_8':r['f1_8'],
                'baseline_pair_f1_16':c['f1_16'],'bora_pair_f1_16':r['f1_16'],
                'baseline_pair_f1_32':c['f1_32'],'bora_pair_f1_32':r['f1_32'],
                'reference_boundary_pixels':c['reference_boundary_pixels'],
                'coarse_boundary_pixels':c['predicted_boundary_pixels'],
                'refined_boundary_pixels':r['predicted_boundary_pixels'],
                'retained_classes':old['refined']['retained_classes'],
                'bora_wall_seconds':old['bora_wall_seconds_including_vectorization'],
                'vector_seconds':old['geojson_conversion_seconds'],
                'geojson_pixel_agreement':old['geojson']['pixel_agreement'],
                'geojson_repeat_identical':old['geojson_repeat']['byte_reproducible']})
    output=ROOT/'reviewer_cohort'
    output.mkdir(exist_ok=True)
    with (output/'roi_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    summary={'selection':'All acquired WSSS4LUAD validation ROI pairs with at least two observed valid classes; patient/slide linkage unavailable.',
             'omitted_or_incomplete':omitted,'pilot_roi':'validation--25','by_tile':{}}
    for tile in (56,90,216):
        # Exclude the pilot ROI from the descriptive extension because it was
        # used during method development. This is not a patient-disjoint test.
        rows=[x for x in data if x['tile']==tile and x['sample_id']!='validation--25']
        delta=np.array([x['bora_macro_iou']-x['baseline_macro_iou'] for x in rows])
        fdelta=np.array([x['bora_pair_f1_8']-x['baseline_pair_f1_8'] for x in rows
                         if x['bora_pair_f1_8'] is not None and x['baseline_pair_f1_8'] is not None])
        times=np.array([x['bora_wall_seconds'] for x in rows])
        summary['by_tile'][str(tile)]={'n_rois':len(rows),
          'macro_iou_delta_median':float(np.median(delta)),
          'macro_iou_delta_iqr':[float(x) for x in np.percentile(delta,[25,75])],
          'macro_iou_improved':int((delta>1e-9).sum()),
          'macro_iou_worsened':int((delta<-1e-9).sum()),
          'macro_iou_unchanged':int((abs(delta)<=1e-9).sum()),
          'n_with_defined_pair_f1':int(len(fdelta)),
          'pair_f1_delta_median':float(np.median(fdelta)) if len(fdelta) else None,
          'pair_f1_improved':int((fdelta>1e-9).sum()),
          'pair_f1_worsened':int((fdelta<-1e-9).sum()),
          'median_bora_wall_seconds':float(np.median(times)),
          'all_geojson_roundtrip_exact':all(x['geojson_pixel_agreement']==1 for x in rows),
          'all_repeat_geojson_identical':all(x['geojson_repeat_identical'] for x in rows)}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(output/'summary.json')

if __name__=='__main__':main()
