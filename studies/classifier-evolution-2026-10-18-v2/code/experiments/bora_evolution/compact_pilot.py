"""Compare chord fitting with stronger coverage simplification on real ROIs."""
from pathlib import Path
import argparse,time
from collections import defaultdict
import numpy as np
from bora.compact_geojson import fit_coverage,boundary_stats,geometry_agreement,compact
from experiments.bora_group.bora_geojson import render
from experiments.bora_supervised.registry import load,save
from experiments.bora_supervised.evaluate import broad_truth,pixel_scores
from threadpoolctl import threadpool_limits

CASES=['tiger_06','tiger_15','bcss_17','bcss_71','bcss_102','ignite_02','gcss_09','puma_16','peso_23']

def run(study):
    study=Path(study);group=study.parent/'bora-anonymous-groups-2026-10-18-v1';registry=load(study.parent/'tissue-supervised-2026-10-18-v1/registry-v1/REGISTRY.json')
    rows={r['case']:r for r in registry['cases']};out=study/'compact-pilot';results=[]
    for case in CASES:
        row=rows[case];source=load(group/'geojson-bora-fill20-v5'/case/'R02-c24.geojson');native=tuple(row['sample']['native_shape_yx']);mpp=row['sample']['native_mpp_xy']
        original=render(source,native);truth=broad_truth(row);baseline=boundary_stats(source);base_iou=pixel_scores(truth,original,len(registry['ontology']['roles']))['macro_iou']
        for method in ('coverage-vw','shared-rdp'):
            for tolerance in (.5,1.,2.,3.,4.):
                start=time.monotonic();candidate=fit_coverage(source,mpp,tolerance,method);stats=boundary_stats(candidate);agreement=geometry_agreement(source,candidate)
                raster=render(candidate,native);score=pixel_scores(truth,raster,len(registry['ontology']['roles']))['macro_iou']
                item=dict(case=case,method=method,tolerance_um=tolerance,before=baseline,after=stats,agreement=agreement,
                    native_pixel_changes=int(np.count_nonzero(original!=raster)),native_tissue_pixels=int(np.count_nonzero(original)),
                    baseline_iou=base_iou,final_iou=score,iou_delta=score-base_iou,seconds=time.monotonic()-start)
                results.append(item)
        candidate=compact(source,mpp,native,tolerance_um=3.);raster=render(candidate,native);report=candidate['compact_boundary_report']
        save(out/case/'compact.geojson',candidate)
        results.append(dict(case=case,method='guarded-shared-rdp',tolerance_um=3.,before=baseline,after=report['after'],agreement=report['agreement'],
            native_pixel_changes=int(np.count_nonzero(original!=raster)),native_tissue_pixels=int(np.count_nonzero(original)),
            baseline_iou=base_iou,final_iou=pixel_scores(truth,raster,len(registry['ontology']['roles']))['macro_iou'],seconds=report['runtime_seconds']))
        print(__import__('json').dumps(dict(pilot_complete=case,guarded_tolerance=report['selected_tolerance_um'],before_edges=baseline['boundary_segments'],after_edges=report['after']['boundary_segments'])),flush=True)
    grouped=defaultdict(list)
    for r in results:grouped[r['method'],r['tolerance_um']].append(r)
    summary=[]
    for (method,tolerance),items in grouped.items():
        before=sum(r['before']['boundary_segments'] for r in items);after=sum(r['after']['boundary_segments'] for r in items)
        summary.append(dict(method=method,tolerance_um=tolerance,cases=len(items),before_edges=before,after_edges=after,reduction=1-after/before,
            before_stairs=sum(r['before']['short_stair_corners'] for r in items),after_stairs=sum(r['after']['short_stair_corners'] for r in items),
            minimum_class_iou=min(r['agreement']['minimum_class_iou'] for r in items),mean_iou_delta=float(np.mean([r['final_iou']-r['baseline_iou'] for r in items])),
            mean_seconds=float(np.mean([r['seconds'] for r in items]))))
    save(out/'REPORT.json',dict(cases=CASES,results=results,summary=summary,expert_masks_not_used_by_fitting=True))
    print(__import__('json').dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True)
    with threadpool_limits(limits=2):run(p.parse_args().study)
