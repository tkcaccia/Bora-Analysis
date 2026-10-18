"""Common native-pixel comparison for square tiles and H&E superpixels.

This evaluates fixed validation predictions. A test evaluator is separate and
requires all test predictions and the validation selection to be frozen first.
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
import tifffile
from .registry import load,save,sha
from .eligibility import eligible


def unit_map(root,row):
    with np.load(Path(root)/'cases'/row['case']/'units.npz') as u:
        if 'region_map' in u:
            return cv2.resize(u['region_map'].astype(np.int32),tuple(row['sample']['working_shape_yx'][::-1]),interpolation=cv2.INTER_NEAREST),u['tissue_fraction'].copy()
        labels=np.empty(row['sample']['working_shape_yx'],np.int32)
        for i,(x,y,w,h) in enumerate(u['boxes']):labels[y:y+h,x:x+w]=i
        return labels,u['tissue_fraction'].copy()


def native_rank(root,row,ranks,j=0):
    units,tf=unit_map(root,row);values=ranks[:,j].copy();values[tf<.02]=0
    working=values[units].astype(np.uint8)
    native=cv2.resize(working,tuple(row['sample']['native_shape_yx'][::-1]),interpolation=cv2.INTER_NEAREST)
    support=tifffile.imread(Path(row['original_path']).parent/'tissue.native.tif')>0
    native[~support]=0
    return native


def broad_truth(row):
    if sha(row['reference_path'])!=row['reference_sha256']:raise ValueError('Reference changed')
    raw=tifffile.imread(row['reference_path']);truth=np.zeros(raw.shape,np.uint8)
    for code,role in row['ontology_raw_to_role'].items():truth[raw==int(code)]=role
    return truth


def pixel_scores(truth,prediction,n_roles,top5_hit=None):
    valid=truth>0
    if truth.shape!=prediction.shape or np.any(prediction>n_roles):raise ValueError('Invalid label map')
    conf=np.bincount((truth[valid].astype(int)*(n_roles+1)+prediction[valid]),minlength=(n_roles+1)**2).reshape(n_roles+1,n_roles+1)
    target=conf.sum(1);proposed=conf.sum(0);intersection=np.diag(conf);union=target+proposed-intersection
    used=np.flatnonzero(union[1:]>0)+1;observed=np.flatnonzero(target[1:]>0)+1
    correct=int(intersection[1:].sum());count=int(target[1:].sum())
    return dict(valid_pixels=count,pixel_top1=correct/count if count else None,
        pixel_top5=float(top5_hit[valid].mean()) if count and top5_hit is not None else None,
        macro_iou=float(np.mean(intersection[used]/union[used])) if len(used) else None,
        gt_role_macro_iou=float(np.mean(intersection[observed]/union[observed])) if len(observed) else None,
        annotated_unassigned_pixels=int(conf[1:,0].sum()),
        per_role={str(g):dict(iou=float(intersection[g]/union[g]),truth_pixels=int(target[g]),prediction_pixels=int(proposed[g])) for g in used})


def choose_formulations(cycles,root,regions):
    candidates=[]
    for cycle in map(Path,cycles):
        seen=set()
        for result in load(cycle/'ELIGIBLE_RESULTS.json')['ranking']:
            family=result.get('method',result['name'].split('-')[0])
            if family in seen:continue
            seen.add(family)
            candidates.append(dict(name=result['name'],cycle=str(cycle),family=family,
                feature_root=str(regions if family in ('S08','S09') else root),
                selection='Best validation unit balanced accuracy within this formulation',
                head_seconds=result.get('mean_head_seconds',result.get('mean_native_prediction_seconds'))))
    return candidates


def aggregate(cases,names):
    ranking=[]
    for name in names:
        rows=[r['models'][name] for r in cases];eligible=[r for r in rows if r['macro_iou'] is not None]
        groups={}
        for r in cases:
            s=r['models'][name]
            if s['macro_iou'] is not None:groups.setdefault(r['group'],[]).append(s['macro_iou'])
        ranking.append(dict(name=name,images=len(eligible),
            mean_native_pixel_macro_iou=float(np.mean([r['macro_iou'] for r in eligible])),
            group_mean_native_pixel_macro_iou=float(np.mean([np.mean(v) for v in groups.values()])),
            mean_pixel_top1=float(np.mean([r['pixel_top1'] for r in eligible])),
            mean_pixel_top5=float(np.mean([r['pixel_top5'] for r in eligible if r.get('pixel_top5') is not None]))
                if any(r.get('pixel_top5') is not None for r in eligible) else None,
            per_dataset={d:float(np.mean([r['models'][name]['macro_iou'] for r in cases if r['dataset']==d and r['models'][name]['macro_iou'] is not None]))
                         for d in sorted({r['dataset'] for r in cases}) if any(r['dataset']==d and r['models'][name]['macro_iou'] is not None for r in cases)}))
    return sorted(ranking,key=lambda r:-r['mean_native_pixel_macro_iou'])


def validation(root,regions,cycles,output):
    root,regions,output=map(Path,(root,regions,output))
    if output.exists():raise ValueError('Preserve validation report')
    registry=load(root/'REGISTRY.json');models=choose_formulations(cycles,root,regions)
    save(output/'PROTOCOL.json',dict(registry_sha256=sha(root/'REGISTRY.json'),models=models,
        criterion='Mean per-image native-pixel macro IoU over the union of truth and predicted tissue roles inside expert-mapped annotation. Spurious predicted roles count as zero-IoU classes.',
        background='Same original-H&E native tissue support for every model. Annotation pixels removed by it count as errors.',
        test_annotations_read=False,source_sha256=sha(Path(__file__))))
    # Every classifier prediction is already frozen in its own cycle.
    cases=[];n=len(registry['ontology']['roles'])
    for row in registry['cases']:
        if row['split']!='validation' or not eligible(row):continue
        truth=broad_truth(row);record=dict(case=row['case'],dataset=row['dataset'],group=row['group'],models={})
        for model in models:
            folder=Path(model['cycle'])/model['name'];prediction=folder/(row['case']+'.npz')
            frozen=load(folder/'PREDICTIONS.json')
            expected=next(v['prediction_sha256'] for v in frozen['predictions'] if v['case']==row['case'])
            if sha(prediction)!=expected:raise ValueError('Prediction changed')
            with np.load(prediction) as p:ranks=p['top5']
            top=native_rank(model['feature_root'],row,ranks)
            hit=top==truth
            for j in range(1,ranks.shape[1]):hit|=native_rank(model['feature_root'],row,ranks,j)==truth
            record['models'][model['name']]=pixel_scores(truth,top,n,hit)
            out=output/'cases'/row['case'];out.mkdir(parents=True,exist_ok=True)
            tifffile.imwrite(out/(model['name']+'.tif'),top,compression='deflate')
        cases.append(record)
    ranking=aggregate(cases,[m['name'] for m in models])
    save(output/'RESULTS.json',dict(cases=cases,ranking=ranking,test_annotations_read=False,
        selected_segmentation_configuration=ranking[0]['name'],
        selected_tile_classification_configuration=load(Path(cycles[1])/'ELIGIBLE_RESULTS.json')['selected_validation_configuration']))
    print(__import__('json').dumps(ranking),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','regions','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cycles',type=Path,nargs='+',required=True);a=p.parse_args()
    cv2.setNumThreads(2);validation(a.root,a.regions,a.cycles,a.output)


if __name__=='__main__':main()
