"""Freeze test predictions for preselected tissue heads, then evaluate once.

Prediction does not decode test annotations. The selection file is written
before any test predictions, including the main decision criteria.
"""
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
from threadpoolctl import threadpool_limits
import tifffile
from sklearn.linear_model import Ridge
from .registry import load,save,sha
from .train import collect_training,prepared
from .cycle_two import feature
from .units import ranked_labels
from .eligibility import eligible


def specifications(base):
    return [dict(name=name,feature_root=str(base/root),cycle=str(base/cycle)) for name,root,cycle in
        [('S01-c24','registry-v1','cycle-01'),('S02-c16','registry-v1','cycle-01'),
         ('S06-c24','registry-v1','cycle-02'),('S07-c24','registry-v1','cycle-02'),
         ('RIDGE-a1','registry-v1','cycle-02'),('S08-c48','regions-v1','cycle-03'),('S09-c48','regions-v1','cycle-03')]]


def fit(spec):
    from fastpls import PLS
    root=Path(spec['feature_root']);registry=load(root/'REGISTRY.json')
    frozen=load(Path(spec['cycle'])/spec['name']/'PREDICTIONS.json');name=spec['name']
    kind=frozen.get('feature','centre' if name.startswith('S01') else 'mean')
    target=frozen.get('target','hard');method=dict(feature='centre' if name.startswith('S01') else 'mean',pure=False,target=target)
    x,y,classes,training=collect_training(root,registry,method)
    if kind=='context':
        positions={};rows={r['case']:r for r in registry['cases']}
        for unit in training['selected_unit_ids']:positions.setdefault(unit['case'],[]).append(unit['unit'])
        x=np.concatenate([feature(root,rows[case],'context')[ids] for case,ids in positions.items()])
    began=time.monotonic()
    if name.startswith('RIDGE'):
        model=Ridge(alpha=float(frozen['parameter'])).fit(x,(y[:,None]==classes).astype(float))
        predictor=lambda a:ranked_labels(model.predict(a),classes,min(5,len(classes)))
        config=dict(engine='scikit-learn Ridge CPU',alpha=frozen['parameter'])
    else:
        classifier=frozen.get('classifier','argmax' if target=='hard' else None)
        n=int(frozen.get('components',frozen.get('parameter')))
        model=PLS(n_components=n,classifier=classifier,backend='cpu',seed=314159).fit(x,y)
        predictor=(lambda a:model.predict(a,top=min(5,len(classes)))) if classifier else (
            lambda a:ranked_labels(model.predict_responses(a),classes,min(5,len(classes))))
        config=dict(engine='fastPLS C++ CPU',components=n,classifier=classifier,scaling='centering',seed=314159,method='simpls')
    config.update(feature=kind,target=target,classes=classes.tolist(),refit_seconds=time.monotonic()-began,
                  training_rows=len(x),training_unit_ids_sha256=sha(Path(spec['cycle'])/('TRAINING.json' if name.startswith(('S06','S07','RIDGE')) else name.split('-')[0]+'_TRAINING.json')))
    return predictor,x,y,config,registry


def values(root,row,kind):
    if kind=='context':return feature(root,row,kind)
    folder,_=prepared(root,row)
    return np.asarray(np.load(folder/(kind+'.npy')),dtype=np.float64)


def predict(base,output):
    from .evaluate import native_rank
    base,output=Path(base),Path(output)
    if output.exists():raise ValueError('Never overwrite final test predictions')
    registry=load(base/'registry-v1/REGISTRY.json');models=specifications(base)
    selection=dict(models=models,registry_sha256=sha(base/'registry-v1/REGISTRY.json'),
        validation_native_results_sha256=sha(base/'validation-native-v1/RESULTS.json'),
        selected_segmentation_model='S09-c48',selected_tile_rank_model='S07-c24',
        decision='S09 maximized eligible validation native-pixel mIoU; S07 maximized eligible validation tile balanced accuracy. Compare the remaining frozen formulations as predefined controls; do not reselect from test scores.',
        test_annotations_decoded=False,test_cases=[r['case'] for r in registry['cases'] if r['split']=='test'],
        benchmark_test_cases=[r['case'] for r in registry['cases'] if r['split']=='test' and eligible(r)],
        benchmark_rule='At least2 distinct mapped expert roles, plus existing physical size/calibration/tiling eligibility.',
        source_sha256=sha(Path(__file__)))
    # Stop if the documented choices no longer match the validation evidence.
    if load(base/'validation-native-v1/RESULTS.json')['selected_segmentation_configuration']!='S09-c48':raise ValueError('Selection evidence changed')
    save(output/'TEST_SELECTION.json',selection);frozen_models={}
    for spec in models:
        predictor,x,y,config,_=fit(spec);name=spec['name'];folder=output/name;folder.mkdir()
        # Reproduce previously frozen validation ranks before touching test inputs.
        parity=[]
        for row in [r for r in registry['cases'] if r['split']=='validation'][:2]:
            ranks=predictor(values(spec['feature_root'],row,config['feature']))
            with np.load(Path(spec['cycle'])/name/(row['case']+'.npz')) as prior:
                if not np.array_equal(ranks,prior['top5']):raise ValueError('Refitted native head does not reproduce validation ranks: '+name)
            parity.append(row['case'])
        # A portable training bundle permits deterministic native reconstruction.
        # The public fastPLS API does not serialize its fitted C++ object.
        bundle=base/'models'/name;bundle.mkdir(parents=True,exist_ok=True)
        np.save(bundle/'training_x.npy',x);np.save(bundle/'training_y.npy',y)
        save(bundle/'MODEL.json',dict(name=name,**config,ontology=registry['ontology'],
            feature_contract=dict(encoder='Virchow2',revision=registry['upstream_encoder_revision'],
                columns=1280,patch_px=14,window_px=224,working_mpp_xy=[.5,.5],
                selected_channels_used=False,description='The feature cache contains stitched spatial tokens, not CLS or register tokens.'),
            load_contract='Recreate the native fastPLS CPU model once from the bundled training arrays; prediction uses that native model. Refit time is separate from warm per-ROI latency.',
            training_x_sha256=sha(bundle/'training_x.npy'),training_y_sha256=sha(bundle/'training_y.npy'),
            test_annotations_in_bundle=False,registry_sha256=selection['registry_sha256']))
        del x,y
        records=[]
        for row in registry['cases']:
            if row['split']!='test':continue
            began=time.monotonic();xv=values(spec['feature_root'],row,config['feature']);feature_seconds=time.monotonic()-began
            predictor(xv);times=[]
            for _ in range(5):
                began=time.monotonic();ranks=predictor(xv);times.append(time.monotonic()-began)
            path=folder/(row['case']+'.npz');np.savez(path,top5=ranks)
            mask=native_rank(spec['feature_root'],row,ranks)
            tifffile.imwrite(folder/(row['case']+'.tif'),mask,compression='deflate')
            records.append(dict(case=row['case'],rank_sha256=sha(path),mask_sha256=sha(folder/(row['case']+'.tif')),
                head_seconds=float(np.median(times)),feature_read_and_context_seconds=feature_seconds))
        save(folder/'PREDICTIONS.json',dict(name=name,config=config,records=records,
            validation_rank_reproduction=parity,test_annotations_decoded=False,selection_sha256=sha(output/'TEST_SELECTION.json')))
        frozen_models[name]=sha(folder/'PREDICTIONS.json')
        print(__import__('json').dumps(dict(test_predictions_frozen=name,cases=len(records))),flush=True)
    save(output/'ALL_TEST_PREDICTIONS_FROZEN.json',dict(selection_sha256=sha(output/'TEST_SELECTION.json'),
        models=frozen_models,test_annotations_decoded=False))


def evaluate(base,output):
    from .evaluate import broad_truth,native_rank,pixel_scores,aggregate
    base,output=Path(base),Path(output)
    if (output/'RESULTS.json').exists():raise ValueError('Test already evaluated; preserve results')
    selection=load(output/'TEST_SELECTION.json');frozen=load(output/'ALL_TEST_PREDICTIONS_FROZEN.json')
    if frozen['selection_sha256']!=sha(output/'TEST_SELECTION.json'):raise ValueError('Test selection changed')
    registry=load(base/'registry-v1/REGISTRY.json')
    if sha(base/'registry-v1/REGISTRY.json')!=selection['registry_sha256']:raise ValueError('Registry changed')
    records={}
    for name,h in frozen['models'].items():
        path=output/name/'PREDICTIONS.json'
        if sha(path)!=h:raise ValueError('Test model manifest changed')
        records[name]=load(path)
    cases=[]
    for row in registry['cases']:
        if row['case'] not in selection['benchmark_test_cases']:continue
        truth=broad_truth(row);case=dict(case=row['case'],dataset=row['dataset'],group=row['group'],models={})
        for spec in selection['models']:
            name=spec['name'];record=next(r for r in records[name]['records'] if r['case']==row['case'])
            path=output/name/(row['case']+'.npz');maskpath=output/name/(row['case']+'.tif')
            if sha(path)!=record['rank_sha256'] or sha(maskpath)!=record['mask_sha256']:raise ValueError('Test output changed')
            with np.load(path) as p:ranks=p['top5']
            top=tifffile.imread(maskpath);hit=top==truth
            for j in range(1,ranks.shape[1]):hit|=native_rank(spec['feature_root'],row,ranks,j)==truth
            case['models'][name]=dict(**pixel_scores(truth,top,len(registry['ontology']['roles']),hit),
                head_seconds=record['head_seconds'],feature_read_and_context_seconds=record['feature_read_and_context_seconds'])
        cases.append(case)
    ranking=aggregate(cases,[s['name'] for s in selection['models']])
    for r in ranking:r['mean_native_head_seconds']=float(np.mean([c['models'][r['name']]['head_seconds'] for c in cases]))
    # A group bootstrap respects known linked source groups rather than ROI pseudo-replication.
    groups=sorted({c['group'] for c in cases});rng=np.random.default_rng(314159);index=rng.integers(0,len(groups),(2000,len(groups)))
    comparisons=[]
    for a,b in [('S09-c48','RIDGE-a1'),('S07-c24','S06-c24'),('S09-c48','S02-c16')]:
        delta=np.array([np.mean([c['models'][a]['macro_iou']-c['models'][b]['macro_iou'] for c in cases if c['group']==g]) for g in groups])
        comparisons.append(dict(a=a,b=b,group_mean_iou_delta=float(delta.mean()),
            group_bootstrap95=np.percentile(delta[index].mean(1),[2.5,97.5]).tolist()))
    save(output/'RESULTS.json',dict(cases=cases,ranking=ranking,paired_comparisons=comparisons,
        selection_sha256=sha(output/'TEST_SELECTION.json'),all_test_predictions_frozen_sha256=sha(output/'ALL_TEST_PREDICTIONS_FROZEN.json'),
        test_annotations_decoded_cases=[r['case'] for r in cases],
        fixed_pretest_segmentation_choice=selection['selected_segmentation_model'],fixed_pretest_rank_choice=selection['selected_tile_rank_model'],
        caveat='Held-out groups for the newly trained tissue head. Images were previously inspected during Bora architecture development; not independent validation of the architecture or a patient-independent clinical cohort.'))
    print(__import__('json').dumps(ranking),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['predict','evaluate','all'])
    p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cv2.setNumThreads(2)
    with threadpool_limits(limits=2):
        if a.mode in ('predict','all'):predict(a.base,a.output)
        if a.mode in ('evaluate','all'):evaluate(a.base,a.output)


if __name__=='__main__':main()
