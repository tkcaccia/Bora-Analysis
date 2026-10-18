"""First supervised cycle: four distinct fastPLS tissue formulations.

Only training targets enter fits. Every validation prediction is frozen before
validation targets are read. Test targets and reference-derived RGB/coarse
masks are not used. Component counts are hyperparameters, not new algorithms.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
from importlib import metadata
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info

from .registry import load,save,sha
from .units import training_rows,ranked_labels

METHODS={
    'S01':dict(name='Centre token PLS discriminant',feature='centre',pure=False,target='hard',
        hypothesis='A single unselected spatial token preserves a focused local appearance with its original encoder context.'),
    'S02':dict(name='Tile pooled PLS discriminant',feature='mean',pure=False,target='hard',
        hypothesis='Area-weighted averaging reduces unstable individual-token variation across a physical 45 micrometre tissue unit.'),
    'S03':dict(name='Pure tissue PLS discriminant',feature='mean',pure=True,target='hard',
        hypothesis='Train on tiles with at least 90 percent annotation coverage and 95 percent purity to reduce boundary label noise; predict every tile.'),
    'S04':dict(name='Tissue composition PLS regression',feature='mean',pure=False,target='fractions',
        hypothesis='Regress annotated tissue fractions instead of forcing every mixed training tile to a single majority label.')}


def prepared(root,row):
    folder=Path(root)/'cases'/row['case'];record=load(folder/'PREPARED.json')
    if record['registry_sha256']!=sha(Path(root)/'REGISTRY.json'):raise ValueError('Registry changed')
    return folder,record


def verify_preparation(root,registry,complete):
    """Hash frozen shards once, without decoding held-out target arrays."""
    root=Path(root)
    if complete['registry_sha256']!=sha(root/'REGISTRY.json') or complete['test_annotations_decoded']:
        raise ValueError('Incomplete or test-contaminated preparation')
    expected=complete['case_preparation_sha256']
    if set(expected)!={r['case'] for r in registry['cases']}:
        raise ValueError('Preparation case coverage changed')
    for row in registry['cases']:
        folder,record=prepared(root,row)
        if sha(folder/'PREPARED.json')!=expected[row['case']]:
            raise ValueError('Preparation record changed')
        annotated=row['split']!='test'
        names={'mean.npy','centre.npy','units.npz'}|({'targets.npz'} if annotated else set())
        if set(record['files_sha256'])!=names or record['reference_decoded']!=annotated:
            raise ValueError('Target isolation contract changed')
        if not annotated and (folder/'targets.npz').exists():
            raise ValueError('Test targets must remain unprepared')
        for name,h in record['files_sha256'].items():
            if sha(folder/name)!=h:raise ValueError('Frozen shard changed: '+row['case']+'/'+name)


def export_linear_responses(model,n_features):
    """Recover a public-API affine response map; validate on unseen X separately."""
    probe=np.vstack([np.zeros((1,n_features)),np.eye(n_features)])
    responses=np.asarray(model.predict_responses(probe))
    if responses.ndim!=2 or responses.shape[0]!=n_features+1:
        raise ValueError('Invalid linear response probes')
    intercept=responses[0]
    return responses[1:]-intercept,intercept


def collect_training(root,registry,method,cap=1000):
    """Stable per-class reservoir; no validation or test target is consulted."""
    locations={};counts=Counter()
    for row in registry['cases']:
        if row['split']!='train':continue
        folder,_=prepared(root,row)
        with np.load(folder/'targets.npz') as labels,np.load(folder/'units.npz') as units:
            f,c=labels['fractions'],labels['coverage'];tf=units['tissue_fraction']
            keep=training_rows(f,c,tf,pure=method['pure'])
            y=np.argmax(f,1)+1
            for i in np.flatnonzero(keep):
                role=int(y[i]);counts[role]+=1
                key=hashlib.sha256(f"bora-train-v1:{row['case']}:{i}".encode()).hexdigest()
                locations.setdefault(role,[]).append((key,row['case'],int(i)))
    selected=sorted((case,i,role) for role,values in locations.items()
                    for _,case,i in sorted(values)[:cap])
    if len(locations)<2 or not selected:raise ValueError('Insufficient training classes')
    by_case={}
    for case,i,role in selected:by_case.setdefault(case,[]).append((i,role))
    xs=[];ys=[];fractions=[]
    for case,indices in by_case.items():
        ids=np.array([i for i,_ in indices]);folder=Path(root)/'cases'/case
        x=np.load(folder/(method['feature']+'.npy'),mmap_mode='r')
        with np.load(folder/'targets.npz') as t:frac=t['fractions'][ids]
        xs.append(np.asarray(x[ids],dtype=np.float64));ys.extend(role for _,role in indices);fractions.append(frac)
    x=np.asfortranarray(np.concatenate(xs));y=np.asarray(ys,dtype=np.int64)
    fractions=np.concatenate(fractions).astype(np.float64)
    # Include every class with training mass, including minor fractions in S04.
    classes=np.flatnonzero(fractions.sum(0)>0)+1 if method['target']=='fractions' else np.unique(y)
    response=fractions[:,classes-1] if method['target']=='fractions' else y
    return x,response,classes,dict(eligible_units=dict(counts),selected_units=len(selected),
        per_majority_class_cap=cap,selected_unit_ids=[dict(case=c,unit=i,majority_role=g) for c,i,g in selected],
        training_case_count=len(by_case),selection='Class-balanced capped stable hash reservoir; all selected units are from train groups only.')


def score_units(truth,rankings,classes):
    truth=np.asarray(truth);rankings=np.asarray(rankings)
    if len(truth)!=len(rankings):raise ValueError('Mismatched score rows')
    top=rankings[:,0];roles=np.unique(truth)
    recall={str(int(g)):float(np.mean(top[truth==g]==g)) for g in roles}
    return dict(units=len(truth),top1=float(np.mean(top==truth)) if len(truth) else None,
        top5=float(np.mean(np.any(rankings[:,:5]==truth[:,None],1))) if len(truth) else None,
        balanced_accuracy=float(np.mean(list(recall.values()))) if recall else None,
        recall_by_role=recall,validation_truth_roles=roles.tolist(),trained_role_count=len(classes),
        top5_is_trivial=len(classes)<=5,
        truth_roles_absent_from_training=sorted(set(map(int,roles))-set(map(int,classes))))


def run(root,output,dependency,components=(4,8,16,24),cap=1000):
    from fastpls import PLS,_core
    root,output,dependency=Path(root),Path(output),Path(dependency)
    if (output/'PROTOCOL.json').exists():raise ValueError('Preserve existing cycle; use a new output version')
    registry=load(root/'REGISTRY.json');complete=load(root/'PREPARATION_COMPLETE.json')
    verify_preparation(root,registry,complete)
    output.mkdir(parents=True,exist_ok=True)
    validation=[r for r in registry['cases'] if r['split']=='validation']
    protocol=dict(cycle=1,task='Supervised tissue-role classification, not another refinement cycle',
        registry_sha256=sha(root/'REGISTRY.json'),preparation_sha256=sha(root/'PREPARATION_COMPLETE.json'),
        methods=METHODS,components=list(components),training_cap_per_majority_role=cap,
        validation_cases=[r['case'] for r in validation],test_annotations_read=False,
        test_predictions_requested=False,selection='Highest validation balanced accuracy; tie-break by top1 then mean native prediction seconds. Test remains unopened for later cycles.',
        scores='Top1/top5 are ranked model responses, not calibrated probabilities. All27 tissue-role IDs retain fixed meanings.',
        precision='float64 predictors and CPU numerical core',threads=2,
        fastpls_version=metadata.version('fastPLS'),upstream_core=load(dependency/'UPSTREAM_CORE.json'),
        dependency_source_sha256={str(p.relative_to(dependency)):sha(p) for p in
            [dependency/'python/fastpls/api.py',dependency/'src/bindings.cpp',dependency/'UPSTREAM_CORE.json']},
        python_source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        native_backend=_core.backend_info(),threadpools=threadpool_info(),
        created_at_utc=datetime.now(timezone.utc).isoformat())
    save(output/'PROTOCOL.json',protocol)
    fitted=[]
    for key,method in METHODS.items():
        x,y,classes,training=collect_training(root,registry,method,cap)
        save(output/(key+'_TRAINING.json'),training)
        for n in components:
            if n>=min(x.shape):continue
            began=time.monotonic()
            model=PLS(n_components=n,method='simpls',classifier='argmax' if method['target']=='hard' else None,
                scaling='centering',backend='cpu',seed=314159).fit(x,y)
            fit_seconds=time.monotonic()-began
            if method['target']=='hard':
                if not np.array_equal(model.classes_,classes):raise ValueError('Class order differs')
            name=f'{key}-c{n:02d}';folder=output/name;folder.mkdir()
            # Linear response export is an evaluated affine equivalent, not a
            # claim that the upstream Python API serializes native objects.
            coefficient,intercept=export_linear_responses(model,x.shape[1])
            np.savez(folder/'linear_response_model.npz',coefficient=coefficient,intercept=intercept,classes=classes)
            records=[];maximum_export_error=0.;export_rank_disagreements=0
            for row in validation:
                src,_=prepared(root,row)
                xv=np.asarray(np.load(src/(method['feature']+'.npy')),dtype=np.float64)
                times=[]
                model.predict_responses(xv)  # Warm-up is excluded from measured head latency.
                for repeat in range(5):
                    began=time.monotonic();values=np.asarray(model.predict_responses(xv));times.append(time.monotonic()-began)
                if values.shape!=(len(xv),len(classes)) or not np.isfinite(values).all():raise ValueError('Invalid response map')
                ranks=ranked_labels(values,classes,min(5,len(classes)))
                exported=xv@coefficient+intercept
                error=float(np.max(np.abs(exported-values)));maximum_export_error=max(maximum_export_error,error)
                export_rank_disagreements+=int(np.count_nonzero(np.any(ranked_labels(exported,classes,min(5,len(classes)))!=ranks,1)))
                np.savez(folder/(row['case']+'.npz'),scores=values.astype(np.float32),top5=ranks)
                records.append(dict(case=row['case'],units=len(xv),native_prediction_seconds=float(np.median(times)),
                    prediction_sha256=sha(folder/(row['case']+'.npz'))))
            record=dict(name=name,method=key,components=n,actual_components=model.n_components_,classes=classes.tolist(),
                fit_seconds=fit_seconds,predictions=records,backend='fastPLS C++ portable CPU',
                linear_export_maximum_absolute_error=maximum_export_error,
                linear_export_top5_disagreement_units=export_rank_disagreements,
                linear_export_qualified=maximum_export_error<1e-8 and export_rank_disagreements==0,
                linear_export_runtime_not_native_head_runtime=True,
                sources=dict(training_sha256=sha(output/(key+'_TRAINING.json')),protocol_sha256=sha(output/'PROTOCOL.json')))
            save(folder/'PREDICTIONS.json',record);fitted.append(record)
            print(__import__('json').dumps(dict(frozen=name,fit_seconds=fit_seconds,validation_cases=len(records),export_error=maximum_export_error)),flush=True)
        del x,y
    save(output/'ALL_VALIDATION_PREDICTIONS_FROZEN.json',dict(protocol_sha256=sha(output/'PROTOCOL.json'),
        validation_target_arrays_decoded_by_fitter=False,test_targets_read=False,
        candidates={r['name']:sha(output/r['name']/'PREDICTIONS.json') for r in fitted}))
    # Target access begins only here; the fitter above cannot see validation labels.
    results=[]
    for model in fitted:
        truths=[];rankings=[];per_case=[]
        for row in validation:
            src,_=prepared(root,row)
            with np.load(src/'targets.npz') as target:
                fractions,coverage=target['fractions'],target['coverage']
                keep=coverage>=.5;truth=np.argmax(fractions,1)+1
            p=output/model['name']/(row['case']+'.npz')
            expected=next(r['prediction_sha256'] for r in model['predictions'] if r['case']==row['case'])
            if sha(p)!=expected:raise ValueError('Prediction changed before scoring')
            with np.load(p) as values:ranks=values['top5'].copy()
            with np.load(src/'units.npz') as units:ranks[units['tissue_fraction']<.02]=0
            truths.append(truth[keep]);rankings.append(ranks[keep])
            per_case.append(dict(case=row['case'],dataset=row['dataset'],**score_units(truth[keep],ranks[keep],model['classes'])))
        scores=score_units(np.concatenate(truths),np.concatenate(rankings),model['classes'])
        scores.update(name=model['name'],method=model['method'],components=model['components'],fit_seconds=model['fit_seconds'],
            mean_native_prediction_seconds=float(np.mean([r['native_prediction_seconds'] for r in model['predictions']])),
            cases=per_case,linear_export_qualified=model['linear_export_qualified'])
        results.append(scores)
    results.sort(key=lambda r:(-r['balanced_accuracy'],-r['top1'],r['mean_native_prediction_seconds']))
    save(output/'RESULTS.json',dict(protocol_sha256=sha(output/'PROTOCOL.json'),ranking=results,
        selected_validation_configuration=results[0]['name'],test_annotations_read=False,
        cycle_complete_for_validation=True,final_model_or_manuscript_complete=False))
    print(__import__('json').dumps(dict(selected=results[0]['name'],top1=results[0]['top1'],top5=results[0]['top5'],
        balanced_accuracy=results[0]['balanced_accuracy'],test_annotations_read=False)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output','dependency'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=2):run(a.root,a.output,a.dependency)


if __name__=='__main__':main()
