"""Context and native PLS-LDA compared with matched linear baselines.

All methods use the same train-only capped reservoir and held-out physical
tile targets. Context uses within-image features and coordinates, never labels.
"""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from .registry import load,save,sha
from .units import ranked_labels
from .train import collect_training,prepared,verify_preparation,score_units,export_linear_responses


def context_features(features,boxes,mpp):
    """Concatenate the tile mean and its contrast to a 90 micrometre context."""
    centres=(boxes[:,:2]+boxes[:,2:]/2)*np.asarray(mpp)
    neighbours=cKDTree(centres).query_ball_point(centres,90.,workers=1)
    broad=np.empty_like(features)
    for i,ids in enumerate(neighbours):
        weights=np.exp(-np.sum((centres[ids]-centres[i])**2,1)/(2*45.**2))
        broad[i]=np.average(features[ids],axis=0,weights=weights)
    return np.concatenate([features,features-broad],axis=1)


def feature(root,row,kind):
    folder,_=prepared(root,row)
    x=np.asarray(np.load(folder/'mean.npy'),dtype=np.float64)
    if kind=='context':
        with np.load(folder/'units.npz') as u:x=context_features(x,u['boxes'],row['sample']['working_mpp_xy'])
    return x


def score_predictions(root,registry,folder,classes,records):
    truths=[];ranks=[];cases=[]
    for row in registry['cases']:
        if row['split']!='validation':continue
        src,_=prepared(root,row)
        with np.load(src/'targets.npz') as t:
            keep=t['coverage']>=.5;truth=t['fractions'].argmax(1)+1
        p=folder/(row['case']+'.npz')
        if sha(p)!=next(r['prediction_sha256'] for r in records if r['case']==row['case']):raise ValueError('Prediction changed')
        with np.load(p) as values:rank=values['top5'].copy()
        with np.load(src/'units.npz') as u:rank[u['tissue_fraction']<.02]=0
        truths.append(truth[keep]);ranks.append(rank[keep])
        cases.append(dict(case=row['case'],dataset=row['dataset'],**score_units(truth[keep],rank[keep],classes)))
    return dict(**score_units(np.concatenate(truths),np.concatenate(ranks),classes),cases=cases)


def run(root,output,previous):
    from fastpls import PLS
    root,output,previous=map(Path,(root,output,previous))
    if output.exists():raise ValueError('Preserve cycle output; choose a new version')
    registry=load(root/'REGISTRY.json');verify_preparation(root,registry,load(root/'PREPARATION_COMPLETE.json'))
    previous_results=load(previous/'RESULTS.json')
    output.mkdir(parents=True)
    method=dict(feature='mean',pure=False,target='hard')
    x,y,classes,training=collect_training(root,registry,method)
    # Locate exactly the same reservoir rows in their original per-case order.
    positions={};rows={r['case']:r for r in registry['cases']}
    for unit in training['selected_unit_ids']:positions.setdefault(unit['case'],[]).append(unit['unit'])
    context_x=np.concatenate([feature(root,rows[case],'context')[ids] for case,ids in positions.items()])
    validation=[r for r in registry['cases'] if r['split']=='validation']
    protocol=dict(cycle=2,registry_sha256=sha(root/'REGISTRY.json'),previous_results_sha256=sha(previous/'RESULTS.json'),
        observation='Cycle1 pooling outperformed centre tokens on balanced accuracy; pure-tile filtering lost rare classes.',
        training_sha256=None,test_annotations_read=False,threads=2,
        candidates=['S05 context PLS-DA','S06 mean native PLS-LDA','S07 context native PLS-LDA',
                    'Ridge one-hot regression','Euclidean nearest class centroid'],
        component_grid=[16,24,48],ridge_alphas=[1.,100.,10000.],
        context='Tile mean and residual against a Gaussian 45um-scale feature context truncated at90um. Same-image features only.',
        latency='Five warmed native head calls per ROI; context construction is measured separately. Encoder and disk load excluded.',
        feature_code_sha256=sha(Path(__file__)),created_at_utc=datetime.now(timezone.utc).isoformat())
    save(output/'TRAINING.json',training);protocol['training_sha256']=sha(output/'TRAINING.json');save(output/'PROTOCOL.json',protocol)
    candidates=[]
    for key,kind,classifier in [('S05','context','argmax'),('S06','mean','lda'),('S07','context','lda')]:
        for components in (16,24,48):candidates.append((f'{key}-c{components:02d}',kind,'pls',classifier,components))
    for alpha in (1.,100.,10000.):candidates.append((f'RIDGE-a{alpha:g}','mean','ridge',None,alpha))
    candidates.append(('CENTROID','mean','centroid',None,0))
    fitted=[]
    for name,kind,engine,classifier,parameter in candidates:
        xt=context_x if kind=='context' else x;folder=output/name;folder.mkdir()
        began=time.monotonic()
        if engine=='pls':
            model=PLS(n_components=int(parameter),classifier=classifier,backend='cpu',seed=314159).fit(xt,y)
            predict=lambda a:model.predict(a,top=min(5,len(classes)))
            if classifier=='argmax':
                coefficient,intercept=export_linear_responses(model,xt.shape[1])
                np.savez(folder/'linear_response_model.npz',coefficient=coefficient,intercept=intercept,classes=classes)
        elif engine=='ridge':
            target=(y[:,None]==classes[None]).astype(float);model=Ridge(alpha=parameter).fit(xt,target)
            predict=lambda a:ranked_labels(model.predict(a),classes,min(5,len(classes)))
            np.savez(folder/'linear_response_model.npz',coefficient=model.coef_.T,intercept=model.intercept_,classes=classes)
        else:
            means=np.array([xt[y==g].mean(0) for g in classes])
            predict=lambda a:ranked_labels(2*a@means.T-np.sum(means**2,1),classes,min(5,len(classes)))
            np.savez(folder/'linear_response_model.npz',coefficient=2*means.T,intercept=-np.sum(means**2,1),classes=classes)
        fit_seconds=time.monotonic()-began
        records=[]
        for row in validation:
            began=time.monotonic();xv=feature(root,row,kind);preparation_seconds=time.monotonic()-began
            predict(xv);times=[]
            for _ in range(5):
                began=time.monotonic();ranks=predict(xv);times.append(time.monotonic()-began)
            if ranks.shape!=(len(xv),min(5,len(classes))):raise ValueError('Rank geometry changed')
            np.savez(folder/(row['case']+'.npz'),top5=ranks)
            records.append(dict(case=row['case'],units=len(xv),native_prediction_seconds=float(np.median(times)),
                feature_read_and_context_seconds=preparation_seconds,prediction_sha256=sha(folder/(row['case']+'.npz'))))
        record=dict(name=name,feature=kind,engine=engine,classifier=classifier,parameter=parameter,
            classes=classes.tolist(),fit_seconds=fit_seconds,predictions=records,
            native_core='fastPLS C++ CPU' if engine=='pls' else 'scikit-learn Ridge/OpenBLAS CPU' if engine=='ridge' else 'NumPy/OpenBLAS CPU',
            lda_runtime_rebuild='Native LDA model is reproducible from frozen training rows and this protocol; native object has no public persistent state API.' if classifier=='lda' else None)
        save(folder/'PREDICTIONS.json',record);fitted.append(record)
        print(__import__('json').dumps(dict(frozen=name,fit_seconds=fit_seconds)),flush=True)
    save(output/'ALL_VALIDATION_PREDICTIONS_FROZEN.json',dict(test_annotations_read=False,
        candidates={r['name']:sha(output/r['name']/'PREDICTIONS.json') for r in fitted}))
    results=[]
    for model in fitted:
        scores=score_predictions(root,registry,output/model['name'],model['classes'],model['predictions'])
        scores.update(name=model['name'],feature=model['feature'],engine=model['engine'],fit_seconds=model['fit_seconds'],
            mean_head_seconds=float(np.mean([r['native_prediction_seconds'] for r in model['predictions']])),
            mean_feature_read_and_context_seconds=float(np.mean([r['feature_read_and_context_seconds'] for r in model['predictions']])))
        results.append(scores)
    results.sort(key=lambda r:(-r['balanced_accuracy'],-r['top1'],r['mean_head_seconds']))
    save(output/'RESULTS.json',dict(ranking=results,selected_validation_configuration=results[0]['name'],
        test_annotations_read=False,cycle_complete_for_validation=True,
        previous_best=previous_results['ranking'][0]['name']))
    print(__import__('json').dumps({k:v for k,v in results[0].items() if k!='cases'}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output','previous'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=2):run(a.root,a.output,a.previous)


if __name__=='__main__':main()
