"""Classify frozen connected Bora regions using native CPU heads.

All candidates return one ranked tissue list per region. Tile ranks may be
aggregated inside that region, but no candidate changes its boundaries.
"""
import argparse
import hashlib
import os
from pathlib import Path
import time
import cv2
import numpy as np
from scipy import sparse
import tifffile
from threadpoolctl import threadpool_limits
from experiments.bora_supervised.registry import load,save,sha
from experiments.bora_supervised.eligibility import eligible
from experiments.bora_supervised.units import ranked_labels
from experiments.bora_supervised.train import score_units
from experiments.bora_supervised.evaluate import pixel_scores,broad_truth
from .context import boundary_context
from .targets import target_counts
from .labels import semantic_mask


def region_features(geometry,case,kind='mean',mpp_xy=None):
    folder=Path(geometry)/'cases'/case;x=np.asarray(np.load(folder/'mean.npy'),dtype=np.float64)
    if kind=='mean_std':x=np.concatenate([x,np.asarray(np.load(folder/'std.npy'),dtype=np.float64)],1)
    if kind=='neighbor':x=boundary_context(tifffile.imread(folder/'regions.native.tif'),x,mpp_xy)
    return np.asfortranarray(x)


def target_labels(fractions,coverage,identity=None):
    fractions=np.asarray(fractions);coverage=np.asarray(coverage)
    if identity is not None:
        # Corrected task: identify the deidentified expert-derived group.
        # Its hidden original identity is the answer, not a fragment majority.
        truth=np.asarray(identity)
        return truth,(truth>0)&(fractions.sum(1)>0)
    return fractions.argmax(1)+1,(coverage>=.5)&(fractions.sum(1)>0)


def collect(registry,geometry,targets,kind,target,cap=1000):
    locations={};eligible_count={}
    for row in registry['cases']:
        if row['split']!='train':continue
        with np.load(Path(targets)/'cases'/row['case']/'targets.npz') as t:
            f,c=t['fractions'],t['coverage'];identity=t['identity'] if 'identity' in t else None
        dominant,keep=target_labels(f,c,identity)
        for i in np.flatnonzero(keep):
            g=int(dominant[i]);eligible_count[g]=eligible_count.get(g,0)+1
            key=hashlib.sha256(f"bora-region-training-v1:{row['case']}:{i+1}".encode()).hexdigest()
            locations.setdefault(g,[]).append((key,row['case'],int(i)))
    selected=sorted((case,i,g) for g,items in locations.items() for _,case,i in sorted(items)[:cap])
    if len(locations)<2:raise ValueError('Insufficient labelled tissue roles')
    grouped={}
    for case,i,g in selected:grouped.setdefault(case,[]).append((i,g))
    xs=[];ys=[];fractions=[]
    for case,items in grouped.items():
        row=next(r for r in registry['cases'] if r['case']==case)
        ids=[i for i,g in items];xs.append(region_features(geometry,case,kind,row['sample']['native_mpp_xy'])[ids]);ys.extend(g for i,g in items)
        with np.load(Path(targets)/'cases'/case/'targets.npz') as t:fractions.append(t['fractions'][ids])
    x=np.asfortranarray(np.concatenate(xs));fractions=np.concatenate(fractions).astype(np.float64);y=np.asarray(ys)
    classes=np.flatnonzero(fractions.sum(0)>0)+1 if target=='fractions' else np.unique(y)
    responses=fractions[:,classes-1] if target=='fractions' else y
    return x,responses,classes,dict(training_regions=len(y),eligible_by_dominant_role=eligible_count,
        selected=[dict(case=case,region_id=i+1,dominant_role=g) for case,i,g in selected],
        coverage_threshold=None if identity is not None else .5,
        target_basis='Hidden original expert group identity' if identity is not None else 'Dominant expert role in connected fragment',
        per_dominant_role_cap=cap,all_regions_from_training_groups=True)


def tile_voter(bundle):
    from fastpls import PLS
    bundle=Path(bundle);record=load(bundle/'MODEL.json')
    for n in ('training_x','training_y'):
        if sha(bundle/(n+'.npy'))!=record[n+'_sha256']:raise ValueError('Tile baseline bundle changed')
    x=np.load(bundle/'training_x.npy');y=np.load(bundle/'training_y.npy');classes=np.asarray(record['classes'])
    model=PLS(n_components=record['components'],classifier=record['classifier'],backend='cpu',seed=record['seed']).fit(x,y)
    def predict(tile_features,weights):
        ranks=model.predict(np.asarray(tile_features,dtype=np.float64),top=len(classes))
        scores=np.empty((len(ranks),len(classes)),float)
        for j,g in enumerate(classes):scores[:,j]=(len(classes)-np.argmax(ranks==g,axis=1))/len(classes)
        region_scores=(weights@scores)/np.asarray(weights.sum(1))
        return ranked_labels(region_scores,classes,min(5,len(classes)))
    return predict,classes


def predict(root,registry_root,geometry,targets,tile_bundle,tile_registry,split='validation'):
    from fastpls import PLS
    root,registry_root,geometry,targets,tile_bundle,tile_registry=map(lambda p:Path(p).resolve(),(root,registry_root,geometry,targets,tile_bundle,tile_registry))
    if root.exists():raise ValueError('Preserve region classifier cycle')
    registry=load(registry_root/'REGISTRY.json');frozen=load(geometry/'ALL_GEOMETRY_FROZEN.json');target_frozen=load(targets/'TARGETS_FROZEN.json')
    if target_frozen['test_annotations_decoded']:raise ValueError('Test targets must remain unprepared')
    for case,record in target_frozen['cases'].items():
        if sha(targets/'cases'/case/'targets.npz')!=record['sha256']:raise ValueError('Frozen target shard changed')
    if set(frozen['cases'])!={r['case'] for r in registry['cases']}:raise ValueError('Missing frozen geometry')
    for case,h in frozen['cases'].items():
        folder=geometry/'cases'/case
        if sha(folder/'GEOMETRY_FROZEN.json')!=h:raise ValueError('Geometry changed')
        record=load(folder/'GEOMETRY_FROZEN.json')
        for name in ['mean.npy','std.npy','tile_overlap.npz','regions.native.tif']:
            if sha(folder/name)!=record['files_sha256'][name]:raise ValueError('Region geometry/evidence changed')
    rows=[r for r in registry['cases'] if r['split']==split]
    unit=frozen.get('prediction_unit','Connected Bora region')
    save(root/'PROTOCOL.json',dict(prediction_unit=unit,
        source_registry_sha256=sha(registry_root/'REGISTRY.json'),geometry_sha256=sha(geometry/'ALL_GEOMETRY_FROZEN.json'),
        targets_sha256=sha(targets/'TARGETS_FROZEN.json'),split=split,
        candidates=['R01 area-weighted tile-rank voting inside each Bora region','R02 pooled-region native PLS-LDA',
                    'R03 pooled region mean and dispersion native PLS-LDA','R04 region-composition native PLS regression',
                    'R05 region and shared-boundary context native PLS-LDA'],
        component_grid=[16,24,48],native_backend='fastPLS C++ CPU',threads=2,
        region_identifiers_preserved=True,source_code_sha256=sha(Path(__file__)),
        test_reuse_disclosed=True,semantic_predictions_do_not_change_geometry=True))
    candidates=[]
    for key,kind,target,classifier in [('R02','mean','hard','lda'),('R03','mean_std','hard','lda'),
                                       ('R04','mean','fractions',None),('R05','neighbor','hard','lda')]:
        x,y,classes,report=collect(registry,geometry,targets,kind,target);save(root/(key+'_TRAINING.json'),report)
        np.save(root/(key+'-training_x.npy'),x);np.save(root/(key+'-training_y.npy'),y)
        for n in (16,24,48):
            began=time.monotonic();model=PLS(n_components=n,classifier=classifier,scaling='autoscaling',backend='cpu',seed=314159).fit(x,y);fit_seconds=time.monotonic()-began
            name=f'{key}-c{n:02d}';folder=root/name;folder.mkdir()
            for column in ('x','y'):os.link(root/(key+'-training_'+column+'.npy'),folder/('training_'+column+'.npy'))
            save(folder/'MODEL.json',dict(name=name,engine='fastPLS C++ CPU',components=n,
                classifier=classifier,scaling='autoscaling',method='simpls',seed=314159,
                feature=kind,target=target,classes=classes.tolist(),ontology=registry['ontology'],
                training_x_sha256=sha(folder/'training_x.npy'),training_y_sha256=sha(folder/'training_y.npy'),
                training_unit=unit,geometry_sha256=sha(geometry/'ALL_GEOMETRY_FROZEN.json'),
                registry_sha256=sha(registry_root/'REGISTRY.json')))
            infer=(lambda a:model.predict(a,top=min(5,len(classes)))) if classifier else (
                lambda a:ranked_labels(model.predict_responses(a),classes,min(5,len(classes))))
            records=[]
            for row in rows:
                began=time.monotonic();xv=region_features(geometry,row['case'],kind,row['sample']['native_mpp_xy']);feature_seconds=time.monotonic()-began
                infer(xv);times=[]
                for _ in range(5):
                    began=time.monotonic();ranks=infer(xv);times.append(time.monotonic()-began)
                path=folder/(row['case']+'.npz');np.savez(path,top5=ranks)
                if not np.isin(ranks,classes).all():raise ValueError('Native head returned an unregistered tissue role')
                records.append(dict(case=row['case'],regions=len(xv),head_seconds=float(np.median(times)),
                                    feature_loading_and_context_seconds=feature_seconds,sha256=sha(path)))
            record=dict(name=name,family=key,feature=kind,target=target,classifier=classifier,scaling='autoscaling',components=n,
                classes=classes.tolist(),fit_seconds=fit_seconds,predictions=records,unit='Bora region')
            save(folder/'PREDICTIONS.json',record);candidates.append(record)
            print(__import__('json').dumps(dict(frozen=name,regions=sum(r['regions'] for r in records))),flush=True)
        del x,y
    infer,classes=tile_voter(tile_bundle);name='R01-tile-vote';folder=root/name;folder.mkdir();records=[]
    for row in rows:
        tiles=np.load(tile_registry/'cases'/row['case']/'mean.npy');weights=sparse.load_npz(geometry/'cases'/row['case']/'tile_overlap.npz')
        infer(tiles,weights);times=[]
        for _ in range(5):
            began=time.monotonic();ranks=infer(tiles,weights);times.append(time.monotonic()-began)
        path=folder/(row['case']+'.npz');np.savez(path,top5=ranks)
        records.append(dict(case=row['case'],regions=len(ranks),head_seconds=float(np.median(times)),sha256=sha(path)))
    record=dict(name=name,family='R01',classes=classes.tolist(),feature='Tile rank voting after Bora; not tile output',predictions=records,
        vote='Native PLS-LDA full ranking converted to Borda scores and averaged by exact overlap area of every tile with each refined region')
    save(folder/'PREDICTIONS.json',record);candidates.append(record)
    save(root/'ALL_PREDICTIONS_FROZEN.json',dict(geometry_unchanged=True,split=split,
        models={r['name']:sha(root/r['name']/'PREDICTIONS.json') for r in candidates},fine_evaluation_targets_decoded_by_predictor=False))


def evaluate(root,registry_root,geometry,targets):
    root,registry_root,geometry,targets=map(Path,(root,registry_root,geometry,targets))
    frozen=load(root/'ALL_PREDICTIONS_FROZEN.json');registry=load(registry_root/'REGISTRY.json')
    models=[load(root/name/'PREDICTIONS.json') for name in frozen['models']]
    for model in models:
        if sha(root/model['name']/'PREDICTIONS.json')!=frozen['models'][model['name']]:raise ValueError('Prediction manifest changed')
    split=frozen['split']
    if split not in ('validation','test'):raise ValueError('Only reserved evaluation groups may be scored')
    geometry_manifest=load(geometry/'ALL_GEOMETRY_FROZEN.json')
    whole_groups=geometry_manifest.get('coarse_reference_derived',False)
    rows=[r for r in registry['cases'] if r['split']==split and eligible(r)]
    results=[]
    for model in models:
        truths=[];rankings=[];cases=[]
        for row in rows:
            case=row['case'];path=root/model['name']/(case+'.npz')
            expected=next(r['sha256'] for r in model['predictions'] if r['case']==case)
            if sha(path)!=expected:raise ValueError('Region prediction changed')
            with np.load(path) as p:ranks=p['top5']
            regions=tifffile.imread(geometry/'cases'/case/'regions.native.tif')
            pixel_truth=broad_truth(row)
            if split=='validation':
                with np.load(targets/'cases'/case/'targets.npz') as t:
                    f,c=t['fractions'],t['coverage'];identity=t['identity'] if 'identity' in t else None
            else:
                f,c=target_counts(regions,pixel_truth,len(registry['ontology']['roles']));identity=None
                if whole_groups:
                    from experiments.bora_group.targets import identity_labels
                    identity=identity_labels(row,geometry)
            truth,keep=target_labels(f,c,identity)
            truths.append(truth[keep]);rankings.append(ranks[keep]);region_score=score_units(truth[keep],ranks[keep],model['classes'])
            mask=semantic_mask(regions,ranks);at=regions>0
            hit=np.zeros(regions.shape,bool)
            for j in range(ranks.shape[1]):hit[at]|=ranks[:,j][regions[at]-1]==pixel_truth[at]
            metric=pixel_scores(pixel_truth,mask,len(registry['ontology']['roles']),hit)
            width=len(registry['ontology']['roles'])+1
            hist=np.bincount(regions.ravel().astype(np.int64)*width+pixel_truth.ravel(),
                minlength=(int(regions.max())+1)*width).reshape(-1,width)[1:,1:]
            oracle=float(hist.max(1).sum()/max(np.count_nonzero(pixel_truth),1))
            tifffile.imwrite(root/model['name']/(case+'.tif'),mask,compression='deflate')
            cases.append(dict(case=case,dataset=row['dataset'],region_score=region_score,pixel_score=metric,
                evaluated_regions=int(keep.sum()),all_predicted_regions=len(ranks),
                mean_annotated_purity=float(f[keep].max(1).mean()) if keep.any() else None,
                oracle_majority_region_pixel_accuracy=oracle))
        summary=score_units(np.concatenate(truths),np.concatenate(rankings),model['classes'])
        results.append(dict(name=model['name'],family=model['family'],**summary,cases=cases,
            mean_native_pixel_iou=float(np.mean([r['pixel_score']['macro_iou'] for r in cases if r['pixel_score']['macro_iou'] is not None])),
            mean_head_seconds=float(np.mean([r['head_seconds'] for r in model['predictions'] if r['case'] in {v['case'] for v in rows}]))))
    results.sort(key=lambda r:(-r['balanced_accuracy'],-r['top1']))
    selected=results[0]['name'] if split=='validation' else load(root/'PROTOCOL.json')['selected_on_validation'][0]
    save(root/'RESULTS.json',dict(unit=geometry_manifest.get('prediction_unit','Connected Bora tissue segment'),ranking=results,split=split,
        selected_segment_classifier=selected,regions_reclassified_without_boundary_changes=True,
        selected_on_this_split=split=='validation',test_annotations_decoded=split=='test',
        test_reuse='Earlier tile/superpixel test has been inspected; exploratory follow-up, not an untouched final test.',
        role_semantics_require_pathology_review=True))
    print(__import__('json').dumps({k:v for k,v in results[0].items() if k not in ('cases','recall_by_role')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['predict','evaluate','all'])
    for name in ('root','registry-root','geometry','targets'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tile-bundle',type=Path);p.add_argument('--tile-registry',type=Path);a=p.parse_args();cv2.setNumThreads(2)
    with threadpool_limits(limits=2):
        if a.mode in ('predict','all'):predict(a.root,a.registry_root,a.geometry,a.targets,a.tile_bundle,a.tile_registry)
        if a.mode in ('evaluate','all'):evaluate(a.root,a.registry_root,a.geometry,a.targets)


if __name__=='__main__':main()
