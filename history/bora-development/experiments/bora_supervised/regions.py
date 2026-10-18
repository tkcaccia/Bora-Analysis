"""Train and evaluate tissue heads on H&E superpixels, without reference input.

SLIC geometry is determined from H&E at approximately 2um per pixel. Frozen
full Virchow2 tokens are area-pooled onto those regions. Fine masks supply
training/validation targets only; test masks remain unopened.
"""
import argparse
from collections import Counter
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
from scipy import sparse
from skimage.segmentation import slic
import tifffile
from threadpoolctl import threadpool_limits
from .registry import load,save,sha
from .train import collect_training,prepared,export_linear_responses
from .units import ranked_labels
from .cycle_two import score_predictions


def image_regions(image,mpp):
    h,w=image.shape[:2]
    shape=(max(2,round(h*mpp[1]/2)),max(2,round(w*mpp[0]/2)))
    small=cv2.resize(image,shape[::-1],interpolation=cv2.INTER_AREA)
    requested=max(2,round(h*w*np.prod(mpp)/45.**2))
    labels=slic(small,n_segments=requested,compactness=10,sigma=.5,start_label=0,
        enforce_connectivity=True,min_size_factor=.5,max_size_factor=3,channel_axis=-1).astype(np.int32)
    _,labels=np.unique(labels,return_inverse=True)
    return labels.reshape(shape).astype(np.int32),requested


def pool_regions(tokens,regions,working_shape,patch=14):
    """Area quadrature on a calibrated analysis grid, including partial tokens."""
    yy=(np.arange(regions.shape[0])+.5)*working_shape[0]/regions.shape[0]
    xx=(np.arange(regions.shape[1])+.5)*working_shape[1]/regions.shape[1]
    token=(np.minimum((yy/patch).astype(int),tokens.shape[0]-1)[:,None]*tokens.shape[1]
           +np.minimum((xx/patch).astype(int),tokens.shape[1]-1)[None,:])
    weights=sparse.coo_matrix((np.ones(regions.size),(regions.ravel(),token.ravel())),
                              shape=(int(regions.max())+1,tokens.shape[0]*tokens.shape[1])).tocsr()
    total=np.asarray(weights.sum(1)).ravel()
    if np.any(total==0):raise ValueError('Noncontiguous superpixel IDs')
    return np.asarray(weights@tokens.reshape(-1,tokens.shape[-1]),dtype=np.float32)/total[:,None]


def region_targets(reference,regions,mapping,n_roles):
    """Count original native annotation pixels inside projected SLIC regions."""
    labels=cv2.resize(regions,reference.shape[::-1],interpolation=cv2.INTER_NEAREST).ravel()
    roles=np.zeros(reference.shape,np.uint8)
    for raw,role in mapping.items():roles[reference==int(raw)]=int(role)
    n=int(regions.max())+1;total=np.bincount(labels,minlength=n)
    counts=np.bincount(labels*(n_roles+1)+roles.ravel(),minlength=n*(n_roles+1)).reshape(n,n_roles+1)[:,1:]
    annotated=counts.sum(1)
    fractions=counts/np.maximum(annotated[:,None],1)
    return fractions.astype(np.float32),(annotated/np.maximum(total,1)).astype(np.float32)


def prepare(root,output):
    root,output=Path(root),Path(output);registry=load(root/'REGISTRY.json')
    registry.update(parent_registry_sha256=sha(root/'REGISTRY.json'),
        representation='H&E SLIC superpixels at2um, requested mean region area45um squared; all1280 tokens area-pooled',
        geometry_code_sha256=sha(Path(__file__)))
    if (output/'REGISTRY.json').exists():
        if load(output/'REGISTRY.json')!=registry:raise ValueError('Region registry changed')
    else:save(output/'REGISTRY.json',registry)
    records=[]
    for row in registry['cases']:
        folder=output/'cases'/row['case'];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'PREPARED.json').exists():
            record=load(folder/'PREPARED.json')
            if record['registry_sha256']!=sha(output/'REGISTRY.json'):raise ValueError('Registry changed')
            for name,h in record['files_sha256'].items():
                if sha(folder/name)!=h:raise ValueError('Region shard changed')
            records.append(record);continue
        began=time.monotonic();sample=row['sample'];h,w=sample['working_shape_yx'];mpp=sample['working_mpp_xy']
        if sha(row['original_path'])!=row['original_sha256']:raise ValueError('Image changed')
        image=np.asarray(Image.open(row['original_path']).convert('RGB'))
        regions,requested=image_regions(image,mpp);shape=regions.shape
        if sha(row['feature']['path'])!=row['feature']['sha256']:raise ValueError('Full feature tensor changed')
        tokens=np.load(row['feature']['path'],mmap_mode='r')
        mean=pool_regions(tokens,regions,(h,w))
        tissue=tifffile.imread(row['tissue_support_path'])>0
        support=cv2.resize(tissue.astype(np.float32),shape[::-1],interpolation=cv2.INTER_AREA)
        mass=np.bincount(regions.ravel());tf=np.bincount(regions.ravel(),weights=support.ravel())/mass
        np.save(folder/'mean.npy',mean);np.save(folder/'centre.npy',mean)
        np.savez_compressed(folder/'units.npz',region_map=regions,tissue_fraction=tf.astype(np.float32))
        annotated=row['split']!='test'
        if annotated:
            if sha(row['reference_path'])!=row['reference_sha256']:raise ValueError('Reference changed')
            f,c=region_targets(tifffile.imread(row['reference_path']),regions,row['ontology_raw_to_role'],len(registry['ontology']['roles']))
            np.savez(folder/'targets.npz',fractions=f,coverage=c)
        names=['mean.npy','centre.npy','units.npz']+(['targets.npz'] if annotated else [])
        record=dict(case=row['case'],split=row['split'],units=len(mean),reference_decoded=annotated,
            registry_sha256=sha(output/'REGISTRY.json'),files_sha256={n:sha(folder/n) for n in names},
            analysis_shape_yx=list(shape),analysis_mpp_xy=[w*mpp[0]/shape[1],h*mpp[1]/shape[0]],
            requested_regions=requested,seconds=time.monotonic()-began,
            fine_reference_used_for_geometry_or_features=False)
        save(folder/'PREPARED.json',record);records.append(record)
        print(__import__('json').dumps(dict(prepared=row['case'],regions=len(mean),seconds=record['seconds'])),flush=True)
    save(output/'PREPARATION_COMPLETE.json',dict(cases=len(records),test_annotations_decoded=False,
        registry_sha256=sha(output/'REGISTRY.json'),case_preparation_sha256={r['case']:sha(output/'cases'/r['case']/'PREPARED.json') for r in records},
        split_counts=dict(Counter(r['split'] for r in records))))


def train(root,output):
    from fastpls import PLS
    from .train import verify_preparation
    root,output=Path(root),Path(output)
    if output.exists():raise ValueError('Preserve cycle output')
    registry=load(root/'REGISTRY.json');verify_preparation(root,registry,load(root/'PREPARATION_COMPLETE.json'))
    output.mkdir(parents=True)
    save(output/'PROTOCOL.json',dict(cycle=3,registry_sha256=sha(root/'REGISTRY.json'),components=[24,48],threads=2,
        formulations=['S08 superpixel PLS-DA','S09 superpixel tissue composition PLS'],
        comparison='These are region-unit metrics; use common native-pixel evaluation to compare square tiles and superpixels.',
        selection='Choose configuration within each formulation by validation balanced accuracy; final segmentation comparison uses native-pixel mIoU.',
        test_annotations_read=False,source_sha256=sha(Path(__file__))))
    frozen=[]
    for name,target in [('S08','hard'),('S09','fractions')]:
        x,y,classes,training=collect_training(root,registry,dict(feature='mean',pure=False,target=target))
        save(output/(name+'_TRAINING.json'),training)
        for n in (24,48):
            key=f'{name}-c{n:02d}';folder=output/key;folder.mkdir()
            began=time.monotonic()
            model=PLS(n_components=n,classifier='argmax' if target=='hard' else None,backend='cpu',seed=314159).fit(x,y)
            fit_seconds=time.monotonic()-began;coefficient,intercept=export_linear_responses(model,x.shape[1])
            np.savez(folder/'linear_response_model.npz',coefficient=coefficient,intercept=intercept,classes=classes)
            records=[];error=0
            for row in registry['cases']:
                if row['split']!='validation':continue
                src,_=prepared(root,row);xv=np.asarray(np.load(src/'mean.npy'),dtype=np.float64)
                model.predict_responses(xv);times=[]
                for _ in range(5):
                    began=time.monotonic();values=model.predict_responses(xv);times.append(time.monotonic()-began)
                error=max(error,float(np.max(np.abs(values-(xv@coefficient+intercept)))))
                ranks=ranked_labels(values,classes,min(5,len(classes)))
                np.savez(folder/(row['case']+'.npz'),top5=ranks)
                records.append(dict(case=row['case'],prediction_sha256=sha(folder/(row['case']+'.npz')),
                    native_prediction_seconds=float(np.median(times))))
            record=dict(name=key,method=name,classes=classes.tolist(),feature='mean',target=target,components=n,
                fit_seconds=fit_seconds,predictions=records,linear_export_max_error=error,
                linear_export_qualified=error<1e-8)
            save(folder/'PREDICTIONS.json',record);frozen.append(record)
    save(output/'ALL_VALIDATION_PREDICTIONS_FROZEN.json',dict(test_annotations_read=False,
        candidates={r['name']:sha(output/r['name']/'PREDICTIONS.json') for r in frozen}))
    results=[]
    for model in frozen:
        scores=score_predictions(root,registry,output/model['name'],model['classes'],model['predictions'])
        scores.update(name=model['name'],method=model['method'],components=model['components'],fit_seconds=model['fit_seconds'],
            mean_head_seconds=float(np.mean([r['native_prediction_seconds'] for r in model['predictions']])))
        results.append(scores)
    results.sort(key=lambda r:(-r['balanced_accuracy'],-r['top1']))
    save(output/'RESULTS.json',dict(ranking=results,test_annotations_read=False,cycle_complete_for_validation=True,
        unit='SLIC region; region-unit metrics cannot be compared numerically with tile-unit metrics'))
    print(__import__('json').dumps({k:v for k,v in results[0].items() if k!='cases'}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','train','cycle'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cv2.setNumThreads(2)
    with threadpool_limits(limits=2):
        if a.mode=='cycle':
            prepare(a.root,a.output/'regions-v1');train(a.output/'regions-v1',a.output/'cycle-03')
        else:(prepare if a.mode=='prepare' else train)(a.root,a.output)


if __name__=='__main__':main()
