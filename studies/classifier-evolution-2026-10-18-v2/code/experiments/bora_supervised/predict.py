"""Native CPU prediction for a new calibrated ROI from full Virchow2 tokens.

No annotation or label-set input is accepted. The training bundle reconstructs
the fitted C++ head once; its startup cost is reported separately.
"""
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from threadpoolctl import threadpool_limits
import tifffile
from .registry import load,save,sha
from .units import tile_boxes,pool_tokens,ranked_labels
from .cycle_two import context_features
from .regions import image_regions,pool_regions


def load_model(bundle):
    from fastpls import PLS
    from sklearn.linear_model import Ridge
    bundle=Path(bundle);record=load(bundle/'MODEL.json')
    for name in ('training_x','training_y'):
        if sha(bundle/(name+'.npy'))!=record[name+'_sha256']:raise ValueError('Training bundle changed')
    x=np.load(bundle/'training_x.npy');y=np.load(bundle/'training_y.npy');classes=np.array(record['classes'])
    began=time.monotonic()
    if record['engine']=='fastPLS C++ CPU':
        model=PLS(n_components=record['components'],classifier=record['classifier'],backend='cpu',seed=record['seed'],
                  method=record['method'],scaling=record['scaling']).fit(x,y)
        predict=(lambda a:model.predict(a,top=min(5,len(classes)))) if record['classifier'] else (
            lambda a:ranked_labels(model.predict_responses(a),classes,min(5,len(classes))))
    else:
        model=Ridge(alpha=record['alpha']).fit(x,(y[:,None]==classes).astype(float))
        predict=lambda a:ranked_labels(model.predict(a),classes,min(5,len(classes)))
    return predict,record,time.monotonic()-began


def validate_features(tokens,image,mpp,metadata,contract,image_path):
    mpp=np.asarray(mpp,float)
    if mpp.shape!=(2,) or not np.isfinite(mpp).all() or np.any(mpp<=0):raise ValueError('Positive pixel spacings required')
    if not np.allclose(mpp,contract['working_mpp_xy'],rtol=.01,atol=0):
        raise ValueError('Resample the H&E to0.5um/px and rerun Virchow2; changing the MPP of existing embeddings does not recalibrate their encoder context')
    if tokens.ndim!=3 or tokens.shape[-1]!=contract['columns'] or tokens.shape[0]*14<image.shape[0] or tokens.shape[1]*14<image.shape[1]:
        raise ValueError('Expected full unselected1280-channel spatial tokens covering the image')
    if metadata['patch_px']!=contract['patch_px'] or metadata['window_px']!=contract['window_px']:
        raise ValueError('Different patch spacing or encoder window changes the physical feature context')
    if metadata['image_sha256']!=sha(image_path):raise ValueError('Features belong to a different image')
    if metadata.get('encoder_revision')!=contract['revision']:raise ValueError('Explicit matching encoder revision provenance is required')
    if metadata.get('selected_channels_used',False) or metadata.get('selected_dimensions') is not None:
        raise ValueError('Selected-channel caches are not accepted')


def run(bundle,features,image_path,mpp,feature_metadata,output):
    output=Path(output)
    if output.exists():raise ValueError('Preserve previous prediction output')
    image=np.asarray(Image.open(image_path).convert('RGB'));tokens=np.load(features,mmap_mode='r')
    predictor,record,startup=load_model(bundle)
    metadata=load(feature_metadata)
    validate_features(tokens,image,mpp,metadata,record['feature_contract'],image_path)
    if sha(features)!=metadata['sha256']:raise ValueError('Feature tensor hash differs from its provenance')
    began=time.monotonic();boxes=None
    if record['name'].startswith(('S08','S09')):
        regions,_=image_regions(image,mpp);x=pool_regions(tokens,regions,image.shape[:2])
        units=cv2.resize(regions,image.shape[1::-1],interpolation=cv2.INTER_NEAREST)
    else:
        boxes=tile_boxes(image.shape[:2],mpp);mean,centre=pool_tokens(tokens,boxes,14)
        x=centre if record['feature']=='centre' else mean
        if record['feature']=='context':x=context_features(x,boxes,mpp)
        units=np.zeros(image.shape[:2],np.int32)
        for i,(bx,by,w,h) in enumerate(boxes):units[by:by+h,bx:bx+w]=i
    feature_seconds=time.monotonic()-began
    began=time.monotonic();ranks=predictor(np.asarray(x,dtype=np.float64));head_seconds=time.monotonic()-began
    # The same conservative exterior-white heuristic used in the experiments.
    rgb=image.astype(np.float32)/255;maximum=rgb.max(-1);minimum=rgb.min(-1)
    white=(minimum>.9)&((maximum-minimum)/np.maximum(maximum,1e-6)<.12)
    cc,_=ndi.label(white,np.ones((3,3)));border=np.unique(np.r_[cc[0],cc[-1],cc[:,0],cc[:,-1]]);border=border[border>0]
    glass=np.isin(cc,border)&white;mask=ranks[:,0][units].astype(np.uint8);mask[glass]=0
    output.mkdir(parents=True);tifffile.imwrite(output/'top1.tif',mask,compression='deflate')
    np.savez_compressed(output/'ranked_units.npz',top5=ranks,unit_map=units)
    roles=record['ontology']['roles']
    save(output/'tissue_options.json',dict(model=record['name'],native_head=record['engine'],
        startup_refit_seconds=startup,feature_pooling_seconds=feature_seconds,head_seconds=head_seconds,
        mpp_xy=list(mpp),feature_metadata_sha256=sha(feature_metadata),input_image_sha256=sha(image_path),
        ranks_are_calibrated_probabilities=False,annotation_input_used=False,
        units=[dict(unit=i,top1_id=int(r[0]),top1_name=roles[r[0]-1],
                    top5=[dict(rank=j+1,id=int(g),name=roles[g-1]) for j,g in enumerate(r)],
                    **({'box_xywh':boxes[i].tolist()} if boxes is not None else {})) for i,r in enumerate(ranks)]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','features','image','feature-metadata','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mpp-xy',type=float,nargs=2,required=True);a=p.parse_args();cv2.setNumThreads(2)
    with threadpool_limits(limits=2):run(a.model,a.features,a.image,a.mpp_xy,a.feature_metadata,a.output)


if __name__=='__main__':main()
