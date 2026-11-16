#!/usr/bin/env python3
"""Pin H&E-only removal of two non-tissue components on tumor_091."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image
import tifffile
from affine import Affine
from rasterio.features import rasterize,shapes

DATA=Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
SOURCE=DATA/'external-camelyon16-five-slide-2026-11-14-v32/he-support'
STATE_SOURCE=ROOT/'bora-camelyon-five-slide-protocol-2026-11-14-v32/SUPPORT_STATE.json'
OUT=DATA/'external-camelyon16-five-slide-2026-11-14-v33/he-support'
STATE=Path(__file__).parent/'CORRECTED_SUPPORT_STATE.json'
SLIDES=('tumor_084','tumor_091','normal_019')
REMOVE={(0,1022,18,36,631):'non-tissue edge fragment',
        (1135,1279,152,110,12785):'cyan scanner artifact'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.partial')
    with temp.open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def main():
    if STATE.exists() or OUT.exists():raise FileExistsError('Preserve prior support version')
    old=json.loads(STATE_SOURCE.read_text())
    assert old['status']=='COMPLETE_PENDING_INDEPENDENT_AUDIT_AND_VISUAL_REVIEW'
    assert [x['slide_id'] for x in old['completed']]==list(SLIDES)
    assert not old['expert_reference_opened']
    OUT.mkdir(parents=True)
    state={'format':'bora-cam-test-support-visual-correction-v1','status':'RUNNING',
           'v32_state_sha256':sha(STATE_SOURCE),'started_unix':time.time(),
           'pid':os.getpid(),'completed':[],'expert_reference_opened':False,
           'online_publication':False}
    atomic(STATE,state)
    try:
        for slide in SLIDES:
            olddir=SOURCE/slide
            newdir=OUT/slide
            newdir.mkdir()
            prior_path=olddir/'SUPPORT_PENDING_REVIEW.json'
            prior=json.loads(prior_path.read_text())
            assert prior['slide']==slide and not prior['expert_annotation_used']
            correction=[]
            if slide!='tumor_091':
                for filename in ('tissue-mask-thumbnail.tif','tissue-area.geojson','review.jpg'):
                    shutil.copy2(olddir/filename,newdir/filename)
                assert all(sha(olddir/f)==sha(newdir/f) for f in
                           ('tissue-mask-thumbnail.tif','tissue-area.geojson','review.jpg'))
                accepted=prior.copy()
            else:
                oldmask=(tifffile.imread(olddir/'tissue-mask-thumbnail.tif')>0).astype(np.uint8)
                labels_count,labels,stats,_=cv2.connectedComponentsWithStats(oldmask,8)
                found={}
                for label in range(1,labels_count):
                    x,y,w,h,area=map(int,stats[label])
                    signature=(x,y,w,h,area)
                    if signature in REMOVE:
                        found[signature]=label
                if set(found)!=set(REMOVE):
                    raise ValueError('Reviewed scanner/edge components changed')
                mask=oldmask.copy()
                for signature,label in found.items():
                    mask[labels==label]=0
                    correction.append({'bbox_xywh':list(signature[:4]),
                                       'pixels_removed':signature[4],
                                       'reason':REMOVE[signature]})
                assert int(oldmask.sum()-mask.sum())==sum(x['pixels_removed'] for x in correction)
                tifffile.imwrite(newdir/'tissue-mask-thumbnail.tif',(mask*255).astype(np.uint8),
                                 photometric='minisblack',compression='zlib')
                previous_geo=json.loads((olddir/'tissue-area.geojson').read_text())
                frame_h,frame_w=previous_geo['unpadded_image_shape_yx']
                height,width=mask.shape
                transform=Affine.scale(frame_w/width,frame_h/height)
                features=[{'type':'Feature','properties':{'source':'H&E-only reviewed tissue area'},
                           'geometry':geom} for geom,value in
                          shapes(mask,mask=mask.astype(bool),transform=transform) if value==1]
                check=rasterize(((f['geometry'],1) for f in features),out_shape=mask.shape,
                                transform=transform,fill=0,dtype=np.uint8)
                if not np.array_equal(check,mask):
                    raise ValueError('Corrected tissue polygons do not roundtrip')
                newgeo=previous_geo.copy();newgeo['features']=features
                (newdir/'tissue-area.geojson').write_text(json.dumps(newgeo,separators=(',',':'))+'\n')
                overview=DATA/'external-camelyon16-five-slide-2026-11-14-v24/calibrated'/slide/'he-overview.jpg'
                rgb=np.asarray(Image.open(overview).convert('RGB'))
                canvas=rgb.copy()
                canvas[mask==0]=(canvas[mask==0]*.4+np.array([80,85,95])*.6).astype(np.uint8)
                contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(canvas,contours,-1,(20,200,110),2)
                Image.fromarray(canvas).save(newdir/'review.jpg',quality=90)
                accepted=prior.copy()
                accepted['support_pixels']=int(mask.sum())
                accepted['fraction']=float(mask.mean())
                accepted['components']=len(features)
            accepted['tissue_mask_sha256']=sha(newdir/'tissue-mask-thumbnail.tif')
            accepted['tissue_geojson']={'path':str(newdir/'tissue-area.geojson'),
                                        'sha256':sha(newdir/'tissue-area.geojson'),
                                        'roundtrip_exact':True}
            accepted['review_image_sha256']=sha(newdir/'review.jpg')
            accepted['v32_candidate_receipt_sha256']=sha(prior_path)
            accepted['reviewed_component_correction']=correction
            path=newdir/'SUPPORT_PENDING_REVIEW.json'
            atomic(path,accepted)
            state['completed'].append({'slide_id':slide,'receipt_path':str(path),
                                       'receipt_sha256':sha(path),
                                       'fraction':accepted['fraction'],
                                       'correction':correction})
            atomic(STATE,state)
            print(slide,accepted['fraction'],correction,flush=True)
        state['status']='COMPLETE_PENDING_INDEPENDENT_AUDIT_AND_VISUAL_REVIEW'
        state['completed_unix']=time.time()
        atomic(STATE,state)
    except BaseException as exc:
        state['status']='FAILED_PRESERVE_PARTIAL'
        state['error']=f'{type(exc).__name__}: {exc}'
        state['failed_unix']=time.time()
        atomic(STATE,state)
        raise


if __name__=='__main__':main()
