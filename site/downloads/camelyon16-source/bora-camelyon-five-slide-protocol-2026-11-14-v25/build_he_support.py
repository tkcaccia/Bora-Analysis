"""H&E-only tissue area for shared display support and strict tile admission."""
import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from affine import Affine
from rasterio.features import rasterize, shapes
import tifffile

DATA=Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
CAL_ROOT=DATA/'external-camelyon16-five-slide-2026-11-14-v24/calibrated'
CAL_STATE=Path('/mnt/sata_ssd/ChatGPT/Bora/migration/bora-camelyon-five-slide-protocol-2026-11-14-v24/CALIBRATION_STATE.json')
SLIDES=('tumor_084','tumor_091','normal_019')
OUT=DATA/'external-camelyon16-five-slide-2026-11-14-v25/he-support'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()

def main(slide):
    if slide not in SLIDES:raise ValueError('Frozen CAMELYON16 three test slides only')
    state=json.loads(CAL_STATE.read_text())
    if state['status']!='COMPLETE' or state['expert_reference_opened']:
        raise RuntimeError('Calibrated H&E receipt differs')
    row=next(item for item in state['completed'] if item['slide_id']==slide)
    source=CAL_ROOT/slide
    receipt={'status':'COMPLETE','expert_annotation_used':False,
             'unpadded_shape_yx':row['unpadded_shape_yx'],
             'target_shape_yxc':row['padded_shape_yx']+[3],
             'target_sha256':row['calibrated_sha256'],
             'overview_sha256':row['overview_sha256']}
    overview=source/'he-overview.jpg'
    if sha(overview)!=receipt['overview_sha256']:
        raise RuntimeError('H&E overview hash differs')
    rgb=np.asarray(Image.open(overview).convert('RGB'))
    h,w=rgb.shape[:2]
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    strong=(hsv[:,:,1]>=17)&(np.min(rgb,axis=2)<242)
    distance=cv2.distanceTransform((~strong).astype(np.uint8),cv2.DIST_L2,5)
    weak=(hsv[:,:,1]>=8)&(np.min(rgb,axis=2)<248)&(distance<=15)
    closed=cv2.morphologyEx(weak.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))>0
    closed=cv2.dilate(closed.astype(np.uint8),np.ones((5,5),np.uint8))>0
    n,lab,stats,_=cv2.connectedComponentsWithStats(closed.astype(np.uint8),8)
    mask=np.zeros((h,w),np.uint8)
    for label in range(1,n):
        if stats[label,cv2.CC_STAT_AREA]>=120 and int(strong[lab==label].sum())>=80:
            mask[lab==label]=1
    inverse=(mask==0).astype(np.uint8)
    n,lab,stats,_=cv2.connectedComponentsWithStats(inverse,8)
    for label in range(1,n):
        x,y,ww,hh,area=stats[label].tolist()
        if x>0 and y>0 and x+ww<w and y+hh<h and area<900:
            mask[lab==label]=1
    frame_h,frame_w=receipt['unpadded_shape_yx']
    transform=Affine.scale(frame_w/w,frame_h/h)
    geoms=[{'type':'Feature','properties':{'source':'H&E-only reviewed tissue area'},'geometry':geom}
           for geom,value in shapes(mask,mask=mask.astype(bool),transform=transform) if value==1]
    check=rasterize(((f['geometry'],1) for f in geoms),out_shape=(h,w),transform=transform,
                    fill=0,all_touched=False,dtype=np.uint8)
    if not np.array_equal(mask,check):
        raise RuntimeError('Tissue GeoJSON does not exactly roundtrip to the source raster')
    directory=OUT/slide
    if directory.exists():raise FileExistsError('Support exists; preserve version')
    directory.mkdir(parents=True)
    tissue=directory/'tissue-mask-thumbnail.tif'
    tifffile.imwrite(tissue,(mask*255).astype(np.uint8),photometric='minisblack',compression='zlib')
    geo={'type':'FeatureCollection','coordinate_system':'calibrated_image_pixel_xy',
         'image_shape_yx':receipt['target_shape_yxc'][:2],
         'unpadded_image_shape_yx':[frame_h,frame_w],'mask_shape_yx':[h,w],
         'source_image_sha256':receipt['target_sha256'],
         'features':geoms}
    geo_path=directory/'tissue-area.geojson'
    geo_path.write_text(json.dumps(geo,separators=(',',':'))+'\n')
    canvas=rgb.copy()
    canvas[mask==0]=(canvas[mask==0]*.4+np.array([80,85,95])*.6).astype(np.uint8)
    contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas,contours,-1,(20,200,110),2)
    Image.fromarray(canvas).save(directory/'review.jpg',quality=90)
    summary={'status':'H_AND_E_ONLY_TISSUE_SUPPORT_PENDING_VISUAL_REVIEW',
             'slide':slide,'source_image_sha256':receipt['target_sha256'],
             'overview_sha256':sha(overview),'tissue_mask_sha256':sha(tissue),
             'tissue_geojson':{'path':str(geo_path),'sha256':sha(geo_path),'roundtrip_exact':True},
             'review_image_sha256':sha(directory/'review.jpg'),
             'mask_shape_yx':[h,w],'calibrated_frame_shape_yx':[frame_h,frame_w],
             'support_pixels':int(mask.sum()),'fraction':float(mask.mean()),
             'components':len(geoms),'method':'weak HSV saturation >=8 and min RGB <248 within 15 thumbnail pixels of strong stain (saturation >=17 and min RGB <242), connected to >=80 strong-stain pixels; 15px close, 5px dilate, keep >=120px components; fill enclosed holes <900px',
             'expert_annotation_used':False,'artifact_model_used':False,
             'large_voids_preserved':True,'small_adipose_lacunae_filled':True}
    (directory/'SUPPORT_PENDING_REVIEW.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(slide,summary['fraction'],summary['components'],flush=True)

if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: build_he_support.py tumor_084|tumor_091|normal_019')
    main(sys.argv[1])
