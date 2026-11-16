"""Freeze test-slide MedSAM boxes from source-only binary U-Net class 2."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA=Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE=Path(__file__).resolve().parent
UNET=DATA/'external-camelyon16-five-slide-2026-11-14-v114/binary-network-masks/unet'
INPUTS=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-14-v103/INPUTS_FROZEN.json'
OLD_PROMPTS=ROOT/'migration/external-benchmark-wsss-camelyon-2026-11-14-v106/MEDSAM_PROMPTS_FROZEN.json'
TILE=128


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def main():
    destination=HERE/'MEDSAM_PROMPTS_FROZEN.json'
    if destination.exists():
        raise FileExistsError('No in-place prompt refreeze')
    old=json.loads(OLD_PROMPTS.read_text())
    assert sha(old['model_checkpoint_path'])==old['model_checkpoint_sha256']
    inputs=json.loads(INPUTS.read_text())
    rows=[]
    for slide in ('tumor_084','tumor_091','normal_019'):
        folder=UNET/slide
        audit_path=folder/'MASKS_AUDIT_VERIFIED.json'
        audit=json.loads(audit_path.read_text())
        assert audit['status']=='VERIFIED' and audit['slide']==slide and not audit['test_reference_opened']
        receipt=json.loads((folder/'ASSEMBLY_COMPLETE.json').read_text())
        mask_path=Path(receipt['shared_path'])
        assert sha(mask_path)==receipt['shared_sha256']==audit['shared_sha256']
        height,width=receipt['comparison_shape_yx']
        assert inputs['slides'][slide]['calibrated_shape_yx']==[height*8,width*8]
        tiles=[]
        with rasterio.open(mask_path) as src:
            assert (src.height,src.width)==(height,width)
            for y0 in range(0,height,TILE):
                for x0 in range(0,width,TILE):
                    th,tw=min(TILE,height-y0),min(TILE,width-x0)
                    crop=src.read(1,window=Window(x0,y0,tw,th))
                    if not np.any(crop==2):
                        continue
                    n,labels,stats,_=cv2.connectedComponentsWithStats((crop==2).astype(np.uint8),8)
                    parts=[]
                    for i in range(1,n):
                        x,y,w,h,area=map(int,stats[i])
                        if area<4:
                            continue
                        pad=math.ceil(.08*max(w,h))
                        left=max(0,x-pad);top=max(0,y-pad)
                        right=min(tw,x+w+pad);bottom=min(th,y+h+pad)
                        parts.append({'component_pixels_4um':area,
                                      'box_xyxy_native_crop':[left*8,top*8,right*8,bottom*8]})
                    parts.sort(key=lambda p:(-p['component_pixels_4um'],p['box_xyxy_native_crop']))
                    if parts:
                        tiles.append({'origin_yx_4um':[y0,x0],'shape_yx_4um':[th,tw],
                                      'boxes':parts[:3]})
        rows.append({'slide':slide,'shape_yx_4um':[height,width],
                     'calibrated_image_path':inputs['slides'][slide]['calibrated_image_path'],
                     'calibrated_image_sha256':inputs['slides'][slide]['calibrated_image_sha256'],
                     'unet_shared_mask_path':str(mask_path),
                     'unet_shared_mask_sha256':receipt['shared_sha256'],
                     'unet_audit_path':str(audit_path),'unet_audit_sha256':sha(audit_path),
                     'all_tiles':math.ceil(height/TILE)*math.ceil(width/TILE),
                     'prompted_tiles':len(tiles),'prompt_boxes':sum(len(t['boxes']) for t in tiles),
                     'tiles':tiles})
    result={'status':'CAM_TEST_MEDSAM_UNET_GUIDED_BOXES_FROZEN_SOURCE_ONLY',
            'model_checkpoint_path':old['model_checkpoint_path'],
            'model_checkpoint_sha256':old['model_checkpoint_sha256'],
            'tile_size_native_px':1024,'tile_size_comparison_px':128,
            'min_component_pixels_4um':4,'max_boxes_per_tile':3,
            'padding_fraction_of_larger_component_side':.08,
            'role':'metastasis_mask_prompt_assisted',
            'semantic_prompt_source':'CAM binary U-Net class 2 on the common H&E support',
            'no_prompt_tile_output':'zero metastasis; not a method execution failure',
            'source_sha256':sha(__file__),'inputs_sha256':sha(INPUTS),
            'historical_pair_prompt_settings_sha256':sha(OLD_PROMPTS),
            'test_reference_opened':False,'online_publication':False,'slides':rows}
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],
          'slides':[(x['slide'],x['all_tiles'],x['prompted_tiles'],x['prompt_boxes']) for x in rows]}))


if __name__=='__main__':
    main()
