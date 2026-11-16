"""Independent exact mask-to-box audit of CAM test MedSAM prompts."""
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
PROMPTS=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-14-v120/MEDSAM_PROMPTS_FROZEN.json'
HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def main():
    frozen=json.loads(PROMPTS.read_text())
    assert frozen['status']=='CAM_TEST_MEDSAM_UNET_GUIDED_BOXES_FROZEN_SOURCE_ONLY'
    assert sha(frozen['model_checkpoint_path'])==frozen['model_checkpoint_sha256']
    assert not frozen['test_reference_opened']
    rows=[]
    for row in frozen['slides']:
        path=Path(row['unet_shared_mask_path'])
        assert sha(path)==row['unet_shared_mask_sha256']
        assert sha(row['unet_audit_path'])==row['unet_audit_sha256']
        h,w=row['shape_yx_4um']
        index=0
        total_boxes=0
        with rasterio.open(path) as image:
            assert (image.height,image.width)==(h,w)
            for y0 in range(0,h,128):
                for x0 in range(0,w,128):
                    th,tw=min(128,h-y0),min(128,w-x0)
                    patch=image.read(1,window=Window(x0,y0,tw,th))
                    n,labels,stats,_=cv2.connectedComponentsWithStats((patch==2).astype(np.uint8),8)
                    expected=[]
                    for component in range(1,n):
                        x,y,width,height,area=map(int,stats[component])
                        if area<4:
                            continue
                        pad=math.ceil(.08*max(width,height))
                        box=[max(0,x-pad)*8,max(0,y-pad)*8,
                             min(tw,x+width+pad)*8,min(th,y+height+pad)*8]
                        expected.append({'component_pixels_4um':area,'box_xyxy_native_crop':box})
                    expected.sort(key=lambda v:(-v['component_pixels_4um'],v['box_xyxy_native_crop']))
                    if not expected:
                        continue
                    tile=row['tiles'][index]
                    assert tile['origin_yx_4um']==[y0,x0]
                    assert tile['shape_yx_4um']==[th,tw]
                    assert tile['boxes']==expected[:3]
                    total_boxes+=len(tile['boxes'])
                    index+=1
        assert index==row['prompted_tiles']==len(row['tiles'])
        assert total_boxes==row['prompt_boxes']
        assert row['all_tiles']==math.ceil(h/128)*math.ceil(w/128)
        rows.append({'slide':row['slide'],'prompted_tiles':index,'prompt_boxes':total_boxes,
                     'all_boxes_recomputed':True,'unet_mask_sha256':sha(path)})
    result={'status':'VERIFIED','prompt_manifest_sha256':sha(PROMPTS),
            'checkpoint_sha256':frozen['model_checkpoint_sha256'],
            'rows':rows,'test_reference_opened':False,'online_publication':False}
    path=HERE/'PROMPT_AUDIT_VERIFIED.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'VERIFIED','rows':rows}))


if __name__=='__main__':
    main()
