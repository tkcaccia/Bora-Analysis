"""Independent CAM test MedSAM source-only pilot and visual audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw
import rasterio
from rasterio.windows import Window
import subprocess

ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA=Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE=Path(__file__).resolve().parent
PROMPTS=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-14-v120/MEDSAM_PROMPTS_FROZEN.json'
AUDIT=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-14-v121/PROMPT_AUDIT_VERIFIED.json'
SOURCE=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-14-v122/infer_medsam_test.py'
FREEZE=SOURCE.with_name('MEDSAM_SOURCE_FROZEN.json')
FOLDER=DATA/'external-camelyon16-five-slide-2026-11-14-v123/medsam-pilot/tumor_091/range-0000-0001'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def main():
    prompts=json.loads(PROMPTS.read_text())
    frozen=json.loads(FREEZE.read_text())
    tile=next(x for x in prompts['slides'] if x['slide']=='tumor_091')['tiles'][0]
    assert sha(PROMPTS)==frozen['prompt_manifest_sha256']
    assert sha(AUDIT)==frozen['prompt_audit_sha256']
    assert sha(SOURCE)==frozen['source_sha256']
    assert sha(prompts['model_checkpoint_path'])==frozen['checkpoint_sha256']
    receipt_path=FOLDER/'tile-0000/TILE_COMPLETE.json'
    receipt=json.loads(receipt_path.read_text())
    complete=json.loads((FOLDER/'RANGE_COMPLETE.json').read_text())
    assert complete['status']=='CAM_TEST_MEDSAM_TILE_RANGE_COMPLETE'
    assert (complete['slide'],complete['start'],complete['stop'])==('tumor_091',0,1)
    assert (receipt['slide'],receipt['prompt_tile_index'])==('tumor_091',0)
    assert receipt['status']=='CAM_TEST_MEDSAM_PROMPTED_TILE_COMPLETE_SOURCE_ONLY'
    assert receipt['origin_yx_4um']==tile['origin_yx_4um']
    assert receipt['shape_yx_4um']==tile['shape_yx_4um']
    assert receipt['boxes']==tile['boxes']
    assert receipt['checkpoint_sha256']==frozen['checkpoint_sha256']
    assert receipt['prompt_manifest_sha256']==sha(PROMPTS)
    assert receipt['source_sha256']==sha(SOURCE)
    assert sha(receipt['mask_path'])==receipt['mask_sha256']
    mask=np.load(receipt['mask_path'],allow_pickle=False)
    assert mask.dtype==np.uint8 and mask.shape==tuple(tile['shape_yx_4um'])
    assert set(np.unique(mask).tolist()).issubset({0,1})
    assert int(mask.sum())==receipt['metastasis_pixels_4um']
    assert not receipt['test_reference_opened']
    slide=next(x for x in prompts['slides'] if x['slide']=='tumor_091')
    y,x=tile['origin_yx_4um'];h,w=tile['shape_yx_4um']
    with rasterio.open(slide['calibrated_image_path']) as source:
        rgb=np.moveaxis(source.read(window=Window(x*8,y*8,w*8,h*8)),0,-1)
    rgb=cv2.resize(rgb,(w,h),interpolation=cv2.INTER_AREA)
    shown=rgb.astype(np.float32)
    shown[mask>0]=.55*shown[mask>0]+.45*np.array((235,65,70))
    image=Image.fromarray(np.uint8(np.clip(shown,0,255))).resize((512,512))
    draw=ImageDraw.Draw(image)
    for item in tile['boxes']:
        a,b,c,d=item['box_xyxy_native_crop']
        draw.rectangle([a/2,b/2,c/2-1,d/2-1],outline=(255,220,0),width=2)
    preview=HERE/'tumor_091-first-medsam-test-tile.png'
    image.save(preview)
    assert subprocess.check_output(['systemctl','--user','list-units','bora-*','--state=running','--no-legend','--no-pager'],text=True).strip()==''
    assert subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True).strip()==''
    record={'status':'VERIFIED','pilot_receipt_sha256':sha(receipt_path),
            'mask_sha256':sha(receipt['mask_path']),'mask_pixels':int(mask.sum()),
            'prompt_manifest_sha256':sha(PROMPTS),'source_freeze_sha256':sha(FREEZE),
            'preview_path':str(preview),'preview_sha256':sha(preview),
            'terminal_owned_gpu_contexts':[],'test_reference_opened':False,'online_publication':False}
    (HERE/'MEDSAM_PILOT_AUDIT_VERIFIED.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'VERIFIED','mask_pixels':int(mask.sum()),
                      'preview':str(preview)}))


if __name__=='__main__':
    main()
