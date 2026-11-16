"""Matched H&E/native-binary-source-only overlays for visual QA."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import rasterio

ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA=Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE=Path(__file__).resolve().parent
IMAGES=DATA/'external-camelyon16-five-slide-2026-11-14-v24/calibrated'
NETWORK=DATA/'external-camelyon16-five-slide-2026-11-14-v114/binary-network-masks'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def overlay(rgb,mask):
    result=rgb.astype(np.float32)
    for klass,color in ((1,(55,130,230)),(2,(235,70,75))):
        area=mask==klass
        result[area]=.62*result[area]+.38*np.array(color)
    return np.uint8(np.clip(result,0,255))


def main():
    rows=[]
    for slide in ('tumor_084','tumor_091','normal_019'):
        he_path=IMAGES/slide/'he-overview.jpg'
        he=np.asarray(Image.open(he_path).convert('RGB'))
        height,width=he.shape[:2]
        panels=[he]
        receipts={}
        if slide=='tumor_091':
            bora=DATA/'external-camelyon16-five-slide-2026-11-14-v99/bora-test-mask/tumor_091/MASK_COMPLETE.json'
        else:
            bora=DATA/f'external-camelyon16-five-slide-2026-11-14-v102/bora-test-masks/{slide}/MASK_COMPLETE.json'
        record=json.loads(bora.read_text())
        path=Path(record['final_mask']['path'])
        assert sha(path)==record['final_mask']['sha256']
        with rasterio.open(path) as src:
            mask=src.read(1)
        panels.append(overlay(he,cv2.resize(mask,(width,height),interpolation=cv2.INTER_NEAREST)))
        receipts['bora']=sha(bora)
        for method in ('unet','deeplabv3plus'):
            folder=NETWORK/method/slide
            audit=folder/'MASKS_AUDIT_VERIFIED.json'
            assert json.loads(audit.read_text())['status']=='VERIFIED'
            record=json.loads((folder/'ASSEMBLY_COMPLETE.json').read_text())
            path=Path(record['shared_path'])
            assert sha(path)==record['shared_sha256']
            with rasterio.open(path) as src:
                mask=src.read(1)
            panels.append(overlay(he,cv2.resize(mask,(width,height),interpolation=cv2.INTER_NEAREST)))
            receipts[method]=sha(audit)
        tile_width=640
        tile_height=round(height*tile_width/width)
        result=Image.new('RGB',(tile_width*4,tile_height+32),'white')
        draw=ImageDraw.Draw(result)
        labels=('H&E','Bora','U-Net','DeepLabV3+')
        for i,(label,panel) in enumerate(zip(labels,panels)):
            image=Image.fromarray(panel).resize((tile_width,tile_height),Image.Resampling.BILINEAR)
            result.paste(image,(i*tile_width,32))
            draw.text((i*tile_width+8,8),f'{slide} | {label}',fill=(20,20,20))
        out=HERE/f'{slide}-source-only-review.png'
        result.save(out)
        rows.append({'slide':slide,'path':str(out),'sha256':sha(out),
                     'source_image_sha256':sha(he_path),'audit_hashes':receipts,
                     'expert_reference_opened':False})
    record={'status':'SOURCE_ONLY_REVIEW_RENDERED','rows':rows,'online_publication':False}
    (HERE/'REVIEW_RENDERED.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'SOURCE_ONLY_REVIEW_RENDERED','slides':len(rows)}))


if __name__=='__main__':
    main()
