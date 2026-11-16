"""Source-only VISTA overview QA using only independently audited masks."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import rasterio

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE = Path(__file__).resolve().parent
HE = DATA / 'external-camelyon16-five-slide-2026-11-14-v24/calibrated'
MASKS = DATA / 'external-camelyon16-five-slide-2026-11-15-v141/vista-masks'
PALETTE = np.array([[0,0,0],[225,55,65],[87,145,80],[135,69,180],
                    [142,101,70],[180,70,110],[80,170,180],[221,178,73],
                    [194,95,176],[98,98,188],[87,180,132],[70,130,220],
                    [208,158,110],[162,115,210]], dtype=np.float32)


def digest(path):
    h = hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def main():
    rows=[]
    for slide in ('tumor_084','tumor_091','normal_019'):
        case=MASKS/slide
        audit=case/'MASKS_AUDIT_VERIFIED.json'
        assert json.loads(audit.read_text())['status']=='VERIFIED'
        receipt=json.loads((case/'ASSEMBLY_COMPLETE.json').read_text())
        he_path=HE/slide/'he-overview.jpg'
        image=np.array(Image.open(he_path).convert('RGB'))
        h,w=image.shape[:2]
        with rasterio.open(receipt['shared_path']) as ds:
            mask=ds.read(1)
        assert digest(receipt['shared_path'])==receipt['shared_sha256']
        mask=cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
        blended=image.astype(np.float32)
        tissue=mask>0
        blended[tissue]=.68*blended[tissue]+.32*PALETTE[mask[tissue]]
        tumor=image.astype(np.float32)
        broad=mask==1
        tumor[broad]=.55*tumor[broad]+.45*PALETTE[1]
        panels=[image,np.uint8(np.clip(blended,0,255)),np.uint8(np.clip(tumor,0,255))]
        labels=('H&E','Fixed 13-class VISTA','Broad tumor prompt only')
        tw=640; th=round(h*tw/w)
        canvas=Image.new('RGB',(3*tw,th+32),'white');draw=ImageDraw.Draw(canvas)
        for i,(panel,label) in enumerate(zip(panels,labels)):
            canvas.paste(Image.fromarray(panel).resize((tw,th),Image.Resampling.BILINEAR),(i*tw,32))
            draw.text((i*tw+8,8),f'{slide}: {label}',fill=(20,20,20))
        path=HERE/f'{slide}-vista-source-only-review.png'
        canvas.save(path)
        rows.append({'slide':slide,'path':str(path),'sha256':digest(path),
                     'he_overview_sha256':digest(he_path),'mask_audit_sha256':digest(audit)})
    out=HERE/'REVIEW_RENDERED.json'
    out.write_text(json.dumps({'status':'SOURCE_ONLY_VISTA_QA_RENDERED',
        'rows':rows,'test_reference_opened':False,'online_publication':False},indent=2)+'\n')
    print(json.dumps({'status':'SOURCE_ONLY_VISTA_QA_RENDERED','slides':len(rows)}))


if __name__=='__main__':main()
