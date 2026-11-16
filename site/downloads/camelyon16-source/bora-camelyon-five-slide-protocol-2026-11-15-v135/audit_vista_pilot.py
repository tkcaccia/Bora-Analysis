"""Independent identity, probability, geometry and lifecycle VISTA pilot audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window

ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA=Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE=Path(__file__).resolve().parent
FROZEN=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-15-v132/VISTA_SOURCE_FROZEN.json'
SOURCE=FROZEN.with_name('vista_test_tiles.py')
SCOPE=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-15-v133/VISTA_SCOPE_AUDIT_VERIFIED.json'
PILOT=DATA/'external-camelyon16-five-slide-2026-11-15-v134/vista-pilot/tumor_091'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def main():
    frozen=json.loads(FROZEN.read_text())
    assert sha(SOURCE)==frozen['tile_source_sha256']
    assert json.loads(SCOPE.read_text())['status']=='VERIFIED'
    slide=frozen['slides']['tumor_091']
    tile=slide['selected_tiles'][9]
    receipt_path=PILOT/'tile-0009/TILE_COMPLETE.json'
    receipt=json.loads(receipt_path.read_text())
    done=json.loads((PILOT/'RANGE_COMPLETE.json').read_text())
    assert done['status']=='CAM_VISTA_TILE_RANGE_COMPLETE'
    assert (done['slide'],done['start'],done['stop'])==('tumor_091',9,10)
    assert receipt['status']=='CAM_VISTA_SOURCE_ONLY_NATIVE_TILE_COMPLETE'
    assert (receipt['slide'],receipt['selected_tile_index'])==('tumor_091',9)
    assert receipt['origin_yx_native']==tile['origin_yx_native']
    assert receipt['origin_yx_4um']==[v//8 for v in tile['origin_yx_native']]
    assert receipt['prompt_names']==frozen['ordered_class_names']
    assert receipt['checkpoint_sha256']==frozen['checkpoint_sha256']
    assert receipt['calibrated_image_sha256']==slide['calibrated_image_sha256']
    assert receipt['support_receipt_sha256']==slide['support_receipt_sha256']
    assert receipt['source_sha256']==sha(SOURCE)
    assert receipt['freeze_sha256']==sha(FROZEN)
    assert sha(receipt['probabilities_path'])==receipt['probabilities_sha256']
    values=np.load(receipt['probabilities_path'],allow_pickle=False)
    assert values.shape==(13,128,128) and values.dtype==np.float32
    assert np.isfinite(values).all() and 0<=values.min()<=values.max()<=1
    assert abs(float(values.min())-receipt['probability_min'])<1e-7
    assert abs(float(values.max())-receipt['probability_max'])<1e-7
    y,x=tile['origin_yx_native']
    with rasterio.open(slide['calibrated_image_path']) as src:
        rgb=np.moveaxis(src.read(window=Window(x,y,1024,1024)),0,-1)
    rgb=cv2.resize(rgb,(512,512),interpolation=cv2.INTER_AREA)
    tumor=cv2.resize(values[0],(512,512),interpolation=cv2.INTER_NEAREST)
    shown=rgb.astype(np.float32)
    shown[tumor>.5]=.55*shown[tumor>.5]+.45*np.array((235,65,70))
    preview=HERE/'tumor_091-vista-pilot-tumor-prompt.png'
    Image.fromarray(np.uint8(np.clip(shown,0,255))).save(preview)
    active=subprocess.check_output(['systemctl','--user','show','bora-cam16-vista-test-pilot-v134.service','-p','ActiveState','--value'],text=True).strip()
    assert active!='active'
    assert not subprocess.check_output(['systemctl','--user','list-units','bora-*','--state=running','--no-legend','--no-pager'],text=True).strip()
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True).strip()
    record={'status':'VERIFIED','pilot_receipt_sha256':sha(receipt_path),
            'probabilities_sha256':receipt['probabilities_sha256'],
            'probability_shape':[13,128,128],
            'tumor_prompt_pixels_gt_0p5':int((values[0]>.5).sum()),
            'preview_path':str(preview),'preview_sha256':sha(preview),
            'source_freeze_sha256':sha(FROZEN),
            'terminal_service_state':active,'terminal_gpu_contexts':[],
            'test_reference_opened':False,'online_publication':False}
    path=HERE/'VISTA_PILOT_AUDIT_VERIFIED.json'
    path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'VERIFIED','tumor_prompt_pixels_gt_0p5':record['tumor_prompt_pixels_gt_0p5']}))


if __name__=='__main__':main()
