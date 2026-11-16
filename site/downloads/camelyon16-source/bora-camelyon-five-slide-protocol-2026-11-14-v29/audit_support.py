#!/usr/bin/env python3
"""Independent source-only TIFF/GeoJSON pixel roundtrip for three CAM supports."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import tifffile
from affine import Affine
from rasterio.features import rasterize

ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
STATE=ROOT/'bora-camelyon-five-slide-protocol-2026-11-14-v33/CORRECTED_SUPPORT_STATE.json'
CAL=ROOT/'bora-camelyon-five-slide-protocol-2026-11-14-v24/CALIBRATION_STATE.json'
OUT=Path(__file__).parent/'SUPPORT_AUDIT_VERIFIED.json'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    if OUT.exists():raise FileExistsError(OUT)
    state=json.loads(STATE.read_text())
    cal=json.loads(CAL.read_text())
    assert state['status']=='COMPLETE_PENDING_INDEPENDENT_AUDIT_AND_VISUAL_REVIEW'
    assert state['calibration_sha256']==sha(CAL)
    assert not state['expert_reference_opened'] and not cal['expert_reference_opened']
    assert [r['slide_id'] for r in state['completed']]==['tumor_084','tumor_091','normal_019']
    by_id={r['slide_id']:r for r in cal['completed']}
    rows=[]
    for item in state['completed']:
        sid=item['slide_id']
        receipt_path=Path(item['receipt_path'])
        assert sha(receipt_path)==item['receipt_sha256']
        receipt=json.loads(receipt_path.read_text())
        assert receipt['slide']==sid and not receipt['expert_annotation_used']
        assert receipt['source_image_sha256']==by_id[sid]['calibrated_sha256']
        mask_path=receipt_path.parent/'tissue-mask-thumbnail.tif'
        geo_path=Path(receipt['tissue_geojson']['path'])
        assert sha(mask_path)==receipt['tissue_mask_sha256']
        assert sha(geo_path)==receipt['tissue_geojson']['sha256']
        geo=json.loads(geo_path.read_text())
        assert geo['type']=='FeatureCollection'
        assert geo['source_image_sha256']==by_id[sid]['calibrated_sha256']
        assert geo['image_shape_yx']==by_id[sid]['padded_shape_yx']
        assert geo['unpadded_image_shape_yx']==by_id[sid]['unpadded_shape_yx']
        image=tifffile.imread(mask_path)
        truth=(image!=0).astype(np.uint8)
        assert list(truth.shape)==geo['mask_shape_yx']==receipt['mask_shape_yx']
        height,width=geo['unpadded_image_shape_yx']
        native_h,native_w=truth.shape
        transform=Affine.scale(width/native_w,height/native_h)
        restored=rasterize(((feature['geometry'],1) for feature in geo['features']),
                           out_shape=truth.shape,transform=transform,fill=0,dtype=np.uint8)
        differences=int(np.count_nonzero(restored!=truth))
        assert differences==0
        assert int(truth.sum())==receipt['support_pixels']
        assert abs(float(truth.mean())-receipt['fraction'])<1e-12
        rows.append({'slide_id':sid,'mask_sha256':sha(mask_path),
                     'geojson_sha256':sha(geo_path),'roundtrip_different_pixels':0,
                     'support_pixels':int(truth.sum()),'fraction':float(truth.mean()),
                     'review_image':str(receipt_path.parent/'review.jpg'),
                     'review_sha256':sha(receipt_path.parent/'review.jpg')})
    result={'format':'bora-cam-three-support-audit-v1','status':'PIXEL_VERIFIED_VISUAL_REVIEW_PENDING',
            'support_state_sha256':sha(STATE),'calibration_state_sha256':sha(CAL),
            'rows':rows,'expert_reference_opened':False,'online_publication':False}
    temporary=OUT.with_suffix('.json.partial')
    with temporary.open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,OUT)
    print(json.dumps({'status':result['status'],'slides':len(rows)}))


if __name__=='__main__':main()
