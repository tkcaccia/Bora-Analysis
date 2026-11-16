#!/usr/bin/env python3
"""Independent source-to-calibrated RGB crop, hash and padding audit."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pyvips

ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
STATE=ROOT/'bora-camelyon-five-slide-protocol-2026-11-14-v24/CALIBRATION_STATE.json'
HERE=Path(__file__).parent
OUT=HERE/'CALIBRATED_AUDIT_VERIFIED.json'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(16<<20),b''):h.update(block)
    return h.hexdigest()


def pixels(image,x,y,w,h):
    crop=image.crop(x,y,w,h)
    return np.ndarray(buffer=crop.write_to_memory(),dtype=np.uint8,
                      shape=(h,w,3)).copy()


def run():
    if OUT.exists():raise FileExistsError(OUT)
    state=json.loads(STATE.read_text())
    if state['status']!='COMPLETE' or state['expert_reference_opened'] or \
       [r['slide_id'] for r in state['completed']]!=['tumor_084','tumor_091','normal_019']:
        raise ValueError('Frozen three-slide calibration incomplete')
    rows=[]
    try:
        for r in state['completed']:
            path=Path(r['calibrated_path'])
            source=Path(r['source_path'])
            assert sha(path)==r['calibrated_sha256']
            assert sha(source)==r['source_sha256']
            page=pyvips.Image.tiffload(str(source),page=1,access='sequential').extract_band(0,n=3)
            h,w=r['unpadded_shape_yx']
            ph,pw=r['padded_shape_yx']
            expected=page.resize(w/page.width,vscale=h/page.height,kernel='lanczos3')
            actual=pyvips.Image.new_from_file(str(path),access='random')
            assert (expected.height,expected.width)==(h,w)
            assert (actual.height,actual.width,actual.bands,actual.format)==(ph,pw,3,'uchar')
            crops=[]
            for x,y in [(0,0),(max(0,w//2-128),max(0,h//2-128)),(w-256,h-256)]:
                got=pixels(actual,x,y,256,256)
                want=pixels(expected,x,y,256,256)
                mismatch=int(np.count_nonzero(got!=want))
                if mismatch:raise ValueError(f"{r['slide_id']} RGB crop mismatch at {x},{y}: {mismatch}")
                crops.append({'x':x,'y':y,'size':256,'different_channels':mismatch})
            pad_w,pad_h=pw-w,ph-h
            if pad_w:
                assert np.all(pixels(actual,w,0,min(64,pad_w),64)==255)
            if pad_h:
                assert np.all(pixels(actual,0,h,64,min(64,pad_h))==255)
            rows.append({'slide_id':r['slide_id'],'calibrated_sha256':r['calibrated_sha256'],
                         'source_sha256':r['source_sha256'],'shape_yx':[ph,pw],
                         'crops':crops,'right_pad_px':pad_w,'bottom_pad_px':pad_h,
                         'overview_sha256':sha(Path(r['overview_path']))})
        result={'format':'bora-cam-test-calibrated-audit-v1','status':'VERIFIED',
                'calibration_state_sha256':sha(STATE),'rows':rows,
                'sampled_rgb_channel_differences':0,'padding_white_verified':True,
                'expert_reference_opened':False,'online_publication':False,
                'completed_unix':time.time()}
        tmp=OUT.with_suffix('.json.partial')
        with tmp.open('x') as stream:
            json.dump(result,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,OUT)
        print(json.dumps({'status':'VERIFIED','slides':len(rows)}))
    except BaseException as exc:
        (HERE/'FAILED.json').write_text(json.dumps({'status':'FAILED_PRESERVE_EVIDENCE',
            'error':f'{type(exc).__name__}: {exc}','completed_rows':rows},indent=2)+'\n')
        raise


if __name__=='__main__':run()
