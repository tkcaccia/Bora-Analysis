#!/usr/bin/env python3
"""Fail-latched, serial source-only H&E support for the frozen CAM test set."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).parent
RECOVERY=Path('/mnt/sata_ssd/ChatGPT/Bora/migration/bora-camelyon-five-slide-protocol-2026-11-14-v25/FAILED_AND_CLEANUP_VERIFIED.json')
CAL=Path('/mnt/sata_ssd/ChatGPT/Bora/migration/bora-camelyon-five-slide-protocol-2026-11-14-v24/CALIBRATION_STATE.json')
CAL_AUDIT=Path('/mnt/sata_ssd/ChatGPT/Bora/migration/bora-camelyon-five-slide-protocol-2026-11-14-v27/CALIBRATED_AUDIT_VERIFIED.json')
SCRIPT=HERE/'build_he_support.py'
STATE=HERE/'SUPPORT_STATE.json'
SLIDES=('tumor_084','tumor_091','normal_019')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def atomic(obj):
    tmp=STATE.with_suffix('.json.partial')
    with tmp.open('x') as stream:
        json.dump(obj,stream,indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp,STATE)


def main():
    if STATE.exists():raise FileExistsError(STATE)
    recovery=json.loads(RECOVERY.read_text())
    if recovery['status']!='FAILED_LATCHED_BEFORE_IMAGE_COMPUTE' or \
       recovery['owned_descendants_present'] is not False or \
       recovery['assigned_gpu_contexts_present'] is not False:
        raise RuntimeError('Prior failure/cleanup proof differs')
    cal=json.loads(CAL.read_text())
    calibrated_audit=json.loads(CAL_AUDIT.read_text())
    if cal['status']!='COMPLETE' or cal['expert_reference_opened'] or \
       calibrated_audit['status']!='VERIFIED' or \
       calibrated_audit['calibration_state_sha256']!=sha(CAL) or \
       [r['slide_id'] for r in cal['completed']]!=list(SLIDES):
        raise ValueError('Complete source-only calibrated test set required')
    state={'format':'bora-cam-source-only-support-controller-v1',
           'status':'RUNNING','calibration_sha256':sha(CAL),
           'calibration_audit_sha256':sha(CAL_AUDIT),
           'prior_failure_cleanup_sha256':sha(RECOVERY),
           'source_sha256':sha(SCRIPT),'pid':os.getpid(),
           'started_unix':time.time(),'completed':[],
           'expert_reference_opened':False,'online_publication':False}
    atomic(state)
    try:
        for slide in SLIDES:
            child=subprocess.run([sys.executable,str(SCRIPT),slide],
                                 capture_output=True,text=True,timeout=1200)
            if child.returncode:
                raise RuntimeError(f'{slide} exit {child.returncode}: {child.stderr[-2000:]}')
            path=Path('/mnt/sata_ssd/ChatGPT/Bora-Data/external-camelyon16-five-slide-2026-11-14-v32/he-support')/slide/'SUPPORT_PENDING_REVIEW.json'
            receipt=json.loads(path.read_text())
            if receipt['status']!='H_AND_E_ONLY_TISSUE_SUPPORT_PENDING_VISUAL_REVIEW' or \
               receipt['expert_annotation_used']:
                raise ValueError(f'{slide} support receipt differs')
            state['completed'].append({'slide_id':slide,'receipt_path':str(path),
                                       'receipt_sha256':sha(path),
                                       'fraction':receipt['fraction']})
            atomic(state)
            print(slide,receipt['fraction'],flush=True)
        state['status']='COMPLETE_PENDING_INDEPENDENT_AUDIT_AND_VISUAL_REVIEW'
        state['completed_unix']=time.time()
        atomic(state)
    except BaseException as exc:
        state['status']='FAILED_PRESERVE_PARTIAL'
        state['error']=f'{type(exc).__name__}: {exc}'
        state['failed_unix']=time.time()
        atomic(state)
        raise


if __name__=='__main__':main()
