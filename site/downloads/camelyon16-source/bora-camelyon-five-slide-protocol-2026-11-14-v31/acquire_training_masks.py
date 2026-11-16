#!/usr/bin/env python3
"""Verify exact official binary masks for the two CAM training slides only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
FROZEN=ROOT/'bora-camelyon-five-slide-protocol-2026-11-14-v19/TEST_IDS_FROZEN.json'
GATE=ROOT/'bora-r-camelyon-2026-11-14-v2/LOCAL_CC0_USE_GATE.json'
OUT=Path('/mnt/sata_ssd/ChatGPT/Bora-Data/external-camelyon16-five-slide-2026-11-14-v31/training-reference')
STATE=Path(__file__).parent/'TRAINING_REFS_STATE.json'
TUMOR_EXISTING=Path('/mnt/sata_ssd/ChatGPT/Bora-Data/raw/camelyon16-external-2026-11-12/tumor_093_mask.tif')


def hashes(path):
    m=hashlib.md5();s=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            m.update(block);s.update(block)
    return m.hexdigest(),s.hexdigest()


def atomic(value):
    temp=STATE.with_suffix('.json.partial')
    with temp.open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,STATE)


def main():
    if STATE.exists() or OUT.exists():raise FileExistsError('Preserve old training-mask attempt')
    gate=json.loads(GATE.read_text())
    frozen=json.loads(FROZEN.read_text())
    if gate['status']!='LOCAL_ANALYSIS_AUTHORIZED_PUBLICATION_HELD' or \
       not gate['local_cam_specific_training_allowed'] or \
       gate['new_cam_online_publication_allowed'] or \
       frozen['training_ids']!=['tumor_093','normal_093'] or \
       frozen['test_ids']!=['tumor_084','tumor_091','normal_019']:
        raise PermissionError('Frozen local-only CAM training scope differs')
    meta={r['slide_id']:r for r in frozen['selected_official_metadata']}
    OUT.mkdir(parents=True)
    state={'format':'bora-cam-training-reference-source-v1','status':'RUNNING',
           'frozen_ids_sha256':hashes(FROZEN)[1],
           'local_gate_sha256':hashes(GATE)[1],
           'started_unix':time.time(),'pid':os.getpid(),'completed':[],
           'reference_pixels_opened':False,'test_reference_downloaded':False,
           'online_publication':False}
    atomic(state)
    try:
        tumor=meta['tumor_093']
        if TUMOR_EXISTING.stat().st_size!=tumor['mask_object_bytes']:
            raise ValueError('Existing tumor mask size differs')
        md5,sha=hashes(TUMOR_EXISTING)
        if md5!=tumor['mask_object_md5']:
            raise ValueError('Existing tumor mask differs from official MD5')
        state['completed'].append({'slide_id':'tumor_093','mask_path':str(TUMOR_EXISTING),
                                   'bytes':TUMOR_EXISTING.stat().st_size,
                                   'official_md5':md5,'sha256':sha})
        atomic(state)
        normal=meta['normal_093']
        partial=OUT/'normal_093_mask.tif.part'
        final=OUT/'normal_093_mask.tif'
        result=subprocess.run(['aws','s3','cp','--no-sign-request',
            's3://camelyon-dataset/CAMELYON16/masks/normal_093_mask.tif',str(partial)],
            capture_output=True,text=True,timeout=3600)
        if result.returncode:
            raise RuntimeError(f'AWS mask transfer exit {result.returncode}: {result.stderr[-1200:]}')
        if partial.stat().st_size!=normal['mask_object_bytes']:
            raise ValueError('Normal mask source size differs')
        md5,sha=hashes(partial)
        if md5!=normal['mask_object_md5']:
            raise ValueError('Normal mask differs from official MD5')
        os.replace(partial,final)
        state['completed'].append({'slide_id':'normal_093','mask_path':str(final),
                                   'bytes':final.stat().st_size,'official_md5':md5,'sha256':sha})
        state['status']='COMPLETE_SOURCE_BYTES_VERIFIED_NOT_OPENED'
        state['completed_unix']=time.time()
        atomic(state)
        print('Two CAM training masks verified against official MD5',flush=True)
    except BaseException as exc:
        state['status']='FAILED_PRESERVE_PARTIAL'
        state['error']=f'{type(exc).__name__}: {exc}'
        state['failed_unix']=time.time()
        atomic(state)
        raise


if __name__=='__main__':main()
