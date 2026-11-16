#!/usr/bin/env python3
"""Download exactly the three frozen CAM test RGB sources to Chiamaka locally."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
FROZEN = ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-14-v19/TEST_IDS_FROZEN.json'
GATE = ROOT/'migration/bora-r-camelyon-2026-11-14-v2/LOCAL_CC0_USE_GATE.json'
OUT = DATA/'external-camelyon16-five-slide-2026-11-14-v20/sources'
STATE = Path(__file__).with_name('DOWNLOAD_STATE.json')
CAP_BYTES = 200*1024**3
RESERVE_BYTES = 100*1024**3


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def hashes(path):
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            md5.update(block);sha256.update(block)
    return md5.hexdigest(),sha256.hexdigest()


def atomic_state(state):
    temporary = STATE.with_name(STATE.name+'.partial')
    temporary.write_text(json.dumps(state,indent=2)+'\n')
    os.replace(temporary,STATE)


def used_cam_bytes():
    paths = list(DATA.glob('external-camelyon16-*'))
    if not paths:
        return 0
    output = subprocess.check_output(['du','-scB1',*(str(x) for x in paths)],text=True)
    return int(output.splitlines()[-1].split()[0])


def main():
    if STATE.exists() or OUT.exists():
        raise FileExistsError('Preserve existing attempt; use a new version')
    gate = json.loads(GATE.read_text())
    if gate['status'] != 'LOCAL_ANALYSIS_AUTHORIZED_PUBLICATION_HELD' or \
       not gate['local_three_slide_scoring_allowed'] or \
       gate['new_cam_online_publication_allowed']:
        raise PermissionError('Local-only CAM use gate differs')
    frozen = json.loads(FROZEN.read_text())
    if frozen['status'] != 'TEST_IDS_FROZEN_SOURCE_SHA_AND_PATIENT_CHECK_PENDING' or \
       frozen['test_ids'] != ['tumor_084','tumor_091','normal_019']:
        raise RuntimeError('Frozen CAM test IDs differ')
    selected = frozen['selected_official_metadata'][2:]
    assert [x['slide_id'] for x in selected] == frozen['test_ids']
    required = sum(x['source_object_bytes'] for x in selected)
    current = used_cam_bytes()
    free = shutil.disk_usage(DATA).free
    if current+required > CAP_BYTES or free-required < RESERVE_BYTES:
        raise OSError('CAM download would breach declared cap or reserve')
    OUT.mkdir(parents=True)
    state = {'format':'bora-cam-local-selected-source-download-v1',
             'status':'RUNNING','started_unix':time.time(),
             'frozen_ids_sha256':sha(FROZEN),'local_gate_sha256':sha(GATE),
             'host':os.uname().nodename,'pid':os.getpid(),
             'campaign_bytes_before':current,'free_bytes_before':free,
             'projected_source_bytes':required,'campaign_cap_bytes':CAP_BYTES,
             'reserve_bytes':RESERVE_BYTES,'completed':[],
             'no_test_reference_opened':True,'online_publication':False}
    atomic_state(state)
    try:
        for item in selected:
            slide = item['slide_id']
            part = OUT/f'{slide}.tif.part'
            target = OUT/f'{slide}.tif'
            if part.exists() or target.exists():
                raise FileExistsError(f'Preserve previous source bytes for {slide}')
            source = f's3://camelyon-dataset/CAMELYON16/images/{slide}.tif'
            result = subprocess.run(['aws','s3','cp','--no-sign-request',source,str(part)],
                                    capture_output=True,text=True,timeout=7200)
            if result.returncode:
                raise RuntimeError(f'{slide}: aws s3 cp exit {result.returncode}: {result.stderr[-1200:]}')
            if part.stat().st_size != item['source_object_bytes']:
                raise RuntimeError(f'{slide}: downloaded source size differs')
            md5,sha256 = hashes(part)
            if md5 != item['source_object_md5']:
                raise RuntimeError(f'{slide}: official source MD5 differs')
            os.replace(part,target)
            state['completed'].append({'slide_id':slide,'source_path':str(target),
                'source_bytes':target.stat().st_size,'official_md5':md5,
                'local_sha256':sha256,'source_url':item['source_url']})
            atomic_state(state)
            print(f'{slide} source verified',flush=True)
        state['status']='COMPLETE';state['completed_unix']=time.time()
        state['campaign_bytes_after']=used_cam_bytes()
        state['free_bytes_after']=shutil.disk_usage(DATA).free
        if state['campaign_bytes_after'] > CAP_BYTES or state['free_bytes_after'] < RESERVE_BYTES:
            raise OSError('Post-download cap or reserve breach')
        atomic_state(state)
    except BaseException as error:
        state['status']='FAILED_PRESERVE_PARTIAL'
        state['error']=repr(error);state['failed_unix']=time.time()
        atomic_state(state)
        raise


if __name__=='__main__':
    main()
