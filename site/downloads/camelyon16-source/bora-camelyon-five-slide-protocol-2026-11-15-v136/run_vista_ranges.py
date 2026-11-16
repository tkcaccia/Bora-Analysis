"""Serial fail-fast Chiamaka VISTA-PATH source-only test-slide ranges."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA=Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE=Path(__file__).resolve().parent
LIFE=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-15-v125/run_medsam_ranges.py'
sys.path.insert(0,str(LIFE.parent))
import run_medsam_ranges as life
SOURCE=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-15-v132/vista_test_tiles.py'
FREEZE=SOURCE.with_name('VISTA_SOURCE_FROZEN.json')
SCOPE=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-15-v133/VISTA_SCOPE_AUDIT_VERIFIED.json'
PILOT=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-15-v135/VISTA_PILOT_AUDIT_VERIFIED.json'
PYTHON=Path('/home/chiamaka/miniconda3/envs/medsam/bin/python')
OUTPUT=DATA/'external-camelyon16-five-slide-2026-11-15-v136/vista-ranges'
GPU_UUID='GPU-59b2b884-a1c9-f7a3-9f35-e8380f4609d5'
EXTRA=DATA/'comparators-2026-11-07-v1/deps-vista'
CHECKOUT=DATA/'comparators-2026-11-07-v1/sources/VISTA-PATH'


def sha(path):return life.sha(path)
def atomic(path,value):return life.atomic(path,value)


def verify_range(folder,slide,start,stop,source_hash,freeze_hash,ordered_names):
    path=folder/'RANGE_COMPLETE.json'
    r=json.loads(path.read_text())
    assert r['status']=='CAM_VISTA_TILE_RANGE_COMPLETE'
    assert (r['slide'],r['start'],r['stop'])==(slide,start,stop)
    assert r['source_sha256']==source_hash and r['freeze_sha256']==freeze_hash
    assert not r['expert_reference_opened']
    for i in range(start,stop):
        receipt=json.loads((folder/f'tile-{i:04d}/TILE_COMPLETE.json').read_text())
        assert receipt['status']=='CAM_VISTA_SOURCE_ONLY_NATIVE_TILE_COMPLETE'
        assert (receipt['slide'],receipt['selected_tile_index'])==(slide,i)
        assert receipt['source_sha256']==source_hash
        assert receipt['prompt_names']==ordered_names
        assert sha(receipt['probabilities_path'])==receipt['probabilities_sha256']
    return sha(path)


def main(run_dir):
    run_dir=Path(run_dir)
    full=run_dir/'FULL_RUN_FROZEN.json'
    spec=json.loads(full.read_text())
    service,slide=spec['service'],spec['slide']
    state_path=run_dir/'CONTROLLER_STATE.json'
    if state_path.exists():raise FileExistsError('No in-place VISTA GPU bundle retry')
    if (sha(__file__)!=spec['controller_sha256'] or sha(LIFE)!=spec['lifecycle_source_sha256'] or
        sha(SOURCE)!=spec['source_sha256'] or sha(FREEZE)!=spec['source_freeze_sha256'] or
        sha(SCOPE)!=spec['scope_audit_sha256'] or sha(PILOT)!=spec['pilot_audit_sha256']):
        raise RuntimeError('Frozen source, audit or model identity differs')
    if socket.gethostname()!='icgeb-bioinformatics-unit' or life.command(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'])!=GPU_UUID:
        raise RuntimeError('Assigned Chiamaka GPU differs')
    own_cgroup=life.cgroup(service)
    if life.other_bora(service):raise RuntimeError('Other Bora service at admission')
    gpu=life.classify_contexts(life.contexts(),own_cgroup)
    if gpu['owned'] or gpu['unknown']:raise RuntimeError('Owned/unknown GPU context at admission')
    fs=os.statvfs(DATA);free=fs.f_bavail*fs.f_frsize
    if free<100*(1<<30)+spec['output_bound_bytes'] or spec['campaign_bytes_before']+spec['output_bound_bytes']>200*(1<<30):
        raise OSError('VISTA storage admission failed')
    freeze=json.loads(FREEZE.read_text())
    row=freeze['slides'][slide]
    total=len(row['selected_tiles'])
    assert total==spec['selected_tiles'] and total>0
    assert sha(freeze['checkpoint_path'])==freeze['checkpoint_sha256']
    (run_dir/'logs').mkdir(exist_ok=False)
    state={'status':'RUNNING','service':service,'slide':slide,
           'start_utc':life.now(),'frozen_sha256':sha(full),
           'assigned_gpu_uuid':GPU_UUID,'free_bytes_at_admission':free,
           'completed':[],'active':None,'latest_cleanup':None,
           'test_reference_opened':False,'online_publication':False}
    atomic(state_path,state)
    deadline=time.monotonic()+spec['internal_deadline_seconds']
    try:
        for start in range(0,total,100):
            stop=min(total,start+100)
            if life.other_bora(service):raise RuntimeError('Other Bora service during bundle')
            gpu=life.classify_contexts(life.contexts(),own_cgroup)
            if gpu['owned'] or gpu['unknown']:raise RuntimeError('Owned/unknown GPU context before range')
            if time.monotonic()+720>deadline:raise TimeoutError('Internal deadline before next range')
            folder=OUTPUT/slide/f'range-{start:04d}-{stop:04d}'
            if folder.exists():raise FileExistsError('No in-place range retry')
            log=run_dir/'logs'/f'range-{start:04d}-{stop:04d}.log'
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
                 'PYTHONPATH':str(EXTRA)+':'+str(CHECKOUT),
                 'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
            with log.open('x') as stream:
                child=subprocess.Popen([str(PYTHON),str(SOURCE),'--slide',slide,
                    '--start',str(start),'--stop',str(stop),'--output',str(folder)],
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=env)
                state['active']={'start':start,'stop':stop,'pid':child.pid,'pgid':child.pid,'log':str(log)}
                atomic(state_path,state)
                try:code=child.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=15)
                    code=None
            proof=life.cleanup(child.pid,own_cgroup)
            state['latest_cleanup']=proof
            atomic(state_path,state)
            if proof['status']!='VERIFIED':raise RuntimeError('Owned process/GPU cleanup uncertain')
            if code is None:raise TimeoutError('VISTA range timed out')
            if code:raise RuntimeError(f'VISTA range exited {code}: {log}')
            receipt_hash=verify_range(folder,slide,start,stop,sha(SOURCE),sha(FREEZE),freeze['ordered_class_names'])
            output_bytes=sum(p.stat().st_size for p in (OUTPUT/slide).rglob('*') if p.is_file())
            if output_bytes>spec['output_bound_bytes'] or os.statvfs(DATA).f_bavail*os.statvfs(DATA).f_frsize<100*(1<<30):
                raise OSError('VISTA output bound or free reserve exceeded')
            state['completed'].append({'start':start,'stop':stop,'receipt_sha256':receipt_hash})
            state['active']=None
            atomic(state_path,state)
        gpu=life.classify_contexts(life.contexts(),own_cgroup)
        if life.other_bora(service) or gpu['owned'] or gpu['unknown']:
            raise RuntimeError('Unexpected terminal Bora/GPU activity')
        state.update(status='COMPLETE',end_utc=life.now())
        atomic(state_path,state)
        atomic(run_dir/'RANGES_COMPLETE.json',{'status':'CAM_TEST_VISTA_SLIDE_RANGES_COMPLETE',
            'slide':slide,'selected_tiles':total,'controller_state_sha256':sha(state_path),
            'test_reference_opened':False,'online_publication':False})
    except BaseException as error:
        active=state.get('active')
        if active and (state.get('latest_cleanup') or {}).get('status')!='VERIFIED':
            state['latest_cleanup']=life.cleanup(active['pgid'],own_cgroup)
        state.update(status='FAILED_LATCHED_PRESERVE_PARTIAL',error=repr(error),end_utc=life.now())
        atomic(state_path,state)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True)
    a=p.parse_args();main(a.run_dir)
