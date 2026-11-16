"""One serial fail-fast Chiamaka GPU bundle for one CAM test MedSAM slide."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA=Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
SOURCE=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-14-v122/infer_medsam_test.py'
FREEZE=SOURCE.with_name('MEDSAM_SOURCE_FROZEN.json')
PROMPTS=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-14-v120/MEDSAM_PROMPTS_FROZEN.json'
PILOT_AUDIT=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-14-v124/MEDSAM_PILOT_AUDIT_VERIFIED.json'
PYTHON=DATA/'experiments/virchow2-inverse-2026-10-16/.venv/bin/python'
OUTPUT=DATA/'external-camelyon16-five-slide-2026-11-15-v125/medsam-ranges'
GPU_UUID='GPU-59b2b884-a1c9-f7a3-9f35-e8380f4609d5'


def now():return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def atomic(path,value):
    partial=path.with_name(path.name+'.partial')
    partial.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(partial,path)


def command(args):
    return subprocess.run(args,capture_output=True,text=True,check=True,timeout=10).stdout.strip()


def contexts():
    rows=command(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'])
    return [line.strip() for line in rows.splitlines() if line.strip()]


def other_bora(service):
    rows=command(['systemctl','--user','list-units','bora-*','--state=running','--no-legend','--no-pager'])
    return [line.split()[0] for line in rows.splitlines() if line.strip() and line.split()[0]!=service]


def cgroup(service):
    raw=command(['systemctl','--user','show',service,'-p','ControlGroup'])
    path=raw.partition('=')[2]
    if not path:
        raise RuntimeError('Service cgroup unavailable')
    return Path('/sys/fs/cgroup'+path)


def owned_pids(group):
    result=[]
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields=path.read_text().split(') ',1)[1].split()
            if int(fields[2])==group:
                result.append(int(path.parent.name))
        except (OSError,ValueError,IndexError):
            pass
    return result


def classify_contexts(rows,own_cgroup):
    owned,unrelated,unknown=[],[],[]
    group=str(own_cgroup).removeprefix('/sys/fs/cgroup')
    for row in rows:
        pid=int(row.split(',')[0].strip())
        try:
            path=Path('/proc')/str(pid)
            cg=(path/'cgroup').read_text()
            comm=(path/'comm').read_text().strip().lower()
        except OSError:
            unknown.append(row);continue
        if group in cg:owned.append(row)
        elif 'rustdesk' in comm:unrelated.append(row)
        else:unknown.append(row)
    return {'owned':owned,'unrelated_ui':unrelated,'unknown':unknown}


def cleanup(group,own_cgroup):
    observations=[]
    for _ in range(11):
        row={'utc':now(),'owned_process_group_pids':owned_pids(group),
             'owned_cgroup_pids':sorted(set(int(p) for p in (own_cgroup/'cgroup.procs').read_text().split())-{os.getpid()}),
             'gpu':classify_contexts(contexts(),own_cgroup)}
        observations.append(row)
        if not row['owned_process_group_pids'] and not row['owned_cgroup_pids'] and not row['gpu']['owned'] and not row['gpu']['unknown']:
            return {'status':'VERIFIED','observations':observations}
        time.sleep(3)
    return {'status':'UNKNOWN_OR_FAILED','observations':observations}


def verify_range(path,slide,start,stop,source_freeze_sha):
    receipt_path=path/'RANGE_COMPLETE.json'
    r=json.loads(receipt_path.read_text())
    assert r['status']=='CAM_TEST_MEDSAM_TILE_RANGE_COMPLETE'
    assert (r['slide'],r['start'],r['stop'])==(slide,start,stop)
    assert r['source_sha256']==sha(SOURCE) and r['freeze_sha256']==source_freeze_sha
    assert not r['test_reference_opened']
    for index in range(start,stop):
        file=path/f'tile-{index:04d}/TILE_COMPLETE.json'
        tile=json.loads(file.read_text())
        assert tile['status']=='CAM_TEST_MEDSAM_PROMPTED_TILE_COMPLETE_SOURCE_ONLY'
        assert (tile['slide'],tile['prompt_tile_index'])==(slide,index)
        assert sha(tile['mask_path'])==tile['mask_sha256']
    return sha(receipt_path)


def main(run_dir):
    run_dir=Path(run_dir)
    full=run_dir/'FULL_RUN_FROZEN.json'
    spec=json.loads(full.read_text())
    slide,service=spec['slide'],spec['service']
    state_path=run_dir/'CONTROLLER_STATE.json'
    if state_path.exists():raise FileExistsError('No in-place GPU bundle retry')
    if (sha(__file__)!=spec['controller_sha256'] or sha(SOURCE)!=spec['source_sha256'] or
        sha(FREEZE)!=spec['source_freeze_sha256'] or sha(PROMPTS)!=spec['prompts_sha256'] or
        sha(PILOT_AUDIT)!=spec['pilot_audit_sha256']):
        raise RuntimeError('Frozen source, prompts or pilot audit changed')
    if socket.gethostname()!='icgeb-bioinformatics-unit' or command(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'])!=GPU_UUID:
        raise RuntimeError('Chiamaka assigned GPU differs')
    own_cgroup=cgroup(service)
    if other_bora(service):raise RuntimeError('Other Bora service active')
    gpu=classify_contexts(contexts(),own_cgroup)
    if gpu['owned'] or gpu['unknown']:raise RuntimeError('Owned/unknown GPU context at admission')
    fs=os.statvfs(DATA);free=fs.f_bavail*fs.f_frsize
    if free<100*(1<<30)+spec['output_bound_bytes'] or spec['campaign_bytes_before']+spec['output_bound_bytes']>200*(1<<30):
        raise OSError('Storage admission failed')
    manifest=json.loads(PROMPTS.read_text())
    row=next(s for s in manifest['slides'] if s['slide']==slide)
    total=row['prompted_tiles']
    if total!=spec['prompted_tiles']:raise RuntimeError('Prompt count differs')
    (run_dir/'logs').mkdir(exist_ok=False)
    state={'status':'RUNNING','start_utc':now(),'service':service,'slide':slide,
           'assigned_gpu_uuid':GPU_UUID,'frozen_sha256':sha(full),
           'free_bytes_at_admission':free,'completed':[],'active':None,
           'latest_cleanup':None,'test_reference_opened':False,'online_publication':False}
    atomic(state_path,state)
    deadline=time.monotonic()+spec['internal_deadline_seconds']
    try:
        next_index=0
        if slide=='tumor_091':
            pilot=Path(spec['reused_pilot_path'])
            assert sha(pilot/'tile-0000/TILE_COMPLETE.json')==json.loads(PILOT_AUDIT.read_text())['pilot_receipt_sha256']
            state['completed'].append({'start':0,'stop':1,'reused_audited_pilot':True,
                                       'receipt_sha256':sha(pilot/'RANGE_COMPLETE.json')})
            next_index=1
            atomic(state_path,state)
        while next_index<total:
            if other_bora(service):raise RuntimeError('Other Bora service during bundle')
            gpu=classify_contexts(contexts(),own_cgroup)
            if gpu['owned'] or gpu['unknown']:raise RuntimeError('Owned/unknown GPU context before next range')
            if time.monotonic()+360>deadline:raise TimeoutError('Internal deadline before next range')
            stop=min(total,next_index+50)
            folder=OUTPUT/slide/f'range-{next_index:04d}-{stop:04d}'
            if folder.exists():raise FileExistsError('Existing range cannot be retried in place')
            log=run_dir/'logs'/f'range-{next_index:04d}-{stop:04d}.log'
            with log.open('x') as stream:
                child=subprocess.Popen([str(PYTHON),str(SOURCE),'--slide',slide,
                    '--start',str(next_index),'--stop',str(stop),'--output',str(folder)],
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
                state['active']={'start':next_index,'stop':stop,'pid':child.pid,'pgid':child.pid,'log':str(log)}
                atomic(state_path,state)
                try:code=child.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=15)
                    code=None
            proof=cleanup(child.pid,own_cgroup)
            state['latest_cleanup']=proof
            atomic(state_path,state)
            if proof['status']!='VERIFIED':raise RuntimeError('Owned process/GPU cleanup uncertain')
            if code is None:raise TimeoutError('MedSAM range timed out')
            if code:raise RuntimeError(f'MedSAM range exited {code}: {log}')
            receipt_hash=verify_range(folder,slide,next_index,stop,sha(FREEZE))
            output_bytes=sum(p.stat().st_size for p in (OUTPUT/slide).rglob('*') if p.is_file())
            if output_bytes>spec['output_bound_bytes']:
                raise OSError('Frozen MedSAM output bound exceeded')
            if os.statvfs(DATA).f_bavail*os.statvfs(DATA).f_frsize<100*(1<<30):
                raise OSError('100-GiB reserve lost during MedSAM bundle')
            state['completed'].append({'start':next_index,'stop':stop,'receipt_sha256':receipt_hash})
            state['active']=None
            atomic(state_path,state)
            next_index=stop
        terminal_gpu=classify_contexts(contexts(),own_cgroup)
        if other_bora(service) or terminal_gpu['owned'] or terminal_gpu['unknown']:
            raise RuntimeError('Unexpected terminal Bora/GPU activity')
        state.update(status='COMPLETE',end_utc=now())
        atomic(state_path,state)
        atomic(run_dir/'RANGES_COMPLETE.json',{'status':'CAM_TEST_MEDSAM_SLIDE_RANGES_COMPLETE',
            'slide':slide,'prompted_tiles':total,'controller_state_sha256':sha(state_path),
            'test_reference_opened':False,'online_publication':False})
    except BaseException as error:
        active=state.get('active')
        if active and (state.get('latest_cleanup') or {}).get('status')!='VERIFIED':
            state['latest_cleanup']=cleanup(active['pgid'],own_cgroup)
        state.update(status='FAILED_LATCHED_PRESERVE_PARTIAL',error=repr(error),end_utc=now())
        atomic(state_path,state)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True)
    args=p.parse_args();main(args.run_dir)
