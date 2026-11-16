"""Fail-fast local-only serial CAM test scoring after the pre-reference freeze."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE = Path(__file__).resolve().parent
SCRIPT = ROOT / 'migration/bora-camelyon-five-slide-protocol-2026-11-15-v144/score_test_case.py'
FREEZE = SCRIPT.with_name('SCORING_FROZEN.json')
READY = ROOT / 'migration/bora-camelyon-five-slide-protocol-2026-11-15-v148/SCORING_READINESS_VERIFIED.json'
REFS = ROOT / 'migration/bora-camelyon-five-slide-protocol-2026-11-15-v150/ALIGNMENT_AUDIT_VERIFIED.json'
OUT = DATA / 'external-camelyon16-five-slide-2026-11-15-v144/test-scores'
PYTHON = Path('/mnt/sata_ssd/ChatGPT/Bora-Data/experiments/virchow2-inverse-2026-10-16/.venv/bin/python')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(temp,path)


def verify_child_gone(pgid):
    for _ in range(10):
        try:
            os.killpg(pgid,0)
        except ProcessLookupError:
            return
        time.sleep(.5)
    raise RuntimeError('CPU score child process group cleanup uncertain')


def main():
    frozen_path=HERE/'CONTROLLER_FROZEN.json'; frozen=json.loads(frozen_path.read_text())
    assert sha(__file__)==frozen['controller_source_sha256']
    assert sha(SCRIPT)==frozen['scorer_source_sha256']
    assert sha(FREEZE)==frozen['scoring_freeze_sha256']
    assert sha(READY)==frozen['pre_reference_readiness_sha256']
    assert sha(REFS)==frozen['reference_alignment_audit_sha256']
    assert not frozen['online_publication']
    statepath=HERE/'CONTROLLER_STATE.json'
    if statepath.exists():raise FileExistsError('No in-place scorer controller retry')
    running=subprocess.check_output(['systemctl','--user','list-units','bora-*','--state=running','--no-legend','--no-pager'],text=True)
    others=[line for line in running.splitlines() if frozen['service'] not in line]
    if others:raise RuntimeError('Other Bora service at CPU admission')
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True).strip()
    state={'status':'RUNNING','service':frozen['service'],'completed':[],
           'active':None,'frozen_sha256':sha(frozen_path),'online_publication':False}
    atomic(statepath,state)
    (HERE/'logs').mkdir(exist_ok=False)
    try:
        for item in frozen['order']:
            slide,method=item['slide'],item['method']
            receipt=OUT/method/slide/'SCORE_COMPLETE.json'
            if receipt.exists():
                if not item.get('already_complete_sha256') or sha(receipt)!=item['already_complete_sha256']:
                    raise FileExistsError('Unexpected existing score: '+str(receipt))
            else:
                log=HERE/'logs'/f'{slide}-{method}.log'
                with log.open('x') as stream:
                    child=subprocess.Popen([str(PYTHON),str(SCRIPT),'--slide',slide,'--method',method],
                                           stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    state['active']={'slide':slide,'method':method,'pid':child.pid,'pgid':child.pid,'log':str(log)}
                    atomic(statepath,state)
                    try:code=child.wait(timeout=300)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGKILL)
                            child.wait(timeout=10)
                        raise TimeoutError('CAM score child exceeded 300s; bundle failed')
                if code:raise RuntimeError(f'CAM score child failed {code}: {log}')
                verify_child_gone(child.pid)
                if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True).strip():
                    raise RuntimeError('Unexpected GPU context after CPU scoring child')
            assert receipt.exists()
            result=json.loads(receipt.read_text())
            assert result['status']=='COMPLETE_LOCAL_CC0_NO_PUBLICATION'
            assert (result['slide'],result['method'])==(slide,method)
            assert result['scoring_freeze_sha256']==sha(FREEZE)
            state['completed'].append({'slide':slide,'method':method,'receipt_sha256':sha(receipt)})
            state['active']=None
            atomic(statepath,state)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True).strip()
        state.update(status='COMPLETE',end_utc=time.time())
        atomic(statepath,state)
        atomic(HERE/'SCORES_COMPLETE.json',{'status':'THREE_TEST_SLIDES_FIVE_METHODS_SCORED_LOCAL_ONLY',
            'completed':len(state['completed']),'state_sha256':sha(statepath),'online_publication':False})
    except BaseException as error:
        state.update(status='FAILED_LATCHED_PRESERVE_PARTIAL',error=repr(error),end_utc=time.time())
        atomic(statepath,state)
        raise


if __name__=='__main__':main()
