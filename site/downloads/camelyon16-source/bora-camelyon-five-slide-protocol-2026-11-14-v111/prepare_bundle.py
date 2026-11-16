"""Freeze one new, serial, source-only CAM binary network WSI bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
CTRL = ROOT / 'migration/bora-camelyon-five-slide-protocol-2026-11-14-v106/run_binary_bands.py'
SOURCE = ROOT / 'migration/bora-camelyon-five-slide-protocol-2026-11-14-v103/infer_binary_band.py'
INPUT = SOURCE.with_name('INPUTS_FROZEN.json')
PILOT = ROOT / 'migration/bora-camelyon-five-slide-protocol-2026-11-14-v108/BINARY_BAND_PILOT_AUDIT_VERIFIED.json'
GPU_UUID = 'GPU-59b2b884-a1c9-f7a3-9f35-e8380f4609d5'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--version',type=int,required=True)
    p.add_argument('--method',choices=('unet','deeplabv3plus'),required=True)
    p.add_argument('--slide',choices=('tumor_084','tumor_091','normal_019'),required=True)
    p.add_argument('--prior-audit',type=Path,required=True)
    args=p.parse_args()
    assert args.version>109
    prior=json.loads(args.prior_audit.read_text())
    assert prior['status']=='VERIFIED' and not prior['test_reference_opened']
    frozen=json.loads(INPUT.read_text())
    assert sha(SOURCE)==frozen['source_sha256']['infer_band']
    assert sha(PILOT) and json.loads(PILOT.read_text())['status']=='VERIFIED'
    assert socket.gethostname()=='icgeb-bioinformatics-unit'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==GPU_UUID
    contexts=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True).strip()
    assert not contexts,contexts
    services=subprocess.check_output(['systemctl','--user','list-units','bora-*','--state=running','--no-legend','--no-pager'],text=True).strip()
    assert not services,services
    used=int(subprocess.check_output('du -scb /mnt/sata_ssd/ChatGPT/Bora-Data/external-camelyon16* 2>/dev/null | tail -1',shell=True,text=True).split()[0])
    fs=os.statvfs(DATA)
    free=fs.f_bavail*fs.f_frsize
    bound=2*(1<<30)
    assert used+bound<200*(1<<30)
    assert free>100*(1<<30)+bound
    run=ROOT/f'migration/bora-camelyon-five-slide-protocol-2026-11-14-v{args.version}'
    run.mkdir(exist_ok=False)
    short={'tumor_084':'t084','tumor_091':'t091','normal_019':'n019'}[args.slide]
    service=f'bora-cam16-binary-{args.method}-{short}-v{args.version}.service'
    spec={'status':'BINARY_NETWORK_SOURCE_ONLY_FULL_BUNDLE_FROZEN',
          'method':args.method,'slide':args.slide,'service':service,
          'controller_sha256':sha(CTRL),'band_source_sha256':sha(SOURCE),
          'band_inputs_sha256':sha(INPUT),'pilot_audit_path':str(PILOT),
          'pilot_audit_sha256':sha(PILOT),
          'model_checkpoint_sha256':frozen['models'][args.method]['checkpoint_sha256'],
          'campaign_bytes_before':used,'new_output_bytes_bound':bound,
          'internal_deadline_seconds':3000,
          'test_reference_opened':False,'online_publication':False}
    (run/'FULL_RUN_FROZEN.json').write_text(json.dumps(spec,indent=2)+'\n')
    admission={'status':'VERIFIED','host':socket.gethostname(),'assigned_gpu_uuid':GPU_UUID,
               'running_bora_services':[],'gpu_contexts':[],'free_bytes_before':free,
               'campaign_bytes_before':used,'prior_audit_path':str(args.prior_audit),
               'prior_audit_sha256':sha(args.prior_audit),
               'frozen_sha256':sha(run/'FULL_RUN_FROZEN.json'),
               'test_reference_opened':False,'online_publication':False}
    (run/'ADMISSION_VERIFIED.json').write_text(json.dumps(admission,indent=2)+'\n')
    print(json.dumps({'run_dir':str(run),'service':service,'campaign_bytes':used,
                      'free_bytes':free,'frozen_sha256':admission['frozen_sha256']}))


if __name__=='__main__':
    main()
