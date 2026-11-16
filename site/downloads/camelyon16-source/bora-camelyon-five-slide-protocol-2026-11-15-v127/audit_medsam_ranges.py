"""Independent source-only MedSAM test range, hash and cleanup audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA=Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
PROMPTS=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-14-v120/MEDSAM_PROMPTS_FROZEN.json'
PILOT=DATA/'external-camelyon16-five-slide-2026-11-14-v123/medsam-pilot/tumor_091/range-0000-0001'
OUTPUT=DATA/'external-camelyon16-five-slide-2026-11-15-v125/medsam-ranges'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def main(run_dir):
    run_dir=Path(run_dir)
    spec=json.loads((run_dir/'FULL_RUN_FROZEN.json').read_text())
    state_path=run_dir/'CONTROLLER_STATE.json'
    state=json.loads(state_path.read_text())
    completed=json.loads((run_dir/'RANGES_COMPLETE.json').read_text())
    prompts=json.loads(PROMPTS.read_text())
    slide=spec['slide']
    row=next(x for x in prompts['slides'] if x['slide']==slide)
    assert state['status']=='COMPLETE' and state['active'] is None
    assert state['frozen_sha256']==sha(run_dir/'FULL_RUN_FROZEN.json')
    assert completed['controller_state_sha256']==sha(state_path)
    assert completed['prompted_tiles']==row['prompted_tiles']==spec['prompted_tiles']
    assert sha(PROMPTS)==spec['prompts_sha256']
    assert state['latest_cleanup']['status']=='VERIFIED'
    assert not state['test_reference_opened'] and not state['online_publication']
    active=subprocess.check_output(['systemctl','--user','show',spec['service'],'-p','ActiveState','--value'],text=True).strip()
    assert active!='active'
    assert not subprocess.check_output(['systemctl','--user','list-units','bora-*','--state=running','--no-legend','--no-pager'],text=True).strip()
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True).strip()
    seen=[]
    pixels=0
    for part in state['completed']:
        start,stop=part['start'],part['stop']
        folder=PILOT if part.get('reused_audited_pilot') else OUTPUT/slide/f'range-{start:04d}-{stop:04d}'
        receipt_path=folder/'RANGE_COMPLETE.json'
        assert sha(receipt_path)==part['receipt_sha256']
        receipt=json.loads(receipt_path.read_text())
        assert (receipt['slide'],receipt['start'],receipt['stop'])==(slide,start,stop)
        assert receipt['status']=='CAM_TEST_MEDSAM_TILE_RANGE_COMPLETE'
        for i in range(start,stop):
            tile=row['tiles'][i]
            path=folder/f'tile-{i:04d}/TILE_COMPLETE.json'
            r=json.loads(path.read_text())
            assert (r['slide'],r['prompt_tile_index'])==(slide,i)
            assert r['status']=='CAM_TEST_MEDSAM_PROMPTED_TILE_COMPLETE_SOURCE_ONLY'
            assert r['origin_yx_4um']==tile['origin_yx_4um']
            assert r['shape_yx_4um']==tile['shape_yx_4um']
            assert r['boxes']==tile['boxes']
            assert r['checkpoint_sha256']==prompts['model_checkpoint_sha256']
            assert r['prompt_manifest_sha256']==sha(PROMPTS)
            assert sha(r['mask_path'])==r['mask_sha256']
            array=np.load(r['mask_path'],allow_pickle=False)
            assert array.shape==tuple(tile['shape_yx_4um']) and array.dtype==np.uint8
            assert set(np.unique(array).tolist()).issubset({0,1})
            assert int(array.sum())==r['metastasis_pixels_4um']
            assert not r['test_reference_opened']
            pixels+=int(array.sum())
            seen.append(i)
    assert seen==list(range(row['prompted_tiles']))
    record={'status':'VERIFIED','slide':slide,'prompted_tiles':len(seen),
            'prompt_boxes':row['prompt_boxes'],'tile_metastasis_pixels_sum':pixels,
            'controller_state_sha256':sha(state_path),
            'prompt_manifest_sha256':sha(PROMPTS),
            'all_source_only_tile_hashes_and_shapes_verified':True,
            'terminal_service_state':active,'terminal_gpu_contexts':[],
            'test_reference_opened':False,'online_publication':False}
    path=run_dir/'MEDSAM_RANGES_AUDIT_VERIFIED.json'
    path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'VERIFIED','slide':slide,'tiles':len(seen),'audit_sha256':sha(path)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True)
    a=p.parse_args();main(a.run_dir)
