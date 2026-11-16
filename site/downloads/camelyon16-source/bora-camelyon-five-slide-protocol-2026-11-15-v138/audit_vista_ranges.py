"""Independent source-only VISTA test tile hash, shape, and lifecycle audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA=Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
FROZEN=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-15-v132/VISTA_SOURCE_FROZEN.json'
SOURCE=FROZEN.with_name('vista_test_tiles.py')
TILES=DATA/'external-camelyon16-five-slide-2026-11-15-v136/vista-ranges'


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
    freeze=json.loads(FROZEN.read_text())
    slide=spec['slide']
    row=freeze['slides'][slide]
    assert state['status']=='COMPLETE' and state['active'] is None
    assert state['frozen_sha256']==sha(run_dir/'FULL_RUN_FROZEN.json')
    assert completed['controller_state_sha256']==sha(state_path)
    assert completed['selected_tiles']==len(row['selected_tiles'])==spec['selected_tiles']
    assert sha(FROZEN)==spec['source_freeze_sha256'] and sha(SOURCE)==spec['source_sha256']
    assert state['latest_cleanup']['status']=='VERIFIED'
    assert not state['test_reference_opened'] and not state['online_publication']
    active=subprocess.check_output(['systemctl','--user','show',spec['service'],'-p','ActiveState','--value'],text=True).strip()
    assert active!='active'
    assert not subprocess.check_output(['systemctl','--user','list-units','bora-*','--state=running','--no-legend','--no-pager'],text=True).strip()
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True).strip()
    seen=[]
    minimum,maximum=1.,0.
    for part in state['completed']:
        start,stop=part['start'],part['stop']
        folder=TILES/slide/f'range-{start:04d}-{stop:04d}'
        path=folder/'RANGE_COMPLETE.json'
        assert sha(path)==part['receipt_sha256']
        r=json.loads(path.read_text())
        assert r['status']=='CAM_VISTA_TILE_RANGE_COMPLETE'
        assert (r['slide'],r['start'],r['stop'])==(slide,start,stop)
        assert r['source_sha256']==sha(SOURCE) and r['freeze_sha256']==sha(FROZEN)
        assert not r['expert_reference_opened']
        for i in range(start,stop):
            tile=row['selected_tiles'][i]
            receipt=json.loads((folder/f'tile-{i:04d}/TILE_COMPLETE.json').read_text())
            assert receipt['status']=='CAM_VISTA_SOURCE_ONLY_NATIVE_TILE_COMPLETE'
            assert (receipt['slide'],receipt['selected_tile_index'])==(slide,i)
            assert receipt['origin_yx_native']==tile['origin_yx_native']
            assert receipt['origin_yx_4um']==[v//8 for v in tile['origin_yx_native']]
            assert receipt['prompt_names']==freeze['ordered_class_names']
            assert receipt['checkpoint_sha256']==freeze['checkpoint_sha256']
            assert receipt['calibrated_image_sha256']==row['calibrated_image_sha256']
            assert receipt['support_receipt_sha256']==row['support_receipt_sha256']
            assert receipt['source_sha256']==sha(SOURCE)
            assert receipt['freeze_sha256']==sha(FROZEN)
            assert sha(receipt['probabilities_path'])==receipt['probabilities_sha256']
            array=np.load(receipt['probabilities_path'],mmap_mode='r',allow_pickle=False)
            assert array.shape==(13,128,128) and array.dtype==np.float32
            assert np.isfinite(array).all()
            assert 0<=array.min()<=array.max()<=1
            assert abs(float(array.min())-receipt['probability_min'])<1e-7
            assert abs(float(array.max())-receipt['probability_max'])<1e-7
            minimum=min(minimum,float(array.min()))
            maximum=max(maximum,float(array.max()))
            seen.append(i)
    assert seen==list(range(len(row['selected_tiles'])))
    record={'status':'VERIFIED','slide':slide,'selected_tiles':len(seen),
            'all_tiles':row['all_tiles'],'probability_min':minimum,'probability_max':maximum,
            'all_tile_hashes_shapes_and_model_identities_verified':True,
            'source_freeze_sha256':sha(FROZEN),
            'controller_state_sha256':sha(state_path),
            'terminal_service_state':active,'terminal_gpu_contexts':[],
            'test_reference_opened':False,'online_publication':False}
    path=run_dir/'VISTA_RANGES_AUDIT_VERIFIED.json'
    path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'VERIFIED','slide':slide,'selected_tiles':len(seen),
                      'audit_sha256':sha(path)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True)
    a=p.parse_args();main(a.run_dir)
