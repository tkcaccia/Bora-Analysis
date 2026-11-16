#!/usr/bin/env python3
"""Independently audit one sparse Virchow2 pilot and terminal GPU cleanup."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
PILOT=ROOT/'bora-camelyon-five-slide-protocol-2026-11-14-v36/PILOT_FROZEN.json'
SPEC=ROOT/'bora-camelyon-five-slide-protocol-2026-11-14-v36/PILOT_SPEC.json'
FEATURES=Path('/mnt/sata_ssd/ChatGPT/Bora-Data/external-camelyon16-five-slide-2026-11-14-v36/pilot-normal_019-f4/FEATURES_COMPLETE.json')
ELIGIBLE=ROOT/'bora-camelyon-five-slide-protocol-2026-11-14-v28/normal_019-eligible.npy'
WORKER=ROOT/'bora-camelyon-five-slide-protocol-2026-11-14-v26/sparse_feature_worker.py'
OUT=Path(__file__).parent/'SPARSE_PILOT_AUDIT_VERIFIED.json'
SERVICE='bora-cam16-sparse-f4-pilot-v36.service'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    if OUT.exists():raise FileExistsError(OUT)
    freeze=json.loads(PILOT.read_text())
    spec=json.loads(SPEC.read_text())
    complete=json.loads(FEATURES.read_text())
    assert freeze['status']=='FROZEN_BEFORE_GPU'
    assert freeze['spec_sha256']==sha(SPEC)
    assert freeze['worker_source_sha256']==sha(WORKER)
    assert complete['status']=='COMPLETE'
    assert complete['source_spec_sha256']==sha(SPEC)
    assert complete['factor']==spec['factor']==4
    assert complete['channels']==1280 and complete['dtype']=='float16'
    base=np.load(ELIGIBLE,allow_pickle=False)
    assert base.dtype==np.bool_
    assert sha(ELIGIBLE)==spec['eligible_sha256']
    gy,gx=base.shape
    grid=base.reshape(gy//4,4,gx//4,4).any(axis=(1,3))
    assert complete['shape_yxc']==[grid.shape[0]*16,grid.shape[1]*16,1280]
    assert complete['rows']==len(complete['row_receipts'])==grid.shape[0]
    total_tiles=total_bytes=0
    for row,receipt in enumerate(complete['row_receipts']):
        path=FEATURES.parent/f'row-{row:04d}.json'
        assert json.loads(path.read_text())==receipt
        expected=np.flatnonzero(grid[row]).astype(np.int32)
        assert receipt['row']==row and receipt['tile_columns']==len(expected)
        if len(expected):
            columns=Path(receipt['column_path'])
            tiles=Path(receipt['tile_path'])
            assert columns.parent==tiles.parent==FEATURES.parent
            assert sha(columns)==receipt['column_sha256']
            assert sha(tiles)==receipt['tile_sha256']
            assert np.array_equal(np.load(columns,allow_pickle=False),expected)
            data=np.load(tiles,mmap_mode='r',allow_pickle=False)
            assert data.dtype==np.float16
            assert data.shape==(len(expected),16,16,1280)
            assert np.isfinite(data).all()
            total_bytes+=columns.stat().st_size+tiles.stat().st_size
        else:
            assert receipt['tile_path'] is None and receipt['column_path'] is None
        total_tiles+=len(expected)
    assert total_tiles==complete['eligible_tiles']==complete['persisted_tiles']==66
    assert total_bytes==complete['persisted_bytes']
    assert total_bytes<=spec['max_feature_bytes']
    status=subprocess.check_output(['systemctl','--user','show',SERVICE,'-p','ActiveState',
        '-p','SubState','-p','MainPID','-p','Result','-p','ExecMainStatus','-p','ControlGroup'],text=True)
    fields=dict(line.split('=',1) for line in status.splitlines())
    assert fields['ActiveState']=='inactive' and fields['SubState']=='dead'
    assert fields['MainPID']=='0' and fields['Result']=='success'
    assert fields['ExecMainStatus']=='0' and fields['ControlGroup']==''
    gpu=subprocess.check_output(['timeout','8','nvidia-smi','--query-compute-apps=pid,gpu_uuid',
                                  '--format=csv,noheader'],text=True)
    assert not gpu.strip()
    result={'format':'bora-cam-sparse-gpu-pilot-audit-v1','status':'VERIFIED',
            'pilot_freeze_sha256':sha(PILOT),'feature_receipt_sha256':sha(FEATURES),
            'source_spec_sha256':sha(SPEC),'worker_source_sha256':sha(WORKER),
            'factor':4,'slide_id':'normal_019','rows':grid.shape[0],
            'persisted_tiles':total_tiles,'persisted_bytes':total_bytes,
            'all_tile_rows_columns_hashes_and_finite_tokens_verified':True,
            'terminal_service':fields,'assigned_gpu_contexts_after':[],
            'peak_host_rss_kb':complete['peak_host_rss_kb'],
            'peak_cuda_reserved_bytes':complete['peak_cuda_reserved_bytes'],
            'wall_seconds':complete['wall_seconds'],
            'expert_reference_opened':False,'online_publication':False}
    tmp=OUT.with_suffix('.json.partial')
    with tmp.open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,OUT)
    print(json.dumps({'status':'VERIFIED','tiles':total_tiles,'wall_seconds':complete['wall_seconds']}))


if __name__=='__main__':main()
