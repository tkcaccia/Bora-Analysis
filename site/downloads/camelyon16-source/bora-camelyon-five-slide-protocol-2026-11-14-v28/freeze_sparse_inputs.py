#!/usr/bin/env python3
"""Source-only, exact tile/byte admission for full-1280 CAM sparse features."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0,'/mnt/sata_ssd/ChatGPT/Bora-Data/worktrees/bora-r-package-2026-11-14-v1/src')
from bora.r_feature_preflight import inspect_input
from bora.r_feature_worker import _scale_eligibility

ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
HERE=Path(__file__).parent
CAL=ROOT/'bora-camelyon-five-slide-protocol-2026-11-14-v24/CALIBRATION_STATE.json'
SUP=ROOT/'bora-camelyon-five-slide-protocol-2026-11-14-v34/SUPPORT_AUDIT_VERIFIED.json'
VIS=ROOT/'bora-camelyon-five-slide-protocol-2026-11-14-v34/VISUAL_REVIEW_VERIFIED.json'
PAIR=ROOT/'external-benchmark-wsss-camelyon-2026-11-13-v85/PAIR_FEATURES_FROZEN.json'
MODEL=Path('/mnt/sata_ssd/ChatGPT/Bora-Data/models/Virchow2/3158645804b69e3f3bc4439d4116edddf0840a72')
DATA=Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
OUT=HERE/'SPARSE_INPUTS_FROZEN.json'
FEATURE_ROOT=DATA/'external-camelyon16-five-slide-2026-11-14-v30/features'
CAP=200*(1<<30)
RESERVE=100*(1<<30)
BLOCK_BYTES=16*16*1280*2
SLIDES=('tumor_084','tumor_091','normal_019')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(16<<20),b''):h.update(block)
    return h.hexdigest()


def atomic_npy(path,array):
    temporary=path.with_suffix(path.suffix+'.partial')
    with temporary.open('xb') as stream:
        np.save(stream,array,allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)


def used_cam_bytes():
    paths=list(DATA.glob('external-camelyon16-*'))
    if not paths:return 0
    output=subprocess.check_output(['du','-scB1',*(str(x) for x in paths)],text=True)
    return int(output.splitlines()[-1].split()[0])


def main():
    if OUT.exists():raise FileExistsError(OUT)
    cal=json.loads(CAL.read_text())
    sup=json.loads(SUP.read_text())
    vis=json.loads(VIS.read_text())
    pair=json.loads(PAIR.read_text())
    if cal['status']!='COMPLETE' or \
       sup['status']!='PIXEL_VERIFIED_VISUAL_REVIEW_PENDING' or \
       vis['status']!='THREE_TEST_SUPPORTS_VISUALLY_ACCEPTED' or \
       pair['status']!='PAIR_FULL1280_THREE_SCALE_FEATURES_FROZEN':
        raise ValueError('Calibrated RGB, support, visual and model receipts must pass')
    if vis['support_audit_sha256']!=sha(SUP):
        raise ValueError('Visual review is not bound to support audit')
    if [r['slide_id'] for r in cal['completed']]!=list(SLIDES) or \
       [r['slide_id'] for r in sup['rows']]!=list(SLIDES):
        raise ValueError('Frozen test slides differ')
    assert not cal['expert_reference_opened'] and not sup['expert_reference_opened']
    model_hash=sha(MODEL/'model.safetensors')
    config_hash=sha(MODEL/'config.json')
    if model_hash!=pair['model_sha256']:
        raise ValueError('Virchow2 model differs from original pair')
    by_support={r['slide_id']:r for r in sup['rows']}
    prepared=[]
    output_specs=[]
    for row in cal['completed']:
        sid=row['slide_id']
        image=Path(row['calibrated_path'])
        tissue=Path('/mnt/sata_ssd/ChatGPT/Bora-Data/external-camelyon16-five-slide-2026-11-14-v33/he-support')/sid/'tissue-area.geojson'
        if sha(tissue)!=by_support[sid]['geojson_sha256']:
            raise ValueError(f'{sid} reviewed tissue GeoJSON differs')
        source=inspect_input(image,tissue)
        if source['image_sha256']!=row['calibrated_sha256'] or \
           source['image_shape_yx']!=row['padded_shape_yx']:
            raise ValueError(f'{sid} source image/preflight differs')
        base=source['eligible']
        if not base.any():raise ValueError(f'{sid} has no tissue tiles')
        eligible_path=HERE/f'{sid}-eligible.npy'
        count_path=HERE/f'{sid}-tissue-counts.npy'
        atomic_npy(eligible_path,base)
        atomic_npy(count_path,source['counts'])
        scales=[]
        for factor in (1,2,4):
            grid=_scale_eligibility(base,factor)
            selected=int(grid.sum())
            rows=int(grid.shape[0])
            feature_bytes=selected*BLOCK_BYTES
            allowance=feature_bytes+rows*4096+(1<<20)
            spec={
                'format':'bora-cam-sparse-feature-worker-v1',
                'slide_id':sid,'image':str(image),
                'image_sha256':source['image_sha256'],
                'tissue_support':str(tissue),
                'tissue_support_sha256':source['tissue_support_sha256'],
                'eligible_path':str(eligible_path),
                'eligible_sha256':sha(eligible_path),
                'base_grid_shape_yx':list(base.shape),
                'padded_shape_yx':source['padded_shape_yx'],
                'image_shape_yx':source['image_shape_yx'],
                'model_dir':str(MODEL),
                'model_checkpoint_sha256':model_hash,
                'model_config_sha256':config_hash,
                'output_dir':str(FEATURE_ROOT/sid/f'factor-{factor}'),
                'factor':factor,'max_seconds':10800,
                'max_feature_bytes':allowance,'reserve_bytes':RESERVE,
            }
            path=HERE/f'{sid}-f{factor}-SPEC.json'
            path.write_text(json.dumps(spec,indent=2)+'\n')
            output_specs.append(str(path))
            scales.append({'factor':factor,'grid_yx':list(grid.shape),
                           'eligible_tiles':selected,
                           'feature_payload_bytes':feature_bytes,
                           'allocated_bytes':allowance,
                           'spec_path':str(path),'spec_sha256':sha(path)})
        prepared.append({'slide_id':sid,'image_sha256':source['image_sha256'],
                         'support_sha256':source['tissue_support_sha256'],
                         'eligible_path':str(eligible_path),'eligible_sha256':sha(eligible_path),
                         'tissue_counts_path':str(count_path),'tissue_counts_sha256':sha(count_path),
                         'base_grid_shape_yx':list(base.shape),'scales':scales})
    current=used_cam_bytes()
    projected=sum(s['allocated_bytes'] for r in prepared for s in r['scales'])
    free=shutil.disk_usage(DATA).free
    if current+projected+(1<<30)>CAP or free-projected-(1<<30)<RESERVE:
        raise OSError('Sparse full-channel features exceed campaign cap or free reserve')
    result={'format':'bora-cam-three-test-sparse-feature-admission-v1',
            'status':'SOURCE_ONLY_INPUTS_FROZEN_STORAGE_ADMITTED',
            'calibration_sha256':sha(CAL),'support_audit_sha256':sha(SUP),
            'visual_review_sha256':sha(VIS),'model_sha256':model_hash,
            'model_config_sha256':config_hash,
            'worker_source_sha256':sha(ROOT/'bora-camelyon-five-slide-protocol-2026-11-14-v26/sparse_feature_worker.py'),
            'cases':prepared,'specs':output_specs,
            'campaign_bytes_before':current,'projected_sparse_bytes':projected,
            'campaign_cap_bytes':CAP,'free_bytes_before':free,
            'reserve_bytes':RESERVE,'expert_reference_opened':False,
            'online_publication':False}
    temporary=OUT.with_suffix('.json.partial')
    with temporary.open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,OUT)
    print(json.dumps({'status':result['status'],
                      'projected_sparse_gib':round(projected/(1<<30),3),
                      'campaign_peak_gib':round((current+projected+(1<<30))/(1<<30),3)}))


if __name__=='__main__':main()
