#!/usr/bin/env python3
"""Independent arithmetic/hash audit of source-only CAM sparse feature admission."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
FROZEN=ROOT/'bora-camelyon-five-slide-protocol-2026-11-14-v28/SPARSE_INPUTS_FROZEN.json'
OUT=Path(__file__).parent/'SPARSE_ADMISSION_AUDIT_VERIFIED.json'
BLOCK=16*16*1280*2


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    if OUT.exists():raise FileExistsError(OUT)
    frozen=json.loads(FROZEN.read_text())
    assert frozen['status']=='SOURCE_ONLY_INPUTS_FROZEN_STORAGE_ADMITTED'
    assert not frozen['expert_reference_opened'] and not frozen['online_publication']
    assert [r['slide_id'] for r in frozen['cases']]==['tumor_084','tumor_091','normal_019']
    total=0
    rows=[]
    for case in frozen['cases']:
        sid=case['slide_id']
        counts=np.load(case['tissue_counts_path'],allow_pickle=False)
        eligible=np.load(case['eligible_path'],allow_pickle=False)
        assert counts.dtype==np.uint32 and eligible.dtype==np.bool_
        assert np.array_equal(eligible,counts>0)
        assert sha(case['eligible_path'])==case['eligible_sha256']
        assert sha(case['tissue_counts_path'])==case['tissue_counts_sha256']
        assert list(counts.shape)==case['base_grid_shape_yx']
        assert int(counts.max())<=224*224
        for scale in case['scales']:
            factor=scale['factor']
            if factor==1: grid=eligible
            else:
                gy,gx=eligible.shape
                grid=eligible.reshape(gy//factor,factor,gx//factor,factor).any(axis=(1,3))
            selected=int(grid.sum())
            payload=selected*BLOCK
            assert scale['grid_yx']==list(grid.shape)
            assert scale['eligible_tiles']==selected
            assert scale['feature_payload_bytes']==payload
            assert scale['allocated_bytes']>=payload+grid.shape[0]*4096
            assert sha(scale['spec_path'])==scale['spec_sha256']
            spec=json.loads(Path(scale['spec_path']).read_text())
            assert spec['format']=='bora-cam-sparse-feature-worker-v1'
            assert spec['slide_id']==sid and spec['factor']==factor
            assert spec['image_sha256']==case['image_sha256']
            assert spec['tissue_support_sha256']==case['support_sha256']
            assert spec['model_checkpoint_sha256']==frozen['model_sha256']
            assert spec['model_config_sha256']==frozen['model_config_sha256']
            assert spec['max_feature_bytes']==scale['allocated_bytes']
            assert spec['reserve_bytes']==100*(1<<30)
            total+=scale['allocated_bytes']
            rows.append({'slide_id':sid,'factor':factor,'grid_yx':list(grid.shape),
                         'eligible_tiles':selected,'payload_bytes':payload,
                         'spec_sha256':scale['spec_sha256']})
    assert total==frozen['projected_sparse_bytes']
    assert frozen['campaign_bytes_before']+total+(1<<30)<=frozen['campaign_cap_bytes']
    assert frozen['free_bytes_before']-total-(1<<30)>=frozen['reserve_bytes']
    result={'format':'bora-cam-sparse-admission-audit-v1','status':'VERIFIED',
            'frozen_inputs_sha256':sha(FROZEN),'rows':rows,
            'projected_sparse_bytes':total,'campaign_cap_bytes':frozen['campaign_cap_bytes'],
            'expert_reference_opened':False,'online_publication':False}
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'VERIFIED','scales':len(rows),
                      'projected_sparse_gib':round(total/(1<<30),3)}))


if __name__=='__main__':main()
