"""Freeze VISTA-PATH fixed-prompt, source-only test-slide tile admission."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA=Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE=Path(__file__).resolve().parent
OLD=ROOT/'migration/external-benchmark-wsss-camelyon-2026-11-14-v116/VISTA_SOURCE_FROZEN.json'
TEST_INPUTS=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-14-v103/INPUTS_FROZEN.json'
TILE_SOURCE=HERE/'vista_test_tiles.py'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def starts(length):
    positions=list(range(0,length-1024+1,896))
    if positions[-1]!=length-1024:
        positions.append(length-1024)
    return positions


def main():
    out=HERE/'VISTA_SOURCE_FROZEN.json'
    if out.exists():raise FileExistsError('No in-place VISTA source refreeze')
    old=json.loads(OLD.read_text())
    inputs=json.loads(TEST_INPUTS.read_text())
    assert old['ordered_class_names'][0]=='tumor' and len(old['ordered_class_names'])==13
    assert sha(old['checkpoint_path'])==old['checkpoint_sha256']
    slides={}
    for name in ('tumor_084','tumor_091','normal_019'):
        row=inputs['slides'][name]
        if name=='tumor_091':
            receipt=DATA/'external-camelyon16-five-slide-2026-11-14-v99/bora-test-mask/tumor_091/MASK_COMPLETE.json'
        else:
            receipt=DATA/f'external-camelyon16-five-slide-2026-11-14-v102/bora-test-masks/{name}/MASK_COMPLETE.json'
        support=json.loads(receipt.read_text())['support']
        path=Path(support['support_path'])
        assert sha(path)==support['support_sha256']
        height,width=row['calibrated_shape_yx']
        assert [height//8,width//8]==support['shape_yx']
        mask=np.memmap(path,mode='r',dtype=np.uint8,shape=tuple(support['shape_yx']))
        selected=[]
        top_positions,left_positions=starts(height),starts(width)
        for top in top_positions:
            for left in left_positions:
                crop=mask[top//8:top//8+128,left//8:left//8+128]
                if crop.shape!=(128,128):raise RuntimeError('Unaligned support crop')
                fraction=float(np.mean(crop>0))
                if fraction>=old['support_gate_fraction']:
                    selected.append({'origin_yx_native':[top,left],
                                     'support_fraction_4um':fraction})
        slides[name]={'calibrated_state_path':row['calibrated_state_path'],
                      'calibrated_state_sha256':row['calibrated_state_sha256'],
                      'calibrated_image_path':row['calibrated_image_path'],
                      'calibrated_image_sha256':row['calibrated_image_sha256'],
                      'calibrated_shape_yx':[height,width],
                      'support_receipt_path':str(receipt),
                      'support_receipt_sha256':sha(receipt),
                      'support_mask_sha256':support['support_sha256'],
                      'all_tiles':len(top_positions)*len(left_positions),
                      'selected_tiles':selected}
    freeze={'status':'CAM_TEST_VISTA_FIXED_13_SOURCE_ONLY_NATIVE_TILES_FROZEN',
            'tile_source_sha256':sha(TILE_SOURCE),
            'vista_utils_sha256':old['vista_utils_sha256'],
            'vista_prior_regression_sha256':old['vista_prior_regression_sha256'],
            'test_inputs_sha256':sha(TEST_INPUTS),
            'old_pair_settings_sha256':sha(OLD),
            'checkpoint_path':old['checkpoint_path'],
            'checkpoint_sha256':old['checkpoint_sha256'],
            'ordered_class_names':old['ordered_class_names'],
            'settings':old['settings'],
            'native_mpp':0.5,'comparison_mpp':4.0,
            'sampling_offset_native_pixels':4,
            'support_gate_fraction':old['support_gate_fraction'],
            'maximum_tiles_per_child':100,
            'slides':slides,'test_reference_opened':False,'online_publication':False}
    out.write_text(json.dumps(freeze,indent=2)+'\n')
    print(json.dumps({'status':freeze['status'],
       'slides':{k:{'all':v['all_tiles'],'selected':len(v['selected_tiles'])} for k,v in slides.items()},
       'freeze_sha256':sha(out)}))


if __name__=='__main__':
    main()
