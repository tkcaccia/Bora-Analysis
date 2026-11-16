"""Independent source-only VISTA tile-admission and identity audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora')
HERE=Path(__file__).resolve().parent
FREEZE=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-15-v132/VISTA_SOURCE_FROZEN.json'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def starts(length):
    values=list(range(0,length-1024+1,896))
    if values[-1]!=length-1024:values.append(length-1024)
    return values


def main():
    frozen=json.loads(FREEZE.read_text())
    assert frozen['status']=='CAM_TEST_VISTA_FIXED_13_SOURCE_ONLY_NATIVE_TILES_FROZEN'
    assert len(frozen['ordered_class_names'])==13 and frozen['ordered_class_names'][0]=='tumor'
    assert sha(frozen['checkpoint_path'])==frozen['checkpoint_sha256']
    assert not frozen['test_reference_opened']
    rows=[]
    for slide,row in frozen['slides'].items():
        receipt=Path(row['support_receipt_path'])
        assert sha(receipt)==row['support_receipt_sha256']
        support=json.loads(receipt.read_text())['support']
        path=Path(support['support_path'])
        assert sha(path)==support['support_sha256']==row['support_mask_sha256']
        h,w=row['calibrated_shape_yx']
        mask=np.memmap(path,mode='r',dtype=np.uint8,shape=(h//8,w//8))
        expected=[]
        for y in starts(h):
            for x in starts(w):
                pixels=np.asarray(mask[y//8:y//8+128,x//8:x//8+128])
                fraction=float(np.count_nonzero(pixels)/16384)
                if fraction>=.05:
                    expected.append(([y,x],fraction))
        actual=row['selected_tiles']
        assert len(expected)==len(actual)
        for (origin,fraction),entry in zip(expected,actual):
            assert entry['origin_yx_native']==origin
            assert abs(entry['support_fraction_4um']-fraction)<1e-12
        rows.append({'slide':slide,'all_tiles':row['all_tiles'],
                     'selected_tiles':len(actual),'all_admissions_recomputed':True,
                     'support_sha256':sha(path)})
    record={'status':'VERIFIED','vista_freeze_sha256':sha(FREEZE),
            'rows':rows,'test_reference_opened':False,'online_publication':False}
    path=HERE/'VISTA_SCOPE_AUDIT_VERIFIED.json'
    path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'VERIFIED','rows':rows}))


if __name__=='__main__':main()
