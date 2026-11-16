"""Independent exact-pixel reconstruction of CAM prompt-assisted MedSAM masks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA=Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE=Path(__file__).resolve().parent
FREEZE=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-15-v129/ASSEMBLY_FROZEN.json'
PROMPTS=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-14-v120/MEDSAM_PROMPTS_FROZEN.json'
PILOT=DATA/'external-camelyon16-five-slide-2026-11-14-v123/medsam-pilot/tumor_091/range-0000-0001'
TILES=DATA/'external-camelyon16-five-slide-2026-11-15-v125/medsam-ranges'
OUTPUT=DATA/'external-camelyon16-five-slide-2026-11-15-v129/medsam-masks'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def main(slide):
    spec=json.loads(FREEZE.read_text())
    assert sha(PROMPTS)==spec['prompt_manifest_sha256']
    info=spec['slides'][slide]
    assert sha(info['range_audit_path'])==info['range_audit_sha256']
    row=next(x for x in json.loads(PROMPTS.read_text())['slides'] if x['slide']==slide)
    height,width=row['shape_yx_4um']
    folder=OUTPUT/slide
    receipt_path=folder/'ASSEMBLY_COMPLETE.json'
    receipt=json.loads(receipt_path.read_text())
    assert receipt['status']=='CAM_TEST_MEDSAM_GUIDED_4UM_MASKS_ASSEMBLED_SOURCE_ONLY'
    assert receipt['assembly_freeze_sha256']==sha(FREEZE)
    assert receipt['range_audit_sha256']==sha(info['range_audit_path'])
    assert sha(receipt['native_path'])==receipt['native_sha256']
    assert sha(receipt['shared_path'])==receipt['shared_sha256']
    support_entry=json.loads(Path(spec['support_receipts'][slide]['path']).read_text())['support']
    support_path=Path(support_entry['support_path'])
    assert sha(support_path)==support_entry['support_sha256']==receipt['support_sha256']
    support=np.memmap(support_path,mode='r',dtype=np.uint8,shape=(height,width))
    expected=np.zeros((height,width),dtype=np.uint8)
    for index,tile in enumerate(row['tiles']):
        if slide=='tumor_091' and index==0:
            source=PILOT/'tile-0000'
        else:
            part=next(p for p in info['ranges'] if p['start']<=index<p['stop'])
            source=TILES/slide/f"range-{part['start']:04d}-{part['stop']:04d}"/f'tile-{index:04d}'
        r=json.loads((source/'TILE_COMPLETE.json').read_text())
        assert sha(r['mask_path'])==r['mask_sha256']
        array=np.load(r['mask_path'],allow_pickle=False)
        y,x=tile['origin_yx_4um'];h,w=tile['shape_yx_4um']
        assert array.shape==(h,w)
        expected[y:y+h,x:x+w]=array
    native_count=shared_count=0
    with rasterio.open(receipt['native_path']) as native, rasterio.open(receipt['shared_path']) as shared:
        assert (native.height,native.width,shared.height,shared.width)==(height,width,height,width)
        for y0 in range(0,height,256):
            y1=min(height,y0+256)
            window=Window(0,y0,width,y1-y0)
            n=native.read(1,window=window)
            s=shared.read(1,window=window)
            assert np.array_equal(n,expected[y0:y1])
            assert np.array_equal(s,np.where(support[y0:y1]>0,expected[y0:y1],0))
            native_count+=int(n.sum());shared_count+=int(s.sum())
    assert native_count==receipt['native_tumor_pixels']
    assert shared_count==receipt['shared_tumor_pixels']
    record={'status':'VERIFIED','slide':slide,'shape_yx':[height,width],
            'native_sha256':receipt['native_sha256'],'shared_sha256':receipt['shared_sha256'],
            'native_tumor_pixels':native_count,'shared_tumor_pixels':shared_count,
            'prompted_tiles':row['prompted_tiles'],'all_tiles':row['all_tiles'],
            'all_pixels_reconstructed_from_audited_tiles':True,'shared_support_exact':True,
            'assembly_receipt_sha256':sha(receipt_path),
            'test_reference_opened':False,'online_publication':False}
    path=folder/'MASKS_AUDIT_VERIFIED.json'
    path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'VERIFIED','slide':slide,
                      'native_tumor_pixels':native_count,'shared_tumor_pixels':shared_count,
                      'audit_sha256':sha(path)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--slide',choices=('tumor_084','tumor_091','normal_019'),required=True)
    a=p.parse_args();main(a.slide)
