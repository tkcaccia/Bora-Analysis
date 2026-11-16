"""Assemble audited prompt-assisted MedSAM test tiles on the 4-µm grid."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA=Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE=Path(__file__).resolve().parent
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


def atomic(path,value):
    partial=path.with_name(path.name+'.partial')
    partial.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(partial,path)


def main(slide):
    freeze_path=HERE/'ASSEMBLY_FROZEN.json'
    freeze=json.loads(freeze_path.read_text())
    assert freeze['source_sha256']==sha(__file__)
    assert freeze['prompt_manifest_sha256']==sha(PROMPTS)
    run=freeze['slides'][slide]
    audit_path=Path(run['range_audit_path'])
    assert sha(audit_path)==run['range_audit_sha256']
    assert json.loads(audit_path.read_text())['status']=='VERIFIED'
    prompts=json.loads(PROMPTS.read_text())
    row=next(x for x in prompts['slides'] if x['slide']==slide)
    height,width=row['shape_yx_4um']
    support_receipt=json.loads(Path(freeze['support_receipts'][slide]['path']).read_text())
    support_entry=support_receipt['support']
    assert sha(support_entry['support_path'])==support_entry['support_sha256']
    support=np.memmap(support_entry['support_path'],mode='r',dtype=np.uint8,shape=(height,width))
    folder=OUTPUT/slide
    if folder.exists():raise FileExistsError('No in-place MedSAM mask retry')
    folder.mkdir(parents=True)
    try:
        mask=np.memmap(folder/'native-tumor.partial.mmap',mode='w+',dtype=np.uint8,shape=(height,width))
        mask[:]=0
        seen=[]
        for index,tile in enumerate(row['tiles']):
            if slide=='tumor_091' and index==0:
                source=PILOT/'tile-0000'
            else:
                part=next(x for x in run['ranges'] if x['start']<=index<x['stop'])
                source=TILES/slide/f"range-{part['start']:04d}-{part['stop']:04d}"/f'tile-{index:04d}'
            receipt=json.loads((source/'TILE_COMPLETE.json').read_text())
            assert receipt['origin_yx_4um']==tile['origin_yx_4um']
            assert receipt['shape_yx_4um']==tile['shape_yx_4um']
            path=Path(receipt['mask_path'])
            assert sha(path)==receipt['mask_sha256']
            partmask=np.load(path,allow_pickle=False)
            y,x=tile['origin_yx_4um'];h,w=tile['shape_yx_4um']
            assert partmask.shape==(h,w)
            mask[y:y+h,x:x+w]=partmask
            seen.append(index)
        assert seen==list(range(row['prompted_tiles']))
        mask.flush()
        profile={'driver':'GTiff','height':height,'width':width,'count':1,'dtype':'uint8',
                 'tiled':True,'blockxsize':256,'blockysize':256,'compress':'zstd'}
        native_path=folder/'native-tumor-4um.tif'
        shared_path=folder/'shared-support-tumor-4um.tif'
        native_count=shared_count=0
        with rasterio.open(native_path,'w',**profile) as native, rasterio.open(shared_path,'w',**profile) as shared:
            for y0 in range(0,height,256):
                y1=min(height,y0+256)
                n=np.asarray(mask[y0:y1,:])
                s=np.where(support[y0:y1,:]>0,n,0).astype(np.uint8)
                native.write(n,1,window=Window(0,y0,width,y1-y0))
                shared.write(s,1,window=Window(0,y0,width,y1-y0))
                native_count+=int(n.sum());shared_count+=int(s.sum())
        record={'status':'CAM_TEST_MEDSAM_GUIDED_4UM_MASKS_ASSEMBLED_SOURCE_ONLY',
                'slide':slide,'shape_yx':[height,width],
                'prompted_tiles':row['prompted_tiles'],'all_tiles':row['all_tiles'],
                'no_prompt_tiles':row['all_tiles']-row['prompted_tiles'],
                'native_tumor_pixels':native_count,'shared_tumor_pixels':shared_count,
                'native_path':str(native_path),'native_sha256':sha(native_path),
                'shared_path':str(shared_path),'shared_sha256':sha(shared_path),
                'support_sha256':support_entry['support_sha256'],
                'range_audit_sha256':sha(audit_path),
                'source_sha256':sha(__file__),'assembly_freeze_sha256':sha(freeze_path),
                'scratch_pending_independent_audit':True,
                'test_reference_opened':False,'online_publication':False}
        atomic(folder/'ASSEMBLY_COMPLETE.json',record)
        print(json.dumps({'slide':slide,'native_tumor_pixels':native_count,
                          'shared_tumor_pixels':shared_count}))
    except BaseException as error:
        atomic(folder/'FAILED.json',{'status':'FAILED_LATCHED_PRESERVE_PARTIAL',
             'slide':slide,'error':repr(error),'test_reference_opened':False})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--slide',choices=('tumor_084','tumor_091','normal_019'),required=True)
    a=p.parse_args();main(a.slide)
