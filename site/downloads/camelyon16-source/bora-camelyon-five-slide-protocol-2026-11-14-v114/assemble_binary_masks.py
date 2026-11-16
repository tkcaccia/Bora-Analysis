"""Assemble independently audited CAM binary bands on the fixed 4-µm grid."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA=Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE=Path(__file__).resolve().parent
BANDS=DATA/'external-camelyon16-five-slide-2026-11-14-v103/binary-network-bands'
OUTPUT=DATA/'external-camelyon16-five-slide-2026-11-14-v114/binary-network-masks'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def atomic(path,obj):
    partial=path.with_name(path.name+'.partial')
    partial.write_text(json.dumps(obj,indent=2)+'\n')
    os.replace(partial,path)


def run(method,slide):
    start=time.monotonic()
    freeze_path=HERE/'ASSEMBLY_FROZEN.json'
    freeze=json.loads(freeze_path.read_text())
    assert freeze['status']=='CAM_BINARY_MASK_ASSEMBLY_FROZEN'
    assert freeze['source_sha256']==sha(__file__)
    key=method+'/'+slide
    entry=freeze['entries'][key]
    audit_path=Path(entry['band_audit_path'])
    assert sha(audit_path)==entry['band_audit_sha256']
    audit=json.loads(audit_path.read_text())
    assert audit['status']=='VERIFIED' and audit['method']==method and audit['slide']==slide
    assert not audit['test_reference_opened']
    receipt_path=Path(freeze['support_receipts'][slide]['path'])
    assert sha(receipt_path)==freeze['support_receipts'][slide]['sha256']
    support_receipt=json.loads(receipt_path.read_text())
    support= support_receipt['support']
    support_path=Path(support['support_path'])
    assert sha(support_path)==support['support_sha256']
    height,width=support['shape_yx']
    mask=np.memmap(support_path,mode='r',dtype=np.uint8,shape=(height,width))
    folder=OUTPUT/method/slide
    if folder.exists():
        raise FileExistsError('No in-place mask assembly retry')
    folder.mkdir(parents=True)
    try:
        score=np.memmap(folder/'score-sum.partial.mmap',mode='w+',dtype=np.float32,shape=(2,height,width))
        count=np.memmap(folder/'patch-count.partial.mmap',mode='w+',dtype=np.uint8,shape=(height,width))
        score[:]=0
        count[:]=0
        patches=0
        for item in audit['details']:
            band=item['band']
            source=BANDS/method/slide/f'band-{band:04d}'
            receipt=json.loads((source/'BAND_COMPLETE.json').read_text())
            assert sha(source/'BAND_COMPLETE.json')==item['receipt_sha256']
            y0,y1=receipt['comparison_row_start'],receipt['comparison_row_stop']
            assert [y0,y1]==item['rows']
            source_score=np.load(source/'score-sum.npy',mmap_mode='r',allow_pickle=False)
            source_count=np.load(source/'patch-count.npy',mmap_mode='r',allow_pickle=False)
            assert source_score.shape==(2,y1-y0,width) and source_count.shape==(y1-y0,width)
            score[:,y0:y1,:]+=source_score
            count[y0:y1,:]+=source_count
            patches+=receipt['patches']
        score.flush()
        count.flush()
        profile={'driver':'GTiff','height':height,'width':width,'count':1,'dtype':'uint8',
                 'tiled':True,'blockxsize':256,'blockysize':256,'compress':'zstd'}
        native_path=folder/'native-binary-classes-4um.tif'
        shared_path=folder/'shared-support-binary-classes-4um.tif'
        native_hist=np.zeros(3,dtype=np.int64)
        shared_hist=np.zeros(3,dtype=np.int64)
        minimum,maximum=255,0
        with rasterio.open(native_path,'w',**profile) as native, rasterio.open(shared_path,'w',**profile) as shared:
            for y0 in range(0,height,256):
                y1=min(height,y0+256)
                c=count[y0:y1,:]
                minimum=min(minimum,int(c.min()))
                maximum=max(maximum,int(c.max()))
                if c.min()==0:
                    raise RuntimeError('Network band assembly left grid pixels uncovered')
                n=(np.argmax(score[:,y0:y1,:],axis=0)+1).astype(np.uint8)
                s=np.where(mask[y0:y1,:]>0,n,0).astype(np.uint8)
                window=Window(0,y0,width,y1-y0)
                native.write(n,1,window=window)
                shared.write(s,1,window=window)
                native_hist+=np.bincount(n.ravel(),minlength=3)
                shared_hist+=np.bincount(s.ravel(),minlength=3)
        record={'status':'CAM_BINARY_NETWORK_MASKS_ASSEMBLED_SOURCE_ONLY',
                'method':method,'slide':slide,'comparison_shape_yx':[height,width],
                'comparison_spacing_um_xy':[4.0,4.0],'bands':audit['bands'],
                'patches':patches,'min_patch_coverage':minimum,'max_patch_coverage':maximum,
                'native_path':str(native_path),'native_sha256':sha(native_path),
                'shared_path':str(shared_path),'shared_sha256':sha(shared_path),
                'native_histogram_0_to_2':native_hist.tolist(),
                'shared_histogram_0_to_2':shared_hist.tolist(),
                'support_sha256':support['support_sha256'],
                'band_audit_sha256':sha(audit_path),
                'assembly_source_sha256':sha(__file__),
                'assembly_freeze_sha256':sha(freeze_path),
                'seconds':time.monotonic()-start,
                'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'scratch_pending_independent_audit':True,
                'test_reference_opened':False,'online_publication':False}
        atomic(folder/'ASSEMBLY_COMPLETE.json',record)
        print(json.dumps({'method':method,'slide':slide,'native_sha256':record['native_sha256'],
                          'shared_sha256':record['shared_sha256']}),flush=True)
    except BaseException as error:
        atomic(folder/'ASSEMBLY_FAILED.json',{'status':'FAILED_LATCHED_PRESERVE_PARTIAL',
                                              'method':method,'slide':slide,'error':repr(error),
                                              'test_reference_opened':False})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=('unet','deeplabv3plus'),required=True)
    parser.add_argument('--slide',choices=('tumor_084','tumor_091','normal_019'),required=True)
    args=parser.parse_args()
    run(args.method,args.slide)
