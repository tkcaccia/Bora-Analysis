"""Independent exact-pixel audit of 4-µm native and common-support masks."""
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
BANDS=DATA/'external-camelyon16-five-slide-2026-11-14-v103/binary-network-bands'
OUTPUT=DATA/'external-camelyon16-five-slide-2026-11-14-v114/binary-network-masks'
FREEZE=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-14-v114/ASSEMBLY_FROZEN.json'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def main(method,slide):
    freeze=json.loads(FREEZE.read_text())
    key=method+'/'+slide
    entry=freeze['entries'][key]
    audit_path=Path(entry['band_audit_path'])
    assert sha(audit_path)==entry['band_audit_sha256']
    audit=json.loads(audit_path.read_text())
    folder=OUTPUT/method/slide
    receipt_path=folder/'ASSEMBLY_COMPLETE.json'
    receipt=json.loads(receipt_path.read_text())
    assert receipt['status']=='CAM_BINARY_NETWORK_MASKS_ASSEMBLED_SOURCE_ONLY'
    assert (receipt['method'],receipt['slide'])==(method,slide)
    assert receipt['band_audit_sha256']==sha(audit_path)
    assert receipt['assembly_freeze_sha256']==sha(FREEZE)
    assert not receipt['test_reference_opened'] and not receipt['online_publication']
    for name in ('native','shared'):
        assert sha(receipt[name+'_path'])==receipt[name+'_sha256']
    source_support_receipt=Path(freeze['support_receipts'][slide]['path'])
    assert sha(source_support_receipt)==freeze['support_receipts'][slide]['sha256']
    support_entry=json.loads(source_support_receipt.read_text())['support']
    support_path=Path(support_entry['support_path'])
    assert sha(support_path)==support_entry['support_sha256']==receipt['support_sha256']
    height,width=receipt['comparison_shape_yx']
    support=np.memmap(support_path,mode='r',dtype=np.uint8,shape=(height,width))
    band_info=[]
    for item in audit['details']:
        path=BANDS/method/slide/f"band-{item['band']:04d}"
        r=json.loads((path/'BAND_COMPLETE.json').read_text())
        assert sha(path/'BAND_COMPLETE.json')==item['receipt_sha256']
        band_info.append((r['comparison_row_start'],r['comparison_row_stop'],path))
    native_hist=np.zeros(3,dtype=np.int64)
    shared_hist=np.zeros(3,dtype=np.int64)
    minimum,maximum=255,0
    with rasterio.open(receipt['native_path']) as native, rasterio.open(receipt['shared_path']) as shared:
        assert (native.height,native.width,native.count)==(height,width,1)
        assert (shared.height,shared.width,shared.count)==(height,width,1)
        for y0 in range(0,height,256):
            y1=min(height,y0+256)
            score=np.zeros((2,y1-y0,width),dtype=np.float32)
            count=np.zeros((y1-y0,width),dtype=np.uint8)
            for by0,by1,path in band_info:
                begin=max(y0,by0)
                end=min(y1,by1)
                if begin>=end:
                    continue
                s=np.load(path/'score-sum.npy',mmap_mode='r',allow_pickle=False)
                c=np.load(path/'patch-count.npy',mmap_mode='r',allow_pickle=False)
                score[:,begin-y0:end-y0,:]+=s[:,begin-by0:end-by0,:]
                count[begin-y0:end-y0,:]+=c[begin-by0:end-by0,:]
            minimum=min(minimum,int(count.min()))
            maximum=max(maximum,int(count.max()))
            assert count.min()>0
            expected=(np.argmax(score,axis=0)+1).astype(np.uint8)
            clipped=np.where(support[y0:y1,:]>0,expected,0).astype(np.uint8)
            window=Window(0,y0,width,y1-y0)
            n=native.read(1,window=window)
            c=shared.read(1,window=window)
            assert np.array_equal(n,expected)
            assert np.array_equal(c,clipped)
            native_hist+=np.bincount(n.ravel(),minlength=3)
            shared_hist+=np.bincount(c.ravel(),minlength=3)
    assert native_hist.tolist()==receipt['native_histogram_0_to_2']
    assert shared_hist.tolist()==receipt['shared_histogram_0_to_2']
    assert minimum==receipt['min_patch_coverage']
    assert maximum==receipt['max_patch_coverage']
    record={'status':'VERIFIED','format':'cam-binary-network-masks-exact-pixel-v1',
            'method':method,'slide':slide,'shape_yx':[height,width],
            'native_sha256':receipt['native_sha256'],
            'shared_sha256':receipt['shared_sha256'],
            'native_histogram_0_to_2':native_hist.tolist(),
            'shared_histogram_0_to_2':shared_hist.tolist(),
            'support_sha256':support_entry['support_sha256'],
            'band_audit_sha256':sha(audit_path),
            'assembly_receipt_sha256':sha(receipt_path),
            'all_pixels_recomputed_from_audited_bands':True,
            'shared_support_exact':True,'test_reference_opened':False,
            'online_publication':False}
    path=folder/'MASKS_AUDIT_VERIFIED.json'
    partial=path.with_name(path.name+'.partial')
    partial.write_text(json.dumps(record,indent=2)+'\n')
    os.replace(partial,path)
    print(json.dumps({'status':'VERIFIED','method':method,'slide':slide,
                      'audit_sha256':sha(path)}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=('unet','deeplabv3plus'),required=True)
    p.add_argument('--slide',choices=('tumor_084','tumor_091','normal_019'),required=True)
    a=p.parse_args()
    main(a.method,a.slide)
