"""Independent full-pixel confusion audit of the locked three-slide CAM pilot."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import rasterio

ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA=Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
FREEZE=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-15-v144/SCORING_FROZEN.json'
CTRL=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-15-v151/CONTROLLER_STATE.json'
OUT=DATA/'external-camelyon16-five-slide-2026-11-15-v144/test-scores'
HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def audit():
    if (HERE/'SCORE_AUDIT_VERIFIED.json').exists():
        raise FileExistsError('No in-place audit retry')
    state=json.loads(CTRL.read_text())
    frozen=json.loads(FREEZE.read_text())
    assert state['status']=='COMPLETE' and state['active'] is None
    assert len(state['completed'])==15
    assert not subprocess.check_output(['systemctl','--user','list-units','bora-*','--state=running','--no-legend','--no-pager'],text=True).strip()
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True).strip()
    completed={(x['slide'],x['method']):x for x in state['completed']}
    rows=[]
    for slide in frozen['test_ids']:
        spec=frozen['slides'][slide]
        shape=tuple(spec['shape_yx'])
        support=np.memmap(spec['support']['path'],mode='r',dtype=np.uint8,shape=shape)
        reference_state=json.loads(Path(spec['reference_receipt_path']).read_text())
        assert reference_state['status']=='COMPLETE'
        official=np.memmap(reference_state['aligned_path'],mode='r',dtype=np.uint8,shape=shape)
        for method,item in spec['methods'].items():
            path=OUT/method/slide/'SCORE_COMPLETE.json'
            result=json.loads(path.read_text())
            assert result['status']=='COMPLETE_LOCAL_CC0_NO_PUBLICATION'
            assert result['scoring_freeze_sha256']==sha(FREEZE)
            assert sha(path)==completed[(slide,method)]['receipt_sha256']
            assert result['reference_receipt_sha256']==sha(spec['reference_receipt_path'])
            assert result['native_mask_sha256']==item['native']['sha256']
            assert result['shared_support_mask_sha256']==item['shared_support']['sha256']
            native_path=item['native']['path'];shared_path=item['shared_support']['path']
            counts={k:np.zeros(4,dtype=np.int64) for k in ('native','shared_support')}
            known_pixels=0;positives=0;known_outside=0;positive_outside=0
            with rasterio.open(native_path) as nsrc,rasterio.open(shared_path) as ssrc:
                for _,window in nsrc.block_windows(1):
                    n=nsrc.read(1,window=window)
                    s=ssrc.read(1,window=window)
                    y0,x0=int(window.row_off),int(window.col_off)
                    y1,x1=y0+int(window.height),x0+int(window.width)
                    ref=official[y0:y1,x0:x1]
                    tissue=support[y0:y1,x0:x1]>0
                    valid=ref>0;actual=ref==2
                    assert np.all(s[~tissue]==0)
                    known_pixels+=int(valid.sum());positives+=int((actual&valid).sum())
                    known_outside+=int((valid&~tissue).sum())
                    positive_outside+=int((actual&valid&~tissue).sum())
                    for name,pred in (('native',n),('shared_support',s)):
                        guess=pred==item['positive_class_code']
                        tp=int((valid&actual&guess).sum())
                        fp=int((valid&~actual&guess).sum())
                        fn=int((valid&actual&~guess).sum())
                        tn=int((valid&~actual&~guess).sum())
                        counts[name]+=np.array([tp,fp,fn,tn],dtype=np.int64)
            metrics=result['metrics']
            assert metrics['scored_known_pixels']==known_pixels
            assert metrics['annotated_pixels_outside_reviewed_support']==known_outside
            assert metrics['reference_positive_pixels_outside_reviewed_support']==positive_outside
            for variant in counts:
                c=counts[variant]
                m=metrics[variant]
                assert [m[k] for k in ('tp','fp','fn','tn')]==c.tolist()
                assert int(c.sum())==known_pixels
                assert m['reference_positive_pixels']==positives
                assert m['boundary']['f1'] is None or 0<=m['boundary']['f1']<=1
                if positives:
                    assert abs(m['precision']-(c[0]/(c[0]+c[1]) if c[0]+c[1] else 0))<1e-12
                    assert abs(m['recall']-c[0]/positives)<1e-12
                    assert abs(m['dice']-2*c[0]/(2*c[0]+c[1]+c[2]))<1e-12
                    assert abs(m['iou']-c[0]/(c[0]+c[1]+c[2]))<1e-12
                else:
                    assert m['iou'] is None and m['dice'] is None
            rows.append({'slide':slide,'method':method,'role':item['role'],
                         'known_pixels':known_pixels,'reference_metastasis_pixels':positives,
                         'known_pixels_outside_support':known_outside,
                         'reference_metastasis_pixels_outside_support':positive_outside,
                         'native_confusion_tp_fp_fn_tn':counts['native'].tolist(),
                         'shared_confusion_tp_fp_fn_tn':counts['shared_support'].tolist(),
                         'native_iou':metrics['native']['iou'],
                         'shared_iou':metrics['shared_support']['iou'],
                         'native_false_positive_tissue_mm2':metrics['native']['false_positive_tissue_area_mm2'],
                         'shared_false_positive_tissue_mm2':metrics['shared_support']['false_positive_tissue_area_mm2'],
                         'score_receipt_sha256':sha(path)})
    assert len(rows)==15
    result={'status':'ALL_15_LOCAL_CAM_TEST_SCORES_INDEPENDENTLY_PIXEL_AUDITED',
            'scoring_freeze_sha256':sha(FREEZE),'controller_state_sha256':sha(CTRL),
            'rows':rows,'completed':15,'selected':15,
            'patient_center_disjointness':'unverified',
            'technical_pilot_only':True,'online_publication':False}
    path=HERE/'SCORE_AUDIT_VERIFIED.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'receipt_sha256':sha(path)}),flush=True)


if __name__=='__main__':audit()
