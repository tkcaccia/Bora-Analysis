"""Technical controls for the optional WSI color-drift step."""
import importlib.util
from pathlib import Path

import numpy as np
import json
import pytest

module = Path(__file__).resolve().parents[2]/'pipeline/wsi_color_preprocess.py'
spec = importlib.util.spec_from_file_location('wsi_color_preprocess', module)
wsi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wsi)


def phantom():
    h,w=768,864
    a=np.broadcast_to(np.array([232,234,242],np.uint8),(h,w,3)).copy()
    yy,xx=np.indices((h,w))
    tissue=(xx>240)&(xx<620)&(yy>140)&(yy<640)
    texture=((xx//7+yy//9)%2)[tissue]
    a[tissue]=np.array([178,115,166])-texture[:,None]*np.array([38,45,16])
    return a,tissue


def test_spatial_field_recovery_preserves_tissue_and_crop_coordinates():
    clean,tissue=phantom();h,w=clean.shape[:2]
    y,x=np.indices((h,w));x=x/(w-1)-.5;y=y/(h-1)-.5
    gain=np.stack((np.ones_like(x),1+.02*x-.025*y,1+.012*x-.022*y),axis=2)
    shaded=np.rint(clean*gain).astype(np.uint8);source=shaded.copy()
    model,glass,_=wsi.fit_glass_plane(shaded)
    assert model['decision']=='APPLY'
    assert [r['channel'] for r in model['channels'] if r['apply']]==['G','B']
    corrected=wsi.correct_region(shaded,model)
    # A median glass anchor preserves global exposure, so compare after one
    # per-channel scalar rather than forcing a stain reference.
    scale=np.array([r['anchor'] for r in model['channels']])/np.array([232,234,242])
    expected=np.rint(clean*scale).clip(0,255)
    assert np.mean(abs(corrected[tissue].astype(float)-expected[tissue]))<.6
    assert np.mean(abs(corrected[glass].astype(float)-expected[glass]))<.6
    for x0,y0,x1,y1 in ((31,91,602,508),(0,0,123,180),(380,420,864,768)):
        crop=wsi.correct_region(shaded[y0:y1,x0:x1],model,(x0,y0),(h,w))
        assert np.array_equal(crop,corrected[y0:y1,x0:x1])
    assert np.array_equal(source,shaded)


def test_no_effect_and_no_glass_do_not_change_pixels():
    clean,_=phantom()
    model,_,_=wsi.fit_glass_plane(clean)
    assert model['decision']=='SKIP'
    assert np.array_equal(wsi.correct_region(clean,model),clean)
    no_glass=np.full((768,864,3),[125,80,130],np.uint8)
    model,_,_=wsi.fit_glass_plane(no_glass)
    assert model['decision']=='SKIP'
    assert np.array_equal(wsi.correct_region(no_glass,model),no_glass)


def test_clipped_glass_blocks_inference_of_unobserved_field():
    a,_=phantom();a[:]=255
    model,_,_=wsi.fit_glass_plane(a)
    assert model['decision']=='SKIP'
    assert model['clipped_glass_fraction']==1


def test_encoder_rejects_original_cache_and_changed_spacing(tmp_path):
    spec=importlib.util.spec_from_file_location('color_encoder',module.with_name('encode_virchow2.py'))
    encoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(encoder)
    image=tmp_path/'corrected.tif';receipt=tmp_path/'PREPROCESS_COMPLETE.json'
    value=dict(format=wsi.VERSION,status='COMPLETE',output_image=str(image),
               output_sha256='a'*64,source_sha256='b'*64,shape_yx=[768,864],
               mpp_xy=[.27,.28],decision='APPLY',correction_written=True)
    plan=dict(image_size_wh=[864,768],source_mpp_xy=[.27,.28])
    receipt.write_text(json.dumps(value))
    assert encoder.checked_preprocessing(receipt,image,'a'*64,plan)['original_image_sha256']=='b'*64
    with pytest.raises(ValueError):encoder.checked_preprocessing(receipt,image,'b'*64,plan)
    with pytest.raises(ValueError):encoder.checked_preprocessing(receipt,tmp_path/'original.tif','a'*64,plan)
    for bad in ([.5,.5],[],[float('nan'),.28]):
        value['mpp_xy']=bad;receipt.write_text(json.dumps(value))
        with pytest.raises(ValueError):encoder.checked_preprocessing(receipt,image,'a'*64,plan)
