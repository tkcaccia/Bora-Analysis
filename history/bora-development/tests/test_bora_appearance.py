import cv2
import numpy as np
import pytest
from bora.appearance import refine_appearance
from bora.evidence import Evidence
from experiments.bora_cycles.cycle_six import edge_appearance
from experiments.bora_cycles.cycle_ten import bending_field


@pytest.mark.parametrize('method',['af','vaf'])
def test_named_api_matches_frozen_experimental_algorithm(method):
    rng=np.random.default_rng(17);rgb=rng.integers(0,256,(96,112,3),dtype=np.uint8)
    coarse=np.ones((96,112),np.uint8);coarse[:,56:]=2;coarse[48:,:28]=3
    tissue=np.ones(coarse.shape,bool);tissue[:3,:]=False
    c=Evidence(rgb,coarse,tissue,[.5,.5])
    q,_=edge_appearance(c,None,rgb) if method=='af' else bending_field(c,None,rgb,'logvariance')
    support=np.ones((103,127),bool);support[:4]=False
    expected=cv2.resize(c.ids[q.argmax(-1)].astype(np.uint8),support.shape[::-1],interpolation=cv2.INTER_NEAREST)
    expected[~support]=0
    actual,details=refine_appearance(rgb,coarse,tissue,[.5,.5],he=rgb,method=method,
        output_shape_yx=support.shape,output_tissue=support)
    np.testing.assert_array_equal(actual,expected)
    assert details['fine_annotation_used'] is False and not details['hard_anchors']
    assert max(f['linear_system_max_residual'] for f in details['fits'])<1e-8


def test_single_class_survives_and_background_is_excluded_last():
    image=np.full((25,30,3),100,np.uint8);coarse=np.full((25,30),7,np.uint8)
    support=np.ones((25,30),bool);support[:5]=False
    out,_=refine_appearance(image,coarse,support,[.5,.5],he=image)
    assert np.all(out[support]==7) and np.all(out[~support]==0)
    with pytest.raises(ValueError,match='output tissue'):
        refine_appearance(image,coarse,support,[.5,.5],he=image,output_shape_yx=(50,60))
