import json
from pathlib import Path
import numpy as np
import pytest

from experiments.bora_supervised.registry import (assign_groups,stable_split,tissue_ids,ONTOLOGY)
from experiments.bora_supervised.registry import validate_ontology_extension
from experiments.bora_supervised.units import tile_boxes,pool_tokens,region_targets,training_rows,ranked_labels


def sample(case,group,dataset='a',**source):
    return dict(case=case,group=group,dataset=dataset,source=source)


def test_known_patient_and_exact_duplicates_never_cross_splits():
    rows=[sample('a','TCGA-BH-A1EW'),sample('b','TCGA-BH-A1EW',dataset='b'),
          sample('c','other',image_sha256='same'),sample('d','different',dataset='c',image_sha256='same')]
    out=assign_groups(rows)
    assert out['a']['group']==out['b']['group'] and out['a']['split']==out['b']['split']
    assert out['c']['group']==out['d']['group'] and out['c']['split']==out['d']['split']


def test_adding_unrelated_groups_preserves_existing_split():
    a=[sample('a','donor1'),sample('b','donor2')]
    b=assign_groups(a+[sample('c','newdonor')])
    assert assign_groups(a)=={k:b[k] for k in ['a','b']}
    assert stable_split('a:donor1') in ['train','validation','test']


def test_provider_test_has_precedence_for_whole_group():
    out=assign_groups([sample('a','donor',split='train'),sample('b','donor',provider_split='test')])
    assert {r['split'] for r in out.values()}=={'test'}


def test_no_automatic_negative_or_residual_is_taught_as_stroma():
    ontology=json.loads(ONTOLOGY.read_text())
    assert tissue_ids({'1':'non-cancer background (not expert stroma)','2':'rest'},ontology)=={'1':0,'2':0}
    with pytest.raises(ValueError,match='Unreviewed'):tissue_ids({'1':'invented tissue'},ontology)


def test_ontology_can_append_roles_but_cannot_reassign_old_ids():
    old=json.loads(ONTOLOGY.read_text());new=json.loads(ONTOLOGY.read_text())
    new['roles'].append('new_reviewed_role');new['mapping']['new reviewed source label']='new_reviewed_role'
    validate_ontology_extension(old,new)
    new['roles'][0],new['roles'][1]=new['roles'][1],new['roles'][0]
    with pytest.raises(ValueError,match='stable role IDs'):validate_ontology_extension(old,new)
    new=json.loads(ONTOLOGY.read_text());new['mapping']['stroma']='tumor'
    with pytest.raises(ValueError,match='existing label meaning'):validate_ontology_extension(old,new)


def test_tiles_have_physical_extent_and_cover_image_once():
    boxes=tile_boxes((100,180),(.5,1.),45.)
    canvas=np.zeros((100,180),int)
    for x,y,w,h in boxes:canvas[y:y+h,x:x+w]+=1
    assert np.all(canvas==1)
    assert tuple(boxes[0])==(0,0,90,45)


def test_token_pooling_area_weights_edge_cells():
    tokens=np.arange(4,dtype=np.float32).reshape(2,2,1)
    mean,centre=pool_tokens(tokens,np.array([[1,0,3,4]]),2)
    assert mean[0,0]==pytest.approx((1*0+2*1+1*2+2*3)/6)
    assert centre[0,0]==3


def test_targets_ignore_zero_and_unmapped_roles_without_inventing_background():
    ref=np.array([[0,1,1,2],[0,1,3,2]])
    f,c=region_targets(ref,np.array([[0,0,4,2]]),(2,4),{'1':1,'2':2,'3':0},2)
    np.testing.assert_allclose(f,[[.6,.4]])
    assert c[0]==pytest.approx(5/8)


def test_purity_is_a_training_filter_not_a_prediction_input():
    f=np.array([[.7,.3],[.96,.04]]);coverage=np.ones(2);tissue=np.ones(2)
    assert training_rows(f,coverage,tissue).tolist()==[True,True]
    assert training_rows(f,coverage,tissue,pure=True).tolist()==[False,True]
    scores=np.array([[.1,.5,.2,.7,.6,.3]])
    assert ranked_labels(scores,np.arange(1,7),5).tolist()==[[4,5,2,6,3]]
    with pytest.raises(ValueError):ranked_labels([[np.nan]],np.array([1]))
