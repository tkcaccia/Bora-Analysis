import numpy as np
from experiments.bora_supervised.evaluate import pixel_scores
from experiments.bora_supervised.eligibility import eligible


def test_single_role_and_automatic_complement_do_not_pass_multiclass_rule():
    assert not eligible({'ontology_raw_to_role':{'1':1,'2':1,'3':1}})
    assert not eligible({'ontology_raw_to_role':{'1':1,'2':0}})
    assert eligible({'ontology_raw_to_role':{'1':1,'2':3,'3':0}})


def test_unannotated_pixels_are_ignored_but_deleting_annotated_tissue_is_error():
    truth=np.array([[1,2,0],[1,2,0]],np.uint8)
    prediction=np.array([[1,2,3],[0,2,3]],np.uint8)
    s=pixel_scores(truth,prediction,3)
    assert s['pixel_top1']==.75 and s['annotated_unassigned_pixels']==1
    assert s['macro_iou']==.75 and '3' not in s['per_role']


def test_spurious_role_inside_annotated_tissue_penalizes_macro_iou():
    truth=np.array([[1,2]],np.uint8);prediction=np.array([[3,2]],np.uint8)
    s=pixel_scores(truth,prediction,3)
    assert s['macro_iou']==1/3 and s['gt_role_macro_iou']==.5
