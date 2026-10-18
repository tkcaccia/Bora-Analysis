import copy
import numpy as np
import pytest
import tifffile
from experiments.bora_cycles.audit_twelve_revision import compare_case
from experiments.bora_cycles.paired_eight import SCORE_CORE


def fixture(tmp_path):
    old,new=tmp_path/'old',tmp_path/'new';score={k:.7 for k in SCORE_CORE}
    row=dict(sample={'mpp':[.5,.5]},methods={k:dict(status='ok',score=copy.deepcopy(score)) for k in ('J01','Q12')})
    manifest=dict(input_manifest_sha256='input',rgb_sha256='rgb',files_sha256={'J01/support.native.tif':'unused-by-helper'})
    for root in (old,new):
        for k in ('J01','Q12'):
            (root/k).mkdir(parents=True);tifffile.imwrite(root/k/'native.tif',np.array([[1,2],[0,2]],np.uint8))
        tifffile.imwrite(root/'J01/support.native.tif',np.array([[1,1],[0,1]],np.uint8))
    return new,old,copy.deepcopy(row),row,copy.deepcopy(manifest),manifest,['J01','Q12']


def test_only_explicit_q12_repair_can_change_labels_and_accuracy(tmp_path):
    args=fixture(tmp_path);tifffile.imwrite(args[0]/'Q12/native.tif',np.array([[2,2],[0,2]],np.uint8));args[2]['methods']['Q12']['score']['macro_iou']=.8
    counts,changed=compare_case(*args)
    assert changed==1 and counts['unchanged_native_masks']==counts['unchanged_accuracy_cores']==counts['native_supports']==1


@pytest.mark.parametrize('change',['mask','support','score','rgb','geometry'])
def test_aggregate_agreement_cannot_hide_unexpected_changes(tmp_path,change):
    args=fixture(tmp_path)
    if change in ('mask','support'):
        name='native.tif' if change=='mask' else 'support.native.tif';tifffile.imwrite(args[0]/'J01'/name,np.ones((2,2),np.uint8))
    elif change=='score':args[2]['methods']['J01']['score']['macro_iou']=.8
    elif change=='rgb':args[4]['rgb_sha256']='other'
    else:args[2]['sample']['mpp']=[1,1]
    with pytest.raises(ValueError,match='differ|changed'):compare_case(*args)
