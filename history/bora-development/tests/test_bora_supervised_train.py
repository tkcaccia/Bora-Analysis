import numpy as np
import pytest

from experiments.bora_supervised.registry import save,sha
from experiments.bora_supervised.train import (collect_training,export_linear_responses,
                                               score_units,verify_preparation)


def fixture_registry(root):
    registry={'cases':[{'case':c,'split':s} for c,s in
                       [('learn','train'),('validate','validation'),('sealed','test')]]}
    save(root/'REGISTRY.json',registry)
    records={}
    for row in registry['cases']:
        folder=root/'cases'/row['case'];folder.mkdir(parents=True)
        np.save(folder/'mean.npy',np.array([[1.,0.],[0.,1.],[.6,.4]]))
        np.save(folder/'centre.npy',np.array([[1.,0.],[0.,1.],[.6,.4]]))
        np.savez(folder/'units.npz',tissue_fraction=np.ones(3))
        names=['mean.npy','centre.npy','units.npz']
        annotated=row['split']!='test'
        if annotated:
            np.savez(folder/'targets.npz',fractions=np.array([[1.,0.],[0.,1.],[.6,.4]]),coverage=np.ones(3))
            names+=['targets.npz']
        save(folder/'PREPARED.json',{'registry_sha256':sha(root/'REGISTRY.json'),
             'reference_decoded':annotated,'files_sha256':{n:sha(folder/n) for n in names}})
        records[row['case']]=sha(folder/'PREPARED.json')
    complete={'registry_sha256':sha(root/'REGISTRY.json'),'test_annotations_decoded':False,
              'case_preparation_sha256':records}
    return registry,complete


def test_fitting_is_independent_of_unavailable_held_out_targets(tmp_path):
    registry,_=fixture_registry(tmp_path)
    (tmp_path/'cases/validate/targets.npz').unlink()
    method={'feature':'mean','pure':False,'target':'hard'}
    x,y,classes,record=collect_training(tmp_path,registry,method)
    assert classes.tolist()==[1,2]
    assert set(record['eligible_units'])=={1,2}
    assert {r['case'] for r in record['selected_unit_ids']}=={'learn'}
    assert x.shape==(3,2) and sorted(y.tolist())==[1,1,2]


def test_preflight_rejects_changed_predictors_and_unsealed_test(tmp_path):
    registry,complete=fixture_registry(tmp_path)
    verify_preparation(tmp_path,registry,complete)
    np.save(tmp_path/'cases/sealed/mean.npy',np.ones((3,2)))
    with pytest.raises(ValueError,match='Frozen shard changed'):
        verify_preparation(tmp_path,registry,complete)


def test_missing_training_class_counts_as_error_not_dropped():
    scores=score_units(np.array([1,7]),np.array([[1,2,3,4,5],[1,2,3,4,5]]),np.arange(1,7))
    assert scores['top1']==.5 and scores['top5']==.5
    assert scores['balanced_accuracy']==.5 and scores['truth_roles_absent_from_training']==[7]
    assert scores['top5_is_trivial'] is False


def test_context_is_coordinate_calibrated_and_label_independent():
    from experiments.bora_supervised.cycle_two import context_features
    x=np.array([[1.,2.],[3.,4.],[9.,1.]])
    boxes=np.array([[0,0,40,40],[40,0,40,40],[80,0,40,40]])
    out=context_features(x,boxes,[1.,1.])
    same_physical=context_features(x,boxes*2,[.5,.5])
    np.testing.assert_allclose(out,same_physical)
    np.testing.assert_array_equal(out[:,:2],x)
    np.testing.assert_allclose(context_features(np.ones((3,2)),boxes,[1.,1.])[:,2:],0,atol=1e-14)


@pytest.mark.parametrize('fractional',[False,True])
def test_native_response_export_and_rank_parity(fractional):
    fastpls=pytest.importorskip('fastpls')
    rng=np.random.default_rng(7)
    x=rng.normal(size=(100,12));y=np.argmax(x[:,:6],axis=1)+1
    target=np.eye(6)[y-1] if fractional else y
    model=fastpls.PLS(n_components=4,classifier=None if fractional else 'argmax',backend='cpu').fit(x,target)
    coefficient,intercept=export_linear_responses(model,12)
    held=rng.normal(size=(41,12))
    native=model.predict_responses(held);exported=held@coefficient+intercept
    np.testing.assert_allclose(native,exported,rtol=1e-9,atol=1e-10)
    np.testing.assert_array_equal(np.argsort(-native,axis=1),np.argsort(-exported,axis=1))
