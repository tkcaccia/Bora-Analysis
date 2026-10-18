"""Functional native integration checks, leakage boundaries and lineage rules."""
import numpy as np
import pytest
from experiments.bora_evolution.catalog import initial,EVOLUTION_DIRECTIONS
from experiments.bora_evolution.evolve import children,instantiate,folds
from experiments.bora_evolution.smoke import Synthetic


def test_puma_evaluation_exclusion_does_not_relabel_or_promote_training():
    from experiments.bora_evolution.data import evaluation_cases
    rows={
        'puma_test':dict(dataset='puma',split='test',ontology_raw_to_role={'1':1,'2':2}),
        'puma_val':dict(dataset='puma',split='validation',ontology_raw_to_role={'1':1,'2':2}),
        'bcss_train':dict(dataset='bcss',split='train',ontology_raw_to_role={'1':1,'2':2}),
        'bcss_test':dict(dataset='bcss',split='test',ontology_raw_to_role={'1':1,'2':2}),
        'tiger_val':dict(dataset='tiger',split='validation',ontology_raw_to_role={'1':1,'2':2}),
        'small':dict(dataset='gcss',split='test',ontology_raw_to_role={'1':1})}
    assert evaluation_cases(rows,dict(excluded_datasets=['puma']))==['bcss_test','tiger_val']
    assert rows['bcss_train']['split']=='train' and rows['puma_test']['split']=='test'


def test_initial_mechanisms_and_five_parents_fifteen_children():
    specs=initial();assert len(specs)==20
    assert len({(s['engine'],s['view']) for s in specs})==20
    descendants=children(specs[:5],2)
    assert len(descendants)==15 and len({s['id'] for s in descendants})==15
    for p in specs[:5]:assert sum(s['parents']==[p['id']] for s in descendants)==3
    assert len({s['operator'] for s in descendants})>=10
    assert np.array_equal(folds(np.array(['a','b','a'])),folds(np.array(['a','b','a'])))


@pytest.mark.parametrize('operator',[op for op,_ in EVOLUTION_DIRECTIONS])
def test_each_structural_operator_runs_native_and_predicts_without_eval_targets(operator):
    pytest.importorskip('fastpls')
    d=Synthetic();parent=initial()[0]
    recipe=dict(id='test',operator=operator,parent_recipe=parent,view='mean',engine='evolved_native_pls',depth=1)
    model=instantiate(recipe,d);prediction=model.predict_case('s55')
    assert prediction.shape==(2,4) and np.isfinite(prediction).all()
    assert model.heads>=1


@pytest.mark.parametrize('spec',initial(),ids=lambda s:s['id'])
def test_native_mechanisms_tolerate_class_missing_from_source_fold(spec):
    pytest.importorskip('fastpls')
    d=Synthetic();d.labels[d.labels==4]=3;d.fractions=np.eye(4)[d.labels-1]
    m=instantiate(spec,d);s=m.predict_case('s55')
    assert s.shape==(2,4) and np.isfinite(s).all()
