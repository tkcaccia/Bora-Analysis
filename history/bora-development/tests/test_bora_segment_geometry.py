import numpy as np
from experiments.bora_segment.geometry import components,overlap,token_evidence,sanitize,grouping_observations
from experiments.bora_segment.targets import target_counts
from experiments.bora_segment.context import boundary_context


def test_segments_are_connected_components_with_group_identity_preserved():
    coarse=np.array([[1,1,0,2],[1,0,0,2],[0,0,1,2]],np.uint8)
    regions,groups=components(coarse)
    assert groups.tolist()==[1,1,2]
    assert np.all(regions[coarse==0]==0)
    for i,g in enumerate(groups,1):assert np.all(coarse[regions==i]==g)


def test_clipped_edge_pixels_cannot_define_a_physical_tissue_group():
    boxes=np.array([[0,0,90,90],[90,0,11,1],[0,90,72,90],[90,90,90,90]])
    assert grouping_observations(boxes,[1,1,1,.49],[.5,.5]).tolist()==[True,False,True,False]
    # Admission follows micrometers, including anisotropic acquisition.
    assert grouping_observations(np.array([[0,0,36,72]]),[1],[1,.5]).tolist()==[True]


def test_native_area_pooling_and_tile_overlap_conserve_every_region_pixel():
    regions=np.array([[1,1,1,2],[1,1,1,2]],np.int32)
    tokens=np.array([[[1.,3.],[5.,7.]]],np.float32)
    mean,std,mass,weights=token_evidence(tokens,regions,(14,28))
    np.testing.assert_allclose(mean,[[7/3,13/3],[5,7]],rtol=1e-6)
    np.testing.assert_allclose(mass,[6,2])
    assert np.all(std[0]>0) and np.all(std[1]==0)
    tiles=np.array([[0,0,1,1],[0,0,1,1]])
    matrix=overlap(regions,tiles,2)
    np.testing.assert_array_equal(matrix.toarray(),[[4,2],[0,2]])
    np.testing.assert_array_equal(np.asarray(matrix.sum(1)).ravel(),mass)


def test_geometry_worker_has_no_fine_reference_or_semantic_vocabulary():
    row=dict(case='a',original_path='/data/a/original.png',original_sha256='image',
        feature=dict(path='/data/a/base.npy',sha256='features'),
        tissue_support_path='/data/a/tissue.working.tif',
        sample=dict(working_mpp_xy=[.5,.5],native_mpp_xy=[.25,.25],native_shape_yx=[100,200]),
        reference_path='/secret/fine.tif',ontology_raw_to_role={'1':1},class_names={'1':'tumor'})
    allowed=vars(sanitize(row,'/data/registry'))
    assert 'reference_path' not in allowed and 'ontology_raw_to_role' not in allowed and 'class_names' not in allowed
    assert not any('secret' in str(v) for v in allowed.values())


def test_region_targets_describe_annotated_region_area_and_never_make_glass_a_class():
    regions=np.array([[0,1,1],[2,2,2]],np.int32);roles=np.array([[1,1,0],[2,2,1]],np.uint8)
    fractions,coverage=target_counts(regions,roles,2)
    np.testing.assert_allclose(fractions,[[1,0],[1/3,2/3]])
    np.testing.assert_allclose(coverage,[.5,1.])


def test_shared_boundary_context_does_not_merge_or_rename_regions():
    regions=np.array([[1,1,2],[1,3,2]],np.int32)
    means=np.array([[1.],[5.],[9.]])
    context=boundary_context(regions,means,[2.,1.])
    np.testing.assert_array_equal(context[:,:1],means)
    # Region1 shares one length1 edge with2 and length1+length2 with3.
    assert context[0,1]==1-(5+3*9)/4
    np.testing.assert_array_equal(boundary_context(np.ones((3,4),np.int32),np.array([[7.]]),[1,1]),[[7,0]])


def test_training_collection_never_opens_validation_or_test_targets(tmp_path):
    from experiments.bora_segment.train import collect
    geo=tmp_path/'geometry';targets=tmp_path/'targets'
    for case,label in [('train_a',0),('train_b',1)]:
        a=geo/'cases'/case;b=targets/'cases'/case;a.mkdir(parents=True);b.mkdir(parents=True)
        np.save(a/'mean.npy',np.array([[1.+label,4.-label]]))
        np.savez(b/'targets.npz',fractions=np.eye(2)[[label]],coverage=[1.])
    registry={'cases':[dict(case=c,split=s,sample={'native_mpp_xy':[.5,.5]})
        for c,s in [('train_a','train'),('train_b','train'),('missing_val','validation'),('missing_test','test')]]}
    x,y,classes,report=collect(registry,geo,targets,'mean','hard')
    assert x.shape==(2,2) and y.tolist()==[1,2] and classes.tolist()==[1,2]
    assert {r['case'] for r in report['selected']}=={'train_a','train_b'}


def test_semantic_assignment_preserves_region_identity_and_background():
    import pytest
    from experiments.bora_segment.labels import semantic_mask
    regions=np.array([[0,1,1,2],[3,3,2,2]],np.int32)
    original=regions.copy();ranks=np.array([[3,1],[3,2],[1,3]])
    labels=semantic_mask(regions,ranks)
    np.testing.assert_array_equal(labels,[[0,3,3,3],[1,1,3,3]])
    np.testing.assert_array_equal(regions,original)
    # Adjacent regions may share a semantic class but keep distinct IDs.
    assert set(np.unique(regions))=={0,1,2,3}
    with pytest.raises(ValueError):semantic_mask(regions,ranks[:2])
