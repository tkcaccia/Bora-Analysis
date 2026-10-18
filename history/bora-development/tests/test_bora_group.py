import numpy as np
import pytest
from experiments.bora_segment.coarse_example import anonymize
from experiments.bora_segment.train import target_labels
from experiments.bora_group.geometry import compact_groups,sanitize
from experiments.bora_supervised.registry import save
from experiments.bora_segment.predict import validate_prediction_unit


def test_anonymization_preserves_every_assignment_and_ignored_pixel():
    source=np.array([[0,2,2,9],[9,9,0,2]],np.uint8)
    anonymous,restore=anonymize(source,'image-hash')
    np.testing.assert_array_equal(restore[anonymous],source)
    np.testing.assert_array_equal(anonymous==0,source==0)
    assert set(np.unique(anonymous))=={0,1,2}
    for a in range(source.size):
        for b in range(source.size):assert (source.flat[a]==source.flat[b])==(anonymous.flat[a]==anonymous.flat[b])


def test_disconnected_pieces_remain_one_group_and_empty_groups_are_recordable():
    anonymous=np.array([[2,0,5],[0,2,0],[5,0,2]],np.uint8)
    internal,surviving=compact_groups(anonymous)
    assert surviving.tolist()==[2,5]
    assert internal.max()==2 and np.all(internal[anonymous==2]==1)
    np.testing.assert_array_equal(np.r_[0,surviving][internal],anonymous)


def test_known_group_identity_is_distinct_from_post_refinement_pixel_majority():
    fractions=np.array([[.2,.8],[1.,0],[.7,.3],[0,0]])
    truth,keep=target_labels(fractions,[.1,1,1,0],identity=[1,2,0,1])
    assert list(truth)==[1,2,0,1] and keep.tolist()==[True,True,False,False]
    fragment_truth,fragment_keep=target_labels(fractions,[.1,1,1,0])
    assert fragment_truth.tolist()==[2,1,1,1] and fragment_keep.tolist()==[False,True,True,False]


def test_group_geometry_worker_receives_no_reference_or_tissue_names(tmp_path):
    folder=tmp_path/'case';folder.mkdir()
    save(folder/'INPUTS_FROZEN.json',{'files_sha256':{'ensemble/representation.png':'rgb'}})
    row=dict(case='case',original_path=str(folder/'original.png'),original_sha256='image',
        feature=dict(path='/features/base.npy',sha256='tokens'),tissue_support_path='/tissue.tif',
        sample=dict(coarse_sha256='coarse',working_mpp_xy=[.5,.5],native_mpp_xy=[.25,.25],native_shape_yx=[100,200],
                    class_names={'1':'tumor','2':'stroma'}),reference_path='/secret/fine.tif',ontology_raw_to_role={'1':1})
    worker=vars(sanitize(row))
    assert 'reference_path' not in worker and 'class_names' not in worker and 'ontology_raw_to_role' not in worker
    assert not any('secret' in str(value) for value in worker.values())


def test_head_requires_matching_units_and_frozen_geometry():
    unit='Whole anonymous Bora group, including all disconnected pieces'
    record=dict(prediction_unit=unit,fine_annotation_read=False,new_clustering_used=False,
        previous_tile_assignments_preserved_exactly=True,old_reference_derived_coarse_or_rgb_read=True)
    assert validate_prediction_unit(record,dict(training_unit=unit))==unit
    with pytest.raises(ValueError,match='historical region head'):
        validate_prediction_unit(record,dict(training_unit='Connected Bora region'))
    with pytest.raises(ValueError,match='fine annotations'):
        validate_prediction_unit(dict(record,fine_annotation_read=True),dict(training_unit=unit))
    with pytest.raises(ValueError,match='preserve the previous tiling'):
        validate_prediction_unit(dict(record,new_clustering_used=True),dict(training_unit=unit))
