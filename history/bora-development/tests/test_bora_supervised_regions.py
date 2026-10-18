import numpy as np
from experiments.bora_supervised.regions import pool_regions,region_targets


def test_region_pooling_uses_area_and_keeps_full_features():
    tokens=np.array([[[0.,2.],[3.,5.]],[[6.,8.],[9.,11.]]])
    regions=np.array([[0,0,1,1],[0,0,1,1],[0,0,1,1],[0,0,1,1]],np.int32)
    out=pool_regions(tokens,regions,(4,4),patch=2)
    np.testing.assert_allclose(out,[[3.,5.],[6.,8.]])


def test_region_truth_is_exact_native_area_and_zero_stays_unscored():
    reference=np.array([[0,1,2,2],[1,1,2,0],[1,1,1,2],[1,1,2,2]],np.uint8)
    regions=np.array([[0,1],[0,1]],np.int32)
    f,c=region_targets(reference,regions,{'1':1,'2':2},2)
    np.testing.assert_allclose(f,[[1.,0.],[1/7,6/7]])
    np.testing.assert_allclose(c,[7/8,7/8])
