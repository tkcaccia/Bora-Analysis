"""Keep anatomical region identity separate from its ranked tissue identity."""
import numpy as np


def semantic_mask(regions,rankings):
    regions=np.asarray(regions);rankings=np.asarray(rankings)
    if rankings.ndim!=2 or len(rankings)!=int(regions.max()) or np.any(regions<0):
        raise ValueError('Exactly one ranking for every frozen positive region ID required')
    if np.any(rankings<1) or np.any(rankings>255):raise ValueError('Registered positive tissue IDs required')
    mask=np.zeros(regions.shape,np.uint8);tissue=regions>0
    mask[tissue]=rankings[:,0][regions[tissue]-1]
    return mask
