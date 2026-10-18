import numpy as np
import pytest
pytest.importorskip('rasterio');pytest.importorskip('shapely')
from rasterio.features import rasterize
from shapely.geometry import shape
from experiments.bora_group.geojson import group_polygons,classify_geojson
from experiments.bora_supervised.viewer import palette
from experiments.bora_segment.labels import semantic_mask


def info(mask):
    return [dict(id=g,anonymous_group=g,area_pixels=int(np.count_nonzero(mask==g)),
                 area_um2=float(np.count_nonzero(mask==g))*.25) for g in range(1,int(mask.max())+1)]


def test_geojson_preserves_holes_disconnected_pieces_and_single_pixels():
    mask=np.ones((9,10),np.int32);mask[2:7,2:8]=2;mask[3:6,3:7]=0;mask[0,0]=2;mask[-1,-1]=2
    data=group_polygons(mask,info(mask),[.5,.5])
    assert len(data['features'])==2
    assert all(shape(f['geometry']).is_valid for f in data['features'])
    restored=rasterize([(f['geometry'],f['properties']['group_id']) for f in data['features']],
        out_shape=mask.shape,fill=0,dtype=np.int32)
    np.testing.assert_array_equal(restored,mask)
    assert sum(shape(f['geometry']).area for f in data['features'])==np.count_nonzero(mask)


def test_named_geojson_is_one_feature_per_group_and_has_predicted_names_only():
    mask=np.array([[1,0,2],[0,1,2],[2,2,1]],np.int32);ranks=np.array([[3,1],[1,3]])
    groups=group_polygons(mask,info(mask),[.5,.5]);data=classify_geojson(groups,ranks,['tumor','unused','stroma'],palette(3))
    assert [f['properties']['predicted_tissue'] for f in data['features']]==['stroma','tumor']
    assert all('reference' not in key for f in data['features'] for key in f['properties'])
    restored=rasterize([(f['geometry'],f['properties']['predicted_tissue_id']) for f in data['features']],
        out_shape=mask.shape,fill=0,dtype=np.uint8)
    np.testing.assert_array_equal(restored,semantic_mask(mask,ranks))
    assert groups['features'][0]['properties']['classification']['name']=='Anonymous A'
