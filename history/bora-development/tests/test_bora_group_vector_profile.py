import json
import numpy as np
import pytest
pytest.importorskip('rasterio');pytest.importorskip('shapely')
import tifffile
from shapely.geometry import shape
from bora.multicpu_geojson import convert
from experiments.bora_group.bora_geojson import PROFILE,annotate,geometry_stats,render
from experiments.bora_group.viewer import expert_legend
from experiments.bora_supervised.viewer import palette


def test_primary_vector_uses_published_profile_with_lossy_simplification(tmp_path):
    yy,xx=np.mgrid[:120,:140]
    mask=np.where((xx-70)**2+(yy-60)**2<35**2,1,2).astype(np.uint8)
    mask[2,2]=3
    path=tmp_path/'mask.tif';target=tmp_path/'mask.geojson';tifffile.imwrite(path,mask)
    convert(path,target,**PROFILE)
    result=json.loads(target.read_text());original=json.loads(target.read_text())
    assert {f['properties']['value'] for f in result['features']}=={1,2}
    annotate(result,['tumor','stroma','small'],palette(3),[.5,.5],mask.shape)
    assert [f['geometry'] for f in result['features']]==[f['geometry'] for f in original['features']]
    gs=[shape(f['geometry']) for f in result['features']]
    assert all(g.is_valid for g in gs) and gs[0].intersection(gs[1]).area==0
    restored=render(result,mask.shape)
    assert np.count_nonzero(restored!=mask)>0
    assert np.count_nonzero(restored!=mask)<mask.size*.02
    assert geometry_stats(result['features'])['vertices']<150
    assert result['pixel_exact'] is False


def test_expert_legend_covers_observed_source_classes_and_marks_unmapped():
    row=dict(ontology_raw_to_role={'2':1,'9':2},sample=dict(class_names={'2':'invasive tumor','9':'stroma','15':'unknown'}))
    entries=expert_legend(row,np.array([[0,2,9],[15,9,2]]))
    assert [e['source_name'] for e in entries]==['invasive tumor','stroma','unknown']
    assert [e['scored'] for e in entries]==[True,True,False]
