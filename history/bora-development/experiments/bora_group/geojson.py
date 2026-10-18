"""Superseded pixel-edge audit; final exports use bora_geojson.py and Bora's converter."""
import argparse
from pathlib import Path
import copy
import numpy as np
import tifffile
from rasterio.features import shapes,rasterize
from shapely.geometry import shape,mapping
from shapely import make_valid
from shapely.ops import unary_union
from experiments.bora_supervised.registry import load,save,sha
from experiments.bora_supervised.viewer import palette
from experiments.bora_segment.labels import semantic_mask
from . import anonymous_palette


def polygon_parts(geometry):
    if geometry.geom_type=='Polygon':return [geometry]
    if geometry.geom_type in ('MultiPolygon','GeometryCollection'):
        return [p for child in geometry.geoms for p in polygon_parts(child)]
    return []


def group_polygons(regions,information,mpp_xy):
    buckets={item['id']:[] for item in information};repaired=0
    for geometry,value in shapes(regions.astype(np.int32),mask=regions>0,connectivity=4):
        polygon=shape(geometry)
        if not polygon.is_valid:polygon=make_valid(polygon);repaired+=1
        buckets[int(value)].extend(polygon_parts(polygon))
    features=[];colours=anonymous_palette(max(item['anonymous_group'] for item in information))
    for item in information:
        polygon=unary_union(buckets[item['id']])
        if not polygon.is_valid:raise ValueError('Invalid native group polygon')
        if not np.isclose(polygon.area,item['area_pixels'],atol=1e-6,rtol=0):raise ValueError('Polygon area differs from native mask')
        g=item['anonymous_group'];name=chr(64+g) if g<=26 else str(g);c=colours[g].tolist()
        features.append(dict(type='Feature',id=item['id'],geometry=mapping(polygon),properties=dict(
            group_id=item['id'],anonymous_group=name,area_pixels=item['area_pixels'],area_um2=item['area_um2'],
            objectType='annotation',classification=dict(name='Anonymous '+name,color='#%02x%02x%02x'%tuple(c)),
            coordinate_space='native_image_pixel_edges')))
    result=dict(type='FeatureCollection',features=features,native_shape_yx=list(regions.shape),native_mpp_xy=list(mpp_xy),
        coordinate_space='native_image_pixel_edges',origin='top_left',minimum_area_pixels=0,simplification_tolerance_pixels=0,
        note='Image pixel coordinates, not geographic longitude and latitude.',invalid_components_repaired=repaired)
    restored=rasterize([(f['geometry'],f['properties']['group_id']) for f in features],out_shape=regions.shape,fill=0,dtype=np.int32)
    if not np.array_equal(restored,regions):raise ValueError('Native group GeoJSON round trip changed a pixel')
    return result


def classify_geojson(groups,ranks,roles,colours):
    result=copy.deepcopy(groups)
    for feature in result['features']:
        props=feature['properties'];role=int(ranks[props['group_id']-1,0]);name=roles[role-1].replace('_',' ')
        props.update(value=role,predicted_tissue_id=role,predicted_tissue=name,name=name,label=name,
            classification=dict(name=name,color='#%02x%02x%02x'%tuple(colours[role])),
            tissue_identity_source='Supervised prediction after Bora; no reference identity in this file')
    return result


def run(registry_root,geometry,cycle,output):
    registry_root,geometry,cycle,output=map(Path,(registry_root,geometry,cycle,output));registry=load(registry_root/'REGISTRY.json')
    frozen=load(cycle/'ALL_PREDICTIONS_FROZEN.json');roles=registry['ontology']['roles'];colours=palette(len(roles));records=[]
    models={name:load(cycle/name/'PREDICTIONS.json') for name in frozen['models']}
    for name,model in models.items():
        if sha(cycle/name/'PREDICTIONS.json')!=frozen['models'][name]:raise ValueError('Prediction manifest changed')
    for row in registry['cases']:
        if row['split']!=frozen['split']:continue
        case=row['case'];src=geometry/'cases'/case;manifest=load(src/'GEOMETRY_FROZEN.json')
        if sha(src/'regions.native.tif')!=manifest['files_sha256']['regions.native.tif']:raise ValueError('Refined geometry changed')
        regions=tifffile.imread(src/'regions.native.tif');information=load(src/'REGIONS.json')['regions']
        groups=group_polygons(regions,information,row['sample']['native_mpp_xy']);folder=output/case
        save(folder/'anonymous.geojson',groups);checks=[]
        for name,model in models.items():
            path=cycle/name/(case+'.npz');expected=next(p['sha256'] for p in model['predictions'] if p['case']==case)
            if sha(path)!=expected:raise ValueError('Frozen group ranks changed')
            with np.load(path) as p:ranks=p['top5']
            result=classify_geojson(groups,ranks,roles,colours);target=folder/(name+'.geojson');save(target,result)
            restored=rasterize([(f['geometry'],f['properties']['predicted_tissue_id']) for f in result['features']],
                out_shape=regions.shape,fill=0,dtype=np.uint8)
            mask=semantic_mask(regions,ranks)
            if not np.array_equal(restored,mask):raise ValueError('Named GeoJSON round trip changed a pixel')
            raster_path=cycle/name/(case+'.tif')
            if raster_path.exists():
                if not np.array_equal(tifffile.imread(raster_path),mask):raise ValueError('Semantic raster differs from frozen prediction')
            else:tifffile.imwrite(raster_path,mask,compression='deflate')
            checks.append(dict(model=name,geojson_sha256=sha(target),roundtrip_differing_pixels=0,features=len(result['features'])))
        records.append(dict(case=case,native_shape_yx=list(regions.shape),groups=len(information),
            group_roundtrip_differing_pixels=0,models=checks,all_polygons_valid=True))
    save(output/'GEOJSON_COMPLETE.json',dict(split=frozen['split'],cases=records,total_cases=len(records),
        prediction_manifest_sha256=sha(cycle/'ALL_PREDICTIONS_FROZEN.json'),native_pixel_coordinates=True,
        all_native_raster_roundtrips_exact=True,minimum_area_pixels=0,simplification_tolerance_pixels=0))
    print(__import__('json').dumps(dict(geojson_cases=len(records),models=len(models),all_roundtrips_exact=True)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('registry-root','geometry','cycle','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.registry_root,a.geometry,a.cycle,a.output)


if __name__=='__main__':main()
