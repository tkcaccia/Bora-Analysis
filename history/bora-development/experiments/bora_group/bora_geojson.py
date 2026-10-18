"""Use Bora's published coverage converter for final benchmark GeoJSON."""
import argparse
import csv
from pathlib import Path
import time
import numpy as np
import tifffile
from rasterio.features import rasterize,shapes
from shapely.geometry import shape
from shapely import get_num_coordinates
from bora.multicpu_geojson import convert
from experiments.bora_supervised.registry import load,save,sha
from experiments.bora_supervised.eligibility import eligible
from experiments.bora_supervised.evaluate import broad_truth,pixel_scores
from experiments.bora_supervised.viewer import palette
from experiments.bora_segment.labels import semantic_mask

PROFILE=dict(page=0,min_area=500.,simplify=2.,workers=1,striped_native=False)


def geometry_stats(features):
    parts=vertices=holes=0
    for f in features:
        g=shape(f['geometry'])
        if not g.is_valid:raise ValueError('Invalid exported polygon')
        polygons=[g] if g.geom_type=='Polygon' else list(g.geoms)
        parts+=len(polygons);vertices+=int(get_num_coordinates(g))
        holes+=sum(len(p.interiors) for p in polygons)
    return dict(features=len(features),polygon_parts=parts,vertices=vertices,holes=holes)


def annotate(data,roles,colours,mpp_xy,native_shape):
    """Add predicted names and image metadata without changing any coordinates."""
    for feature in data['features']:
        p=feature['properties'];value=int(p['value']);g=shape(feature['geometry'])
        name=roles[value-1].replace('_',' ')
        p.update(predicted_tissue_id=value,predicted_tissue=name,name=name,objectType='annotation',
            classification=dict(name=name,color='#%02x%02x%02x'%tuple(colours[value])),
            area_pixels=float(g.area),area_um2=float(g.area*np.prod(mpp_xy)),
            tissue_identity_source='Supervised prediction after Bora')
    data.update(native_shape_yx=list(native_shape),native_mpp_xy=list(mpp_xy),
        coordinate_space='native_ROI_image_pixel_edges',origin='top_left',coordinates_are_roi_local=True,
        converter='bora.multicpu_geojson.convert',profile='accurate',
        minimum_component_area_pixels=500.,simplification_tolerance_pixels=2.,pixel_exact=False,
        note='Bora coverage-aware simplification and per-label dissolution. Image coordinates, not geographic coordinates.')
    return data


def render(data,native_shape):
    pairs=[(f['geometry'],f['properties']['value']) for f in data['features']]
    return rasterize(pairs,out_shape=native_shape,fill=0,dtype=np.uint8) if pairs else np.zeros(native_shape,np.uint8)


def run(registry_root,geometry,cycle,output,cases=None,fill_below_pixels=0):
    registry_root,geometry,cycle,output=map(Path,(registry_root,geometry,cycle,output))
    registry=load(registry_root/'REGISTRY.json');frozen=load(cycle/'ALL_PREDICTIONS_FROZEN.json')
    roles=registry['ontology']['roles'];colours=palette(len(roles));output.mkdir(parents=True,exist_ok=True)
    import bora.multicpu_geojson as implementation
    converter_sha=sha(Path(implementation.__file__))
    protocol=dict(converter='bora.multicpu_geojson.convert',converter_sha256=converter_sha,
        upstream_commit='08b63a6595df687bd20a516f43f65dc531e664dd',profile='accurate',settings=PROFILE,
        native_pixel_parameters_match_bora_cli=True,threads=2,
        physical_parameter_note='Published accurate-profile parameters are in native pixels. Physical equivalents are reported for each image.',
        source_prediction_manifest_sha256=sha(cycle/'ALL_PREDICTIONS_FROZEN.json'),
        implementation_sha256=sha(Path(__file__)),classifier_and_bora_rasters_unchanged=True,
        requested_cases=sorted(cases) if cases else None,fill_below_pixels=fill_below_pixels)
    if (output/'PROTOCOL.json').exists() and load(output/'PROTOCOL.json')!=protocol:
        raise ValueError('Use a new immutable vector version')
    save(output/'PROTOCOL.json',protocol)
    models={n:load(cycle/n/'PREDICTIONS.json') for n in frozen['models']}
    for name in models:
        if sha(cycle/name/'PREDICTIONS.json')!=frozen['models'][name]:raise ValueError('Frozen prediction manifest changed')
    rows=[r for r in registry['cases'] if r['split']==frozen['split'] and (cases is None or r['case'] in cases)];records=[]
    # All geometry and predictions have frozen; conversion never uses references.
    for row in rows:
        case=row['case'];folder=output/case;folder.mkdir(exist_ok=True)
        regions=tifffile.imread(geometry/'cases'/case/'regions.native.tif');items=[]
        manifest=load(geometry/'cases'/case/'GEOMETRY_FROZEN.json')
        if sha(geometry/'cases'/case/'regions.native.tif')!=manifest['files_sha256']['regions.native.tif']:
            raise ValueError('Frozen Bora geometry changed')
        for name,model in models.items():
            path=cycle/name/(case+'.npz');expected=next(p['sha256'] for p in model['predictions'] if p['case']==case)
            if sha(path)!=expected:raise ValueError('Frozen prediction changed')
            with np.load(path) as p:mask=semantic_mask(regions,p['top5'])
            source=cycle/name/(case+'.tif')
            if not np.array_equal(tifffile.imread(source),mask):raise ValueError('Raster differs from frozen group ranks')
            target=folder/(name+'.geojson');record_file=folder/(name+'.conversion.json')
            if record_file.exists():
                item=load(record_file)
                if sha(target)!=item['geojson_sha256'] or sha(source)!=item['source_raster_sha256']:
                    raise ValueError('Completed vector output changed')
                items.append(item);continue
            cleanup=None;conversion_source=source
            if fill_below_pixels:
                from bora.mask_cleanup import fill_small_regions
                cleaned,cleanup=fill_small_regions(mask,fill_below_pixels)
                conversion_source=folder/(name+'.cleaned-input.tif')
                tifffile.imwrite(conversion_source,cleaned,compression='deflate')
            report=convert(conversion_source,target,**PROFILE)
            raw_archive=geometry/'geojson-test'/case/(name+'.geojson')
            before=(load(raw_archive)['features'] if raw_archive.exists() else
                [dict(geometry=g) for g,v in shapes(mask,mask=mask>0,connectivity=8)])
            data=annotate(load(target),roles,colours,row['sample']['native_mpp_xy'],mask.shape)
            if cleanup:
                from bora.mask_cleanup import finalize_vector_regions
                data['small_region_cleanup']=cleanup;data=finalize_vector_regions(data,mask.shape,fill_below_pixels)
            after=geometry_stats(data['features']);save(target,data)
            restored=render(data,mask.shape);tifffile.imwrite(folder/(name+'.native.tif'),restored,compression='deflate')
            gs=[shape(f['geometry']) for f in data['features']]
            overlap=sum(gs[i].intersection(gs[j]).area for i in range(len(gs)) for j in range(i))
            if overlap>1e-7:raise ValueError('Vector conversion introduced class overlap')
            item=dict(model=name,case=case,geojson_sha256=sha(target),source_raster_sha256=sha(source),
                source_ranks_sha256=sha(path),conversion=report,small_region_cleanup=cleanup,before=geometry_stats(before),after=after,
                differing_native_pixels=int(np.count_nonzero(mask!=restored)),native_pixels=mask.size,
                file_bytes=target.stat().st_size,raw_archive_bytes=raw_archive.stat().st_size if raw_archive.exists() else None,
                interclass_overlap_area_pixels=float(overlap),all_polygons_valid=True,
                simplify_um_xy=(2*np.asarray(row['sample']['native_mpp_xy'])).tolist(),
                min_component_area_um2=float(500*np.prod(row['sample']['native_mpp_xy'])))
            save(record_file,item);items.append(item)
        records.append(dict(case=case,models=items))
        print(__import__('json').dumps(dict(converted=case,models=len(items))),flush=True)
    save(output/'ALL_VECTORS_FROZEN.json',dict(total_cases=len(rows),cases=records,
        predictions_unchanged=True,expert_annotations_used_for_conversion=False,protocol_sha256=sha(output/'PROTOCOL.json')))
    # Score the exported polygons only after every vector is fixed.
    for row,record in zip(rows,records):
        if not eligible(row):continue
        truth=broad_truth(row)
        for item in record['models']:
            name=item['model'];restored=tifffile.imread(output/row['case']/(name+'.native.tif'))
            item['pixel_score']=pixel_scores(truth,restored,len(roles))
    summary=[]
    for name in models:
        all_items=[m for c in records for m in c['models'] if m['model']==name]
        scored=[m for m in all_items if 'pixel_score' in m]
        summary.append(dict(name=name,images=len(scored),
            mean_geojson_iou=float(np.mean([m['pixel_score']['macro_iou'] for m in scored])),
            mean_conversion_seconds=float(np.mean([m['conversion']['runtime_seconds'] for m in all_items])),
            raw_vertices=sum(m['before']['vertices'] for m in all_items),vertices=sum(m['after']['vertices'] for m in all_items),
            raw_polygon_parts=sum(m['before']['polygon_parts'] for m in all_items),polygon_parts=sum(m['after']['polygon_parts'] for m in all_items)))
    result=dict(split=frozen['split'],profile='accurate',converter='bora.multicpu_geojson.convert',
        converter_sha256=converter_sha,total_cases=len(rows),total_exports=len(rows)*len(models),cases=records,models=summary,
        native_pixel_coordinates=True,all_polygons_valid=True,all_interclass_overlaps_zero=True,
        simplification_tolerance_pixels=2.,minimum_component_area_pixels=500.,fill_below_pixels=fill_below_pixels,
        all_native_raster_roundtrips_exact=False,scored_cases=sum(eligible(r) for r in rows))
    save(output/'GEOJSON_COMPLETE.json',result)
    with (output/'scores.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['case','model','geojson_iou','vertices_before','vertices_after','parts_before','parts_after','conversion_seconds'])
        for c in records:
            for m in c['models']:
                writer.writerow([c['case'],m['model'],m.get('pixel_score',{}).get('macro_iou'),m['before']['vertices'],m['after']['vertices'],
                    m['before']['polygon_parts'],m['after']['polygon_parts'],m['conversion']['runtime_seconds']])
    print(__import__('json').dumps(dict(exports=result['total_exports'],models=summary)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('registry-root','geometry','cycle','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cases',nargs='+')
    p.add_argument('--viewer-output',type=Path)
    p.add_argument('--fill-below-pixels',type=int,default=0)
    a=p.parse_args();run(a.registry_root,a.geometry,a.cycle,a.output,a.cases,a.fill_below_pixels)
    if a.viewer_output:
        from .viewer import run as publish
        publish(a.registry_root,a.geometry,a.cycle,a.output,a.viewer_output)


if __name__=='__main__':main()
