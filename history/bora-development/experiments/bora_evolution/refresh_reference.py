"""Refresh the inspected five-method reference viewer after a cohort revision."""
from pathlib import Path
from copy import deepcopy
import argparse,time,os
import numpy as np
import tifffile
from PIL import Image
from threadpoolctl import threadpool_limits
from shapely.geometry import shape,Polygon
from bora.mask_cleanup import finalize_vector_regions,fill_small_regions
from bora.compact_geojson import boundary_stats,geometry_agreement
from experiments.bora_supervised.registry import load,save,sha
from experiments.bora_supervised.evaluate import broad_truth,pixel_scores
from experiments.bora_supervised.train import score_units
from experiments.bora_supervised.viewer import overlay
from experiments.bora_group.targets import identity_labels
from experiments.bora_group.bora_geojson import render,geometry_stats


def run(study):
    study=Path(study);base=study.parent;group=base/'bora-anonymous-groups-2026-10-18-v1'
    rows={r['case']:r for r in load(study/'registry-v2/REGISTRY.json')['cases']}
    source=group/'viewer-bora-diagonal-v6';output=group/'viewer-no-puma-area20-v7';output.mkdir(exist_ok=True)
    vectors=group/'geojson-no-puma-area20-v7';data=load(source/'data.json');before=len(data['cases'])
    removed=[dict(case=c['case'],dataset=c['dataset'],reason='PUMA excluded by user') for c in data['cases'] if c['dataset']=='puma']
    data['cases']=[c for c in data['cases'] if c['dataset']!='puma'];data['excluded'].extend(removed)
    colors=np.asarray(data['colours'],np.uint8);cache={};checks=[];ys=[];ranks={m['name']:[] for m in data['models']}
    for c in data['cases']:
        case=c['case'];row=rows[case];truth=broad_truth(row);folder=output/case;folder.mkdir(exist_ok=True)
        for src in (source/case).iterdir():
            dst=folder/src.name
            if not dst.exists():os.link(src,dst)
        image=np.asarray(Image.open(folder/'original.jpg').convert('RGB'));y=identity_labels(row,group);keep=y>0;ys.append(y[keep])
        for model in c['models']:
            name=model['name'];old=group/'geojson-bora-diagonal-v6'/case/(name+'.geojson');fingerprint=sha(old)
            if fingerprint not in cache:
                start=time.monotonic();v=finalize_vector_regions(load(old),row['sample']['native_shape_yx'],20)
                cache[fingerprint]=(v,time.monotonic()-start)
            v,seconds=cache[fingerprint];v=deepcopy(v)
            path=vectors/case/(name+'.geojson');save(path,v);mask=render(v,tuple(row['sample']['native_shape_yx']))
            cleaned,_=fill_small_regions(mask,20)
            if not np.array_equal(mask,cleaned):raise AssertionError('Remaining small raster region/hole')
            ps=[]
            for f in v['features']:
                g=shape(f['geometry']);ps.extend([g] if g.geom_type=='Polygon' else g.geoms)
            if any(p.area<20 or any(Polygon(r).area<20 for r in p.interiors) for p in ps):raise AssertionError('Remaining small vector geometry/hole')
            resized=np.asarray(Image.fromarray(mask).resize((image.shape[1],image.shape[0]),Image.Resampling.NEAREST))
            # Do not overwrite a hard-linked old preview.
            preview=folder/(name+'.jpg');preview.unlink(missing_ok=True)
            Image.fromarray(overlay(image,resized,colors)).save(preview,quality=92)
            model['metrics']['pixel_score']=pixel_scores(truth,mask,len(data['roles']));model['geojson']='../'+vectors.name+'/'+case+'/'+name+'.geojson'
            model['previous_geojson']='../geojson-bora-diagonal-v6/'+case+'/'+name+'.geojson'
            stats=model['vector_stats'];stats['after']=geometry_stats(v['features']);stats['file_bytes']=path.stat().st_size
            stats['compact_boundary_report']['after']=boundary_stats(v);stats['continuous_area_cleanup_seconds']=seconds
            stats['final_native_pixel_cleanup']=v['final_native_pixel_region_cleanup'];stats['geojson_sha256']=sha(path)
            with np.load(group/'classifiers-test-v1'/name/(case+'.npz')) as p:ranks[name].append(p['top5'][keep])
            checks.append(dict(case=case,model=name,geojson_sha256=sha(path),parts=len(ps),native_pixels=int(mask.size),all_polygons_valid=all(p.is_valid for p in ps)))
        print('REFERENCE '+case,flush=True)
    # Recompute all displayed aggregates without the excluded PUMA images.
    y=np.concatenate(ys);classes=np.asarray(load(group/'classifiers-test-v1'/data['models'][0]['name']/'PREDICTIONS.json')['classes'])
    for m in data['models']:
        m.update(score_units(y,np.concatenate(ranks[m['name']]),classes))
        items=[next(v for v in c['models'] if v['name']==m['name']) for c in data['cases']]
        m['mean_native_pixel_iou']=float(np.mean([v['raster_pixel_score']['macro_iou'] for v in items]))
        m['mean_geojson_iou']=float(np.mean([v['metrics']['pixel_score']['macro_iou'] for v in items]))
    data['predicted_test_images']=len(data['cases']);data['cohort_revision']=dict(previous_viewer_images=before,shown=len(data['cases']),puma_images=0,new_expanded_test_images=78)
    summary=dict(exports=len(checks),positive_polygon_parts_below20=0,polygon_holes_below20=0,positive_raster_regions_below20=0,
        enclosed_raster_holes_below20=0,native_pixels=sum(c['native_pixels'] for c in checks),checks=checks)
    save(vectors/'VALIDATION.json',summary);save(output/'data.json',data);save(output/'excluded.json',data['excluded'])
    html=(source/'index.html').read_text().replace('<h1>Bora · compact diagonal tissue contours</h1>',
        '<h1>Bora · PUMA excluded · final polygon cleanup</h1><p><b>51 retained test images; PUMA removed.</b> This reference viewer keeps the previous five classifiers. All displayed scores have been recalculated without PUMA. Final polygon parts and enclosed holes smaller than 20 native pixels of area are removed/filled, with a separate native raster check. Exactly 20 is retained. The expanded evolution cohort contains 78 test and 38 validation images; its fresh best-five results will appear after the first completed generation.</p>')
    html=html.replace('Previous export</option>','Before geometric-area cleanup</option>')
    html=html.replace('${d.predicted_test_images} reserved test images predicted; ${d.cases.length} eligible multiclass images shown.', '${d.cases.length} eligible multiclass test images shown; PUMA excluded from every displayed aggregate.')
    (output/'index.html').write_text(html)
    save(group/'NO_PUMA_AREA20_REFERENCE_COMPLETE.json',dict(viewer=str(output/'index.html'),shown=len(data['cases']),puma_images=0,
        validation_sha256=sha(vectors/'VALIDATION.json'),vector_root=str(vectors),expanded_study=str(study)))
    save(study/'CONTINUOUS_AREA_CLEANUP_APPROVED.json',dict(reference_exports=len(checks),
        validation_sha256=sha(vectors/'VALIDATION.json'),positive_polygon_parts_below20=0,polygon_holes_below20=0,
        positive_raster_regions_below20=0,enclosed_raster_holes_below20=0))
    pending=study/'VECTOR_PROFILE_PENDING.json'
    if pending.exists():pending.rename(study/'VECTOR_PROFILE_PENDING.fulfilled.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True)
    with threadpool_limits(limits=2):run(p.parse_args().study)
