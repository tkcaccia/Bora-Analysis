"""Upgrade existing published-converter exports using final native-pixel filling."""
from pathlib import Path
import argparse,time
import numpy as np
import tifffile
from threadpoolctl import threadpool_limits
from bora.mask_cleanup import finalize_vector_regions
from experiments.bora_group.bora_geojson import render,geometry_stats
from experiments.bora_group import viewer
from experiments.bora_supervised.registry import load,save,sha
from experiments.bora_supervised.evaluate import broad_truth,pixel_scores
from experiments.bora_supervised.eligibility import eligible

def run(study):
    study=Path(study);group=study.parent/'bora-anonymous-groups-2026-10-18-v1';registry_root=study.parent/'tissue-supervised-2026-10-18-v1/registry-v1'
    registry=load(registry_root/'REGISTRY.json');rows={r['case']:r for r in registry['cases']};source=group/'geojson-bora-fill20-v4';output=group/'geojson-bora-fill20-v5'
    report=load(source/'GEOJSON_COMPLETE.json');models=report['models'];records=[];total_changed=0
    save(output/'PROTOCOL.json',dict(previous_published_converter_exports_sha256=sha(source/'GEOJSON_COMPLETE.json'),
        threshold_native_pixels=20,code_sha256=sha(Path(__file__)),pixel_rule_applied_after_simplification=True))
    for case in report['cases']:
        row=rows[case['case']];dst=output/case['case'];dst.mkdir(parents=True,exist_ok=True);items=[]
        for old in case['models']:
            name=old['model'];target=dst/(name+'.geojson');done=dst/(name+'.conversion.json')
            if done.exists():items.append(load(done));continue
            start=time.monotonic();data=finalize_vector_regions(load(source/case['case']/(name+'.geojson')),tuple(row['sample']['native_shape_yx']),20)
            seconds=time.monotonic()-start;save(target,data);mask=render(data,tuple(row['sample']['native_shape_yx']))
            tifffile.imwrite(dst/(name+'.native.tif'),mask,compression='deflate')
            item=dict(old,geojson_sha256=sha(target),after=geometry_stats(data['features']),file_bytes=target.stat().st_size,
                final_native_pixel_cleanup=data['final_native_pixel_region_cleanup'],final_cleanup_seconds=seconds,
                previous_export_sha256=old['geojson_sha256']);item.pop('pixel_score',None)
            save(done,item);items.append(item)
        total_changed+=sum(m['final_native_pixel_cleanup']['final_vector_raster_pixels_changed'] for m in items)
        records.append(dict(case=case['case'],models=items));print(__import__('json').dumps(dict(cleaned=case['case'],models=len(items))),flush=True)
    save(output/'ALL_VECTORS_FROZEN.json',dict(cases=records,total_cases=len(records),final_rule_pixels=20))
    for case in records:
        row=rows[case['case']]
        if not eligible(row):continue
        truth=broad_truth(row)
        for item in case['models']:
            mask=tifffile.imread(output/case['case']/(item['model']+'.native.tif'));item['pixel_score']=pixel_scores(truth,mask,len(registry['ontology']['roles']))
    for model in models:
        items=[m for c in records for m in c['models'] if m['model']==model['name']]
        model.update(mean_geojson_iou=float(np.mean([m['pixel_score']['macro_iou'] for m in items if 'pixel_score' in m])),
            vertices=sum(m['after']['vertices'] for m in items),polygon_parts=sum(m['after']['polygon_parts'] for m in items))
    report.update(cases=records,models=models,final_native_pixel_rule=20,total_final_pixels_repaired=total_changed,
        source_report_sha256=sha(source/'GEOJSON_COMPLETE.json'))
    save(output/'GEOJSON_COMPLETE.json',report)
    view=group/'viewer-bora-fill20-v5';viewer.run(registry_root,group,group/'classifiers-test-v1',output,view)
    html=(view/'index.html').read_text().replace('<h1>Anonymous tile groups → Bora → tissue identity</h1>',
        '<h1>Anonymous tile groups → Bora → tissue identity</h1><p><b>20-pixel cleanup:</b> tissue fragments and enclosed holes smaller than 20 native-image pixels are filled from their surroundings. The rule is checked on the final rasterized GeoJSON. Exactly 20 pixels is retained. Exterior background is preserved.</p>')
    (view/'index.html').write_text(html);save(group/'FILL20_COMPLETE.json',dict(viewer=str(view/'index.html'),vector_report_sha256=sha(output/'GEOJSON_COMPLETE.json'),threshold_native_pixels=20,total_final_pixels_repaired=total_changed))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True)
    with threadpool_limits(limits=2):run(p.parse_args().study)
