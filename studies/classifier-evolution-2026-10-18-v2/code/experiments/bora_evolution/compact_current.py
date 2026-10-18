"""Compact all accepted GeoJSON masks using calibrated shared straight chords."""
from pathlib import Path
import argparse,time,csv
import numpy as np
import tifffile
from threadpoolctl import threadpool_limits
from bora.compact_geojson import compact,boundary_stats
from experiments.bora_group.bora_geojson import render,geometry_stats
from experiments.bora_group import viewer
from experiments.bora_supervised.registry import load,save,sha
from experiments.bora_supervised.evaluate import broad_truth,pixel_scores
from experiments.bora_supervised.eligibility import eligible


def publish_viewer(registry,group,source,output,report):
    view=group/'viewer-bora-diagonal-v6';viewer.run(registry,group,group/'classifiers-test-v1',output,view)
    data=load(view/'data.json')
    for c in data['cases']:
        for m in c['models']:
            m['previous_geojson']='../'+source.name+'/'+c['case']+'/'+m['name']+'.geojson'
    data['compact_summary']=report['compact_summary'];save(view/'data.json',data)
    html=(view/'index.html').read_text()
    html=html.replace('<h1>Anonymous tile groups → Bora → tissue identity</h1>',
        '<h1>Bora · compact diagonal tissue contours</h1><p><b>New boundary fitting:</b> a shared border is fitted once with long straight chords, replacing staircase chains with diagonals. Tolerance is calibrated in micrometres and reduced when shape agreement would fall too far. The rule filling regions below 20 native pixels is retained.</p><p id="compact-summary"></p>')
    html=html.replace('<select id="method"></select>', '<select id="method"></select><label>Contour style <select id="contour-style"><option value="compact">Compact diagonal</option><option value="previous">Previous export</option></select></label>')
    html=html.replace("el('method').onchange=", "el('contour-style').onchange=()=>drawVector(shown[idx],shown[idx].models.find(m=>m.name===modelName));el('method').onchange=")
    html=html.replace("el('download').href=m.geojson", "const vectorUrl=el('contour-style').value==='previous'?m.previous_geojson:m.geojson;el('download').href=vectorUrl")
    html=html.replace("fetch(m.geojson)", "fetch(vectorUrl)")
    html=html.replace("Bora accurate profile · simplified shared borders", "${el('contour-style').value==='previous'?'Previous export':'Shared diagonal fitting'} · simplified shared borders")
    html=html.replace("el('download').textContent='Download '+c.case+' '+m.name+' GeoJSON'", "el('download').textContent='Download '+c.case+' '+m.name+' '+el('contour-style').value+' GeoJSON'")
    html=html.replace("el('scope').innerHTML=", "el('compact-summary').textContent=`Across all ${d.compact_summary.exports} exports: ${d.compact_summary.before_segments.toLocaleString()} → ${d.compact_summary.after_segments.toLocaleString()} boundary segments (${(100*d.compact_summary.segment_reduction).toFixed(1)}% fewer). Each shared border is represented consistently in its neighboring tissues. Use Contour style below to compare the previous and compact polygons.`;el('scope').innerHTML=")
    # Preserve native zoom when switching contour style for the same image.
    html=html.replace("fit();styleVectors();el('vector-info')", "if(!box||el('geo').dataset.case!==c.case){fit();el('geo').dataset.case=c.case}else applyBox();styleVectors();el('vector-info')")
    html=html.replace("m.vector_stats.after.polygon_parts+' polygon parts", "m.vector_stats.compact_boundary_report.before.boundary_segments.toLocaleString()+' → '+m.vector_stats.compact_boundary_report.after.boundary_segments.toLocaleString()+' boundary segments<br>'+m.vector_stats.after.polygon_parts+' polygon parts")
    html=html.replace("Final polygons use Bora’s published accurate profile: component filtering, shared-border simplification, and one feature per predicted tissue label.", "Final polygons use the published Bora converter followed by calibrated shared-border chord fitting and the 20-pixel cleanup.")
    html=html.replace('let data,shown=[],idx=0,modelName,vectorToken=0,box,drag;', 'let data,shown=[],idx=0,modelName,vectorToken=0,box,drag,lastVector;')
    html=html.replace('<button id="fit">Fit</button>', '<button id="fit">Fit</button><button id="boundary-detail">Boundary detail</button>')
    html=html.replace("el('fit').onclick=fit;", "el('fit').onclick=fit;el('boundary-detail').onclick=boundaryDetail;")
    html=html.replace("const svg=el('geo');svg.replaceChildren();", "lastVector=g;const svg=el('geo');svg.replaceChildren();")
    html=html.replace('function styleVectors()', """function boundaryDetail(){if(!lastVector)return;const c=shown[idx],cell=Math.max(40,Math.round(40/c.native_mpp_xy[0])),buckets=new Map();for(const f of lastVector.features){let parts=f.geometry.type==='Polygon'?[f.geometry.coordinates]:f.geometry.coordinates;for(const p of parts)for(const ring of p)for(const xy of ring){if(xy[0]<cell||xy[1]<cell||xy[0]>c.native_shape_yx[1]-cell||xy[1]>c.native_shape_yx[0]-cell)continue;let key=Math.floor(xy[0]/cell)+','+Math.floor(xy[1]/cell);buckets.set(key,(buckets.get(key)||0)+1)}}if(!buckets.size)return;const key=[...buckets].sort((a,b)=>b[1]-a[1])[0][0].split(',').map(Number);box=[(key[0]-.5)*cell,(key[1]-.5)*cell,2*cell,2*cell];applyBox()}
function styleVectors()""")
    (view/'index.html').write_text(html)
    save(group/'COMPACT_DIAGONAL_COMPLETE.json',dict(viewer=str(view/'index.html'),vector_report_sha256=sha(output/'GEOJSON_COMPLETE.json'),summary=report['compact_summary']))


def run(study):
    study=Path(study);group=study.parent/'bora-anonymous-groups-2026-10-18-v1';registry=study.parent/'tissue-supervised-2026-10-18-v1/registry-v1'
    reg=load(registry/'REGISTRY.json');rows={r['case']:r for r in reg['cases']};source=group/'geojson-bora-fill20-v5';output=group/'geojson-bora-diagonal-v6'
    previous=load(source/'GEOJSON_COMPLETE.json');records=[];source_cache={}
    save(output/'PROTOCOL.json',dict(previous_export_sha256=sha(source/'GEOJSON_COMPLETE.json'),
        profile='compact-diagonal',method='shared-rdp',maximum_tolerance_um=3.,minimum_class_shape_iou=.97,maximum_tissue_area_change=.015,
        small_regions_below_native_pixels=20,expert_annotations_used_for_fitting=False,code_sha256=sha(Path(__file__))))
    for case in previous['cases']:
        row=rows[case['case']];dst=output/case['case'];dst.mkdir(parents=True,exist_ok=True);items=[]
        for old in case['models']:
            name=old['model'];target=dst/(name+'.geojson');done=dst/(name+'.conversion.json')
            if done.exists():
                item=load(done)
                if sha(target)!=item['geojson_sha256']:raise ValueError('Completed compact export changed')
                source_cache[item['previous_export_sha256']]=target;items.append(item);continue
            cache_hit=old['geojson_sha256'] in source_cache
            if cache_hit:data=load(source_cache[old['geojson_sha256']])
            else:
                original=load(source/case['case']/(name+'.geojson'));data=compact(original,row['sample']['native_mpp_xy'],row['sample']['native_shape_yx'])
            save(target,data);mask=render(data,tuple(row['sample']['native_shape_yx']));tifffile.imwrite(dst/(name+'.native.tif'),mask,compression='deflate')
            item=dict(old,geojson_sha256=sha(target),after=geometry_stats(data['features']),file_bytes=target.stat().st_size,
                compact_boundary_report=data['compact_boundary_report'],previous_export_sha256=old['geojson_sha256'],
                final_native_pixel_cleanup=data.get('final_native_pixel_region_cleanup'),compact_seconds=data['compact_boundary_report']['runtime_seconds'],exact_source_content_cache_hit=cache_hit)
            item.pop('pixel_score',None);save(done,item);source_cache[old['geojson_sha256']]=target;items.append(item)
        records.append(dict(case=case['case'],models=items));print(__import__('json').dumps(dict(compacted=case['case'],models=len(items))),flush=True)
    save(output/'ALL_VECTORS_FROZEN.json',dict(cases=records,all_vectors_before_scoring=True,expert_annotations_used_for_fitting=False))
    for case in records:
        row=rows[case['case']]
        if not eligible(row):continue
        truth=broad_truth(row)
        for item in case['models']:
            item['pixel_score']=pixel_scores(truth,tifffile.imread(output/case['case']/(item['model']+'.native.tif')),len(reg['ontology']['roles']))
    models=[]
    for old in previous['models']:
        items=[m for c in records for m in c['models'] if m['model']==old['name']]
        models.append(dict(old,mean_geojson_iou=float(np.mean([m['pixel_score']['macro_iou'] for m in items if 'pixel_score' in m])),
            previous_mean_geojson_iou=old['mean_geojson_iou'],vertices=sum(m['after']['vertices'] for m in items),polygon_parts=sum(m['after']['polygon_parts'] for m in items)))
    all_items=[m for c in records for m in c['models']];before=sum(m['compact_boundary_report']['before']['boundary_segments'] for m in all_items);after=sum(m['compact_boundary_report']['after']['boundary_segments'] for m in all_items)
    summary=dict(exports=len(all_items),before_segments=before,after_segments=after,segment_reduction=1-after/before,
        before_stair_corners=sum(m['compact_boundary_report']['before']['axis_aligned_stair_corners'] for m in all_items),
        after_stair_corners=sum(m['compact_boundary_report']['after']['axis_aligned_stair_corners'] for m in all_items),
        before_polygon_parts=sum(m['compact_boundary_report']['before']['polygon_parts'] for m in all_items),
        after_polygon_parts=sum(m['compact_boundary_report']['after']['polygon_parts'] for m in all_items),
        mean_fitting_seconds=float(np.mean([m['compact_seconds'] for m in all_items])),fallback_exports=sum(m['compact_boundary_report']['original_export_fallback'] for m in all_items))
    report=dict(previous,cases=records,models=models,compact_summary=summary,source_report_sha256=sha(source/'GEOJSON_COMPLETE.json'),boundary_profile='compact-diagonal')
    save(output/'GEOJSON_COMPLETE.json',report);publish_viewer(registry,group,source,output,report);print(__import__('json').dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True)
    with threadpool_limits(limits=2):run(p.parse_args().study)
