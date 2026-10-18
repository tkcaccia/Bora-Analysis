"""A test viewer organized around anonymous tiles, refinement and GeoJSON."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile
from experiments.bora_supervised.registry import load,save,sha
from experiments.bora_supervised.eligibility import eligible
from experiments.bora_supervised.viewer import palette,overlay
from experiments.bora_segment.labels import semantic_mask
from . import anonymous_palette


def small_mask(mask,size):
    return np.asarray(Image.fromarray(mask).resize(size,Image.Resampling.NEAREST))


def expert_legend(row,source):
    """List observed source labels, using the colors of the displayed overlay."""
    entries=[]
    for value in np.unique(source):
        value=int(value)
        if value==0:continue
        role=int(row['ontology_raw_to_role'].get(str(value),0))
        entries.append(dict(source_id=value,source_name=row['sample']['class_names'].get(str(value),str(value)),
            role=role,scored=role>0))
    return entries


def run(registry_root,geometry,cycle,geojson_root,output):
    registry_root,geometry,cycle,geojson_root,output=map(Path,(registry_root,geometry,cycle,geojson_root,output))
    registry=load(registry_root/'REGISTRY.json');result=load(cycle/'RESULTS.json');summary=result['ranking']
    vector_report=load(geojson_root/'GEOJSON_COMPLETE.json');smoothed=vector_report.get('profile')=='accurate'
    vector_cases={c['case']:{m['model']:m for m in c['models']} for c in vector_report['cases']}
    vector_models={m['name']:m for m in vector_report.get('models',[])}
    families=set();selected=[]
    for model in summary:
        if model['family'] in families:continue
        families.add(model['family']);selected.append(model)
    selected=selected[:5];colours=palette(len(registry['ontology']['roles']));cases=[];excluded=[]
    for row in registry['cases']:
        if row['split']!=result['split']:continue
        if not eligible(row):excluded.append(dict(case=row['case'],dataset=row['dataset'],reason='Fewer than two distinct mapped expert tissue roles'));continue
        case=row['case'];src=geometry/'cases'/case;folder=output/case;folder.mkdir(parents=True,exist_ok=True)
        original=Image.open(row['original_path']).convert('RGB');original.thumbnail((1000,850));im=np.asarray(original)
        original.save(folder/'original.jpg',quality=92);size=original.size
        source=tifffile.imread(row['reference_path']);truth=np.zeros(source.shape,np.uint8)
        for code,role in row['ontology_raw_to_role'].items():truth[source==int(code)]=role
        Image.fromarray(overlay(im,small_mask(truth,size),colours)).save(folder/'expert.jpg',quality=92)
        anonymous=tifffile.imread(src/'coarse.working.tif');bora=tifffile.imread(src/'groups.native.tif')
        information=load(src/'REGIONS.json')['regions'];record=load(src/'GEOMETRY_FROZEN.json')
        anonymous_colours=anonymous_palette(int(anonymous.max()))
        for filename,mask in [('tiles',anonymous),('bora',bora)]:
            Image.fromarray(overlay(im,small_mask(mask,size),anonymous_colours)).save(folder/(filename+'.jpg'),quality=92)
        converted=Image.open(src/'converted.png');converted=converted.resize(size,Image.Resampling.BILINEAR);converted.save(folder/'converted.jpg',quality=92)
        regions=tifffile.imread(src/'regions.native.tif');key=load(geometry/'evaluation_keys'/(case+'.json'))
        groups=[]
        for item,source_class in zip(information,key['source_class_by_output_group']):
            groups.append(dict(**item,anonymous_name=chr(64+item['anonymous_group']),
                reference_name=row['sample']['class_names'][str(source_class)],
                reference_role=row['ontology_raw_to_role'].get(str(source_class),0)))
        case_record=dict(case=case,dataset=row['dataset'],native_shape_yx=row['sample']['native_shape_yx'],
            native_mpp_xy=row['sample']['native_mpp_xy'],expert_legend=expert_legend(row,source),groups=groups,input_groups=record['input_groups'],
            output_groups=record['regions'],geometry_timing=record['timing'],models=[])
        for model in selected:
            name=model['name']
            with np.load(cycle/name/(case+'.npz')) as p:ranks=p['top5']
            mask=semantic_mask(regions,ranks)
            if smoothed:
                Image.fromarray(overlay(im,small_mask(mask,size),colours)).save(folder/(name+'-raster.jpg'),quality=92)
                mask=tifffile.imread(geojson_root/case/(name+'.native.tif'))
            Image.fromarray(overlay(im,small_mask(mask,size),colours)).save(folder/(name+'.jpg'),quality=92)
            vector=geojson_root/case/(name+'.geojson')
            if not vector.exists():raise ValueError('Missing GeoJSON for '+case+'/'+name)
            metric=next(c for c in model['cases'] if c['case']==case)
            case_record['models'].append(dict(name=name,family=model['family'],top1=ranks[:,0].tolist(),
                metrics=dict(metric,pixel_score=vector_cases[case][name]['pixel_score']) if smoothed else metric,
                raster_pixel_score=metric['pixel_score'],vector_stats=vector_cases[case][name] if smoothed else None,
                geojson='../'+geojson_root.name+'/'+case+'/'+name+'.geojson'))
        cases.append(case_record)
    save(output/'data.json',dict(cases=cases,excluded=excluded,split=result['split'],predicted_test_images=len(cases)+len(excluded),
        models=[dict({k:v for k,v in m.items() if k!='cases'},**{k:v for k,v in vector_models.get(m['name'],{}).items() if k!='name'}) for m in selected],roles=registry['ontology']['roles'],colours=colours.tolist(),
        primary_model=result['selected_segment_classifier'],tile_input='Exact previous expert-derived tiling, with IDs anonymized only',
        input_rgb_unchanged=True,new_clustering_used=False,geojson_profile=vector_report.get('profile'),geojson_checks=vector_report,
        region_geometry_sha256=sha(geometry/'ALL_GEOMETRY_FROZEN.json')))
    save(output/'excluded.json',excluded);(output/'index.html').write_text(HTML)
    print(__import__('json').dumps(dict(viewer=str(output/'index.html'),shown=len(cases),excluded=len(excluded),predicted=len(cases)+len(excluded))),flush=True)


HTML='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Bora anonymous groups and GeoJSON</title>
<style>body{font:16px system-ui;color:#152e3b;background:#f3f6f7;margin:22px}h1{font-size:26px;margin-bottom:8px}h2{font-size:20px;margin:24px 0 10px}p{line-height:1.45;max-width:1200px}button,select{font:inherit;padding:7px}table{border-collapse:collapse;background:white}td,th{padding:8px;border:1px solid #cbd5dc;text-align:right}td:first-child,th:first-child{text-align:left}.context{display:grid;grid-template-columns:repeat(3,minmax(280px,1fr));gap:13px}figure{padding:10px;background:white;margin:0}figcaption{font-weight:650;margin-bottom:6px}img{width:100%;cursor:zoom-in}.methods{display:grid;grid-template-columns:repeat(5,minmax(200px,1fr));gap:12px;overflow-x:auto}.methods figure.active{outline:3px solid #1e6b91;outline-offset:-3px}.note{font-size:14px;color:#47606e}.swatch{display:inline-block;width:12px;height:12px;margin:0 5px 0 12px}#geo{background:white;width:100%;max-height:820px;touch-action:none;cursor:grab;border:1px solid #cbd5dc}#nav{position:sticky;top:0;padding:10px;background:#e5edf1;z-index:2}#vector-info{padding:10px;background:#143c50;color:white;min-height:22px}a{color:#006992}.controls{padding:10px;background:white;display:flex;align-items:center;gap:12px;flex-wrap:wrap}.muted{background:#e9eef0;padding:10px}.key{display:inline-block;margin:4px 0}@media(max-width:850px){.context{grid-template-columns:1fr 1fr}}</style>
<style>@media(max-width:1000px){.methods{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:500px){.context,.methods{grid-template-columns:1fr}}</style>
<h1>Anonymous tile groups → Bora → tissue identity</h1>
<p>The input is the <b>previous expert-derived tile mask with its IDs anonymized</b>. Tile assignments and RGB conversion are preserved. After Bora, fastPLS identifies each whole group; all disconnected pieces of a group share its predicted identity.</p>
<p class="note" id="scope"></p><details><summary>Aggregate test scores for the five approaches</summary><table id="scores"></table></details>
<div id="nav"><button id="prev">Previous</button> <select id="dataset"></select> <select id="case"></select> <button id="next">Next</button> <span id="case-info"></span></div>
<h2>Input and geometric refinement</h2><div class="context" id="context"></div><p id="anonymous-legend"></p>
<h2>Final tissue masks from Bora GeoJSON</h2><p class="note">Select a method to inspect its GeoJSON below. All five classifiers start from the same Bora raster. The final vectors use the published Bora converter, including shared-border simplification and component filtering. Click an image to open its larger preview.</p><div class="methods" id="methods"></div><div id="legend"></div>
<h2>GeoJSON polygons over the original image</h2><div class="controls"><select id="method"></select><label>Opacity <input id="opacity" type="range" min="0" max="90" value="42"></label><label><input id="outlines" type="checkbox" checked>Outlines</label><button id="zoom-in">Zoom +</button><button id="zoom-out">Zoom −</button><button id="fit">Fit</button><a id="download" download>Download this GeoJSON</a></div>
<div id="vector-info">The vector view uses the exported GeoJSON polygons in native image pixels. Drag to pan after zooming.</div><svg id="geo" xmlns="http://www.w3.org/2000/svg"></svg>
<h2>Group identities</h2><p class="note">Reference tissue classes are the original expert labels, revealed here for evaluation after predictions are fixed. They are not inputs to the classifier.</p><table id="groups"></table><p id="timing" class="note"></p>
<script>
let data,shown=[],idx=0,modelName,vectorToken=0,box,drag;const el=x=>document.getElementById(x),ns='http://www.w3.org/2000/svg';
const names={R01:'Tile evidence voting',R02:'Whole-group mean',R03:'Mean and dispersion',R04:'Tissue composition',R05:'Neighbor context'};
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;'),pc=x=>x==null?'n/a':(100*x).toFixed(1)+'%';
fetch('data.json').then(r=>r.json()).then(d=>{data=d;modelName=d.primary_model;el('scope').innerHTML=`${d.predicted_test_images} reserved test images predicted; ${d.cases.length} eligible multiclass images shown. <a href="excluded.json">${d.excluded.length} exclusions</a> retain the single-tissue rule. This is an exploratory follow-up on previously inspected data. Configurations were chosen on validation. ${d.geojson_profile==='accurate'?'Final polygons use Bora’s published accurate profile: component filtering, shared-border simplification, and one feature per predicted tissue label. Final GeoJSON IoU is measured after rasterizing these simplified polygons.':'Raw pixel-edge audit; this export is superseded by the Bora converter update.'}`;el('scores').innerHTML='<tr><th>Method</th><th>Group accuracy</th><th>Balanced accuracy</th><th>Raster IoU</th><th>Final GeoJSON IoU</th><th>Warm head ms/ROI</th></tr>'+d.models.map(m=>`<tr><td>${names[m.family]} · ${m.name}</td><td>${pc(m.top1)}</td><td>${pc(m.balanced_accuracy)}</td><td>${m.mean_native_pixel_iou.toFixed(3)}</td><td>${m.mean_geojson_iou==null?'audit':m.mean_geojson_iou.toFixed(3)}</td><td>${(1000*m.mean_head_seconds).toFixed(2)}</td></tr>`).join('');el('dataset').innerHTML='<option value="">All datasets</option>'+[...new Set(d.cases.map(c=>c.dataset))].sort().map(n=>`<option>${n}</option>`).join('');el('method').innerHTML=d.models.map(m=>`<option value="${m.name}">${names[m.family]} · ${m.name}</option>`).join('');el('method').onchange=()=>{modelName=el('method').value;selectModel()};el('dataset').onchange=()=>filter();el('case').onchange=()=>select(+el('case').value);el('prev').onclick=()=>select((idx-1+shown.length)%shown.length);el('next').onclick=()=>select((idx+1)%shown.length);el('opacity').oninput=styleVectors;el('outlines').onchange=styleVectors;el('fit').onclick=fit;el('zoom-in').onclick=()=>zoom(.70);el('zoom-out').onclick=()=>zoom(1.43);filter(location.hash.slice(1))});
function filter(wanted){shown=data.cases.filter(c=>!el('dataset').value||c.dataset===el('dataset').value);el('case').innerHTML=shown.map((c,i)=>`<option value="${i}">${c.case}</option>`).join('');select(Math.max(0,shown.findIndex(c=>c.case===wanted)))}
function expertKey(c){return '<div id="expert-legend" class="note">'+(c.expert_legend||[]).map(g=>`<span class="key"><i class="swatch" style="background:${g.role?'rgb('+data.colours[g.role]+')':'transparent;border:1px dashed #777'}"></i>${esc(g.source_name)}${g.scored?'':' (unmapped; excluded)'}</span>`).join('<br>')+'<p>No overlay: unannotated or unmapped pixels, excluded from quality scores.</p></div>'}
function select(i){idx=i;const c=shown[idx];location.hash=c.case;el('case').value=idx;el('case-info').textContent=`${idx+1}/${shown.length} · ${c.input_groups} input groups → ${c.output_groups} Bora groups · native ${c.native_mpp_xy.join(' × ')} µm/px`;const panels=[['Original H&E','original'],['Expert tissue annotation','expert'],['Previous tiles with anonymous IDs','tiles'],['Unchanged RGB conversion 9 + 19','converted'],['Bora refinement with anonymous IDs','bora'],['Identified tissue from final GeoJSON',modelName]];el('context').innerHTML=panels.map(([label,file],i)=>`<figure><figcaption ${i===5?'id="selected-label"':''}>${label}</figcaption><a ${i===5?'id="selected-link"':''} target="_blank" href="${c.case}/${file}.jpg"><img ${i===5?'id="selected-raster"':''} src="${c.case}/${file}.jpg"></a>${i===1?expertKey(c):''}</figure>`).join('');let ids=[...new Set(c.groups.map(g=>g.anonymous_group))];el('anonymous-legend').textContent='Anonymous groups: '+ids.map(g=>String.fromCharCode(64+g)).join(', ')+'. Colors denote group IDs, not tissue names.';el('methods').innerHTML=c.models.map(m=>`<figure data-model="${m.name}"><figcaption>${names[m.family]}</figcaption><p class="note">Final GeoJSON IoU ${m.metrics.pixel_score.macro_iou.toFixed(3)} · raster IoU ${m.raster_pixel_score.macro_iou.toFixed(3)} · group accuracy ${pc(m.metrics.region_score.top1)}${m.vector_stats?'<br>'+m.vector_stats.after.polygon_parts+' polygon parts · '+m.vector_stats.after.vertices.toLocaleString()+' vertices · '+(m.vector_stats.file_bytes/1024).toFixed(0)+' KiB · export '+m.vector_stats.conversion.runtime_seconds.toFixed(3)+' s':''}</p><a target="_blank" href="${c.case}/${m.name}.jpg"><img src="${c.case}/${m.name}.jpg"></a><button data-select="${m.name}">Inspect polygons</button></figure>`).join('');document.querySelectorAll('[data-select]').forEach(b=>b.onclick=()=>{modelName=b.dataset.select;selectModel()});let roles=new Set(c.groups.map(g=>g.reference_role).filter(Boolean));c.models.forEach(m=>m.top1.forEach(g=>roles.add(g)));el('legend').innerHTML=[...roles].sort((a,b)=>a-b).map(g=>`<span class="key"><i class="swatch" style="background:rgb(${data.colours[g]})"></i>${esc(data.roles[g-1].replaceAll('_',' '))}</span>`).join('');let t=c.geometry_timing;el('timing').textContent=`Bora refinement ${t.bora_seconds.toFixed(3)} s; feature pooling and recorded output work ${t.component_and_feature_pooling_seconds.toFixed(3)} s. Existing RGB and encoder features are cached. Startup training/refit and the separate GeoJSON export are excluded from head latency.`;selectModel()}
function selectModel(){const c=shown[idx],m=c.models.find(m=>m.name===modelName)||c.models[0];modelName=m.name;el('method').value=modelName;el('selected-raster').src=c.case+'/'+modelName+'.jpg';el('selected-link').href=c.case+'/'+modelName+'.jpg';el('selected-label').textContent='Final GeoJSON tissue · '+names[m.family];document.querySelectorAll('[data-model]').forEach(f=>f.classList.toggle('active',f.dataset.model===modelName));el('groups').innerHTML='<tr><th>Anonymous group</th><th>Bora raster pieces before GeoJSON</th><th>Predicted tissue</th><th>Reference tissue class</th><th>Bora raster area µm²</th></tr>'+c.groups.map((g,i)=>`<tr><td>${g.anonymous_name}</td><td>${g.disconnected_pieces}</td><td>${esc(data.roles[m.top1[i]-1].replaceAll('_',' '))}</td><td>${esc(g.reference_name)}</td><td>${g.area_um2.toFixed(0)}</td></tr>`).join('');drawVector(c,m)}
async function drawVector(c,m){const token=++vectorToken;el('download').href=m.geojson;el('download').textContent='Download '+c.case+' '+m.name+' GeoJSON';el('vector-info').textContent='Loading native GeoJSON polygons…';const g=await fetch(m.geojson).then(r=>r.json());if(token!==vectorToken)return;const svg=el('geo');svg.replaceChildren();const image=document.createElementNS(ns,'image');image.setAttribute('href',c.case+'/original.jpg');image.setAttribute('preserveAspectRatio','none');image.setAttribute('width',c.native_shape_yx[1]);image.setAttribute('height',c.native_shape_yx[0]);svg.append(image);for(const f of g.features){let path=document.createElementNS(ns,'path'),parts=f.geometry.type==='Polygon'?[f.geometry.coordinates]:f.geometry.coordinates;path.setAttribute('d',parts.flatMap(poly=>poly.map(ring=>ring.map((p,i)=>(i?'L':'M')+p[0]+','+p[1]).join(' ')+'Z')).join(' '));path.setAttribute('fill-rule','evenodd');path.setAttribute('fill',f.properties.classification.color);path.setAttribute('vector-effect','non-scaling-stroke');path.dataset.vector='true';path.onpointerenter=()=>{el('vector-info').textContent=f.properties.predicted_tissue+' · '+f.properties.area_um2.toFixed(0)+' µm²'};const title=document.createElementNS(ns,'title');title.textContent=f.properties.predicted_tissue;path.append(title);svg.append(path)}fit();styleVectors();el('vector-info').textContent=`${g.features.length} tissue-label features · Bora accurate profile · simplified shared borders. Zoom with the buttons; drag to pan.`}
function styleVectors(){document.querySelectorAll('[data-vector]').forEach(p=>{p.setAttribute('fill-opacity',el('opacity').value/100);p.setAttribute('stroke',el('outlines').checked?'#12222a':'none');p.setAttribute('stroke-width','.7')})}
function fit(){if(!shown.length)return;const c=shown[idx];box=[0,0,c.native_shape_yx[1],c.native_shape_yx[0]];el('geo').style.aspectRatio=box[2]+'/'+box[3];applyBox()}
function applyBox(){el('geo').setAttribute('viewBox',box.join(' '))}
function zoom(f){let w=box[2]*f,h=box[3]*f;box=[box[0]+(box[2]-w)/2,box[1]+(box[3]-h)/2,w,h];applyBox()}
el('geo').onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,box:box.slice()};el('geo').setPointerCapture(e.pointerId)};
el('geo').onpointermove=e=>{if(!drag)return;const r=el('geo').getBoundingClientRect();box=[drag.box[0]-(e.clientX-drag.x)*drag.box[2]/r.width,drag.box[1]-(e.clientY-drag.y)*drag.box[3]/r.height,drag.box[2],drag.box[3]];applyBox()};el('geo').onpointerup=e=>{drag=null;el('geo').releasePointerCapture(e.pointerId)};
</script></html>'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('registry-root','geometry','cycle','geojson-root','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.registry_root,a.geometry,a.cycle,a.geojson_root,a.output)


if __name__=='__main__':main()
