"""Inspect whole Bora segments with one semantic ranking per fixed region."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import tifffile
from experiments.bora_supervised.registry import load,save,sha
from experiments.bora_supervised.viewer import palette,overlay


def thumbnail_labels(labels,size):
    return np.asarray(Image.fromarray(labels.astype(np.int32)).resize(size,Image.Resampling.NEAREST))


def outlined(image,labels,regions,colours):
    result=overlay(image,labels,colours)
    boundary=np.zeros(regions.shape,bool)
    boundary[:,1:]|=regions[:,1:]!=regions[:,:-1]
    boundary[1:,:]|=regions[1:,:]!=regions[:-1,:]
    result[boundary&(regions>0)]=[25,25,25]
    return result


def run(registry_root,geometry,cycle,output):
    registry_root,geometry,cycle,output=map(Path,(registry_root,geometry,cycle,output))
    registry=load(registry_root/'REGISTRY.json');results=load(cycle/'RESULTS.json')
    families=set();selected=[]
    for result in results['ranking']:
        if result['family'] in families:continue
        families.add(result['family']);selected.append(result)
    selected=selected[:5];roles=registry['ontology']['roles'];colours=palette(len(roles));cases=[]
    eligible_cases={r['case'] for r in selected[0]['cases']}
    for row in registry['cases']:
        case=row['case']
        if case not in eligible_cases:continue
        folder=output/case;folder.mkdir(parents=True,exist_ok=True);src=geometry/'cases'/case
        original=Image.open(row['original_path']).convert('RGB');original.thumbnail((1000,800));im=np.asarray(original);size=original.size
        original.save(folder/'original.jpg',quality=92)
        native=tifffile.imread(row['reference_path']);truth=np.zeros(native.shape,np.uint8)
        for raw,role in row['ontology_raw_to_role'].items():truth[native==int(raw)]=role
        Image.fromarray(overlay(im,thumbnail_labels(truth,size),colours)).save(folder/'reference.jpg',quality=92)
        regions=tifffile.imread(src/'regions.native.tif');small=thumbnail_labels(regions,size)
        encoded=np.stack([small&255,(small>>8)&255,(small>>16)&255],-1).astype(np.uint8)
        Image.fromarray(encoded).save(folder/'region_ids.png')
        geom_colours=palette(int(regions.max()))
        Image.fromarray(outlined(im,small,small,geom_colours)).save(folder/'regions.jpg',quality=92)
        coarse=tifffile.imread(src/'coarse.working.tif')
        Image.fromarray(overlay(im,thumbnail_labels(coarse,size),palette(4))).save(folder/'coarse.jpg',quality=92)
        rgb=Image.open(src/'converted19.png');rgb=rgb.resize(size,Image.Resampling.BILINEAR);rgb.save(folder/'converted19.jpg',quality=92)
        record=load(src/'GEOMETRY_FROZEN.json');entry=dict(case=case,dataset=row['dataset'],
            working_mpp_xy=row['sample']['working_mpp_xy'],native_mpp_xy=row['sample']['native_mpp_xy'],
            native_shape_yx=row['sample']['native_shape_yx'],display_size_xy=list(size),
            regions=load(src/'REGIONS.json')['regions'],geometry_timing=record['timing'],models=[])
        for model in selected:
            name=model['name']
            with np.load(cycle/name/(case+'.npz')) as p:ranks=p['top5']
            mask=np.zeros(small.shape,np.uint8);at=small>0;mask[at]=ranks[:,0][small[at]-1]
            Image.fromarray(outlined(im,mask,small,colours)).save(folder/(name+'.jpg'),quality=92)
            score=next(r for r in model['cases'] if r['case']==case)
            timing=next(r for r in load(cycle/name/'PREDICTIONS.json')['predictions'] if r['case']==case)
            entry['models'].append(dict(name=name,top5=ranks.tolist(),metrics=score,head_seconds=timing['head_seconds']))
        cases.append(entry)
    save(output/'data.json',dict(cases=cases,models=[{k:v for k,v in m.items() if k!='cases'} for m in selected],
        roles=roles,colours=colours.tolist(),split=results['split'],selected_on_validation=results['selected_segment_classifier'],
        geometry_sha256=sha(geometry/'ALL_GEOMETRY_FROZEN.json'),
        prediction_unit='Whole connected Bora segment',reference_used_to_define_geometry=False,
        reference_used_to_select_candidates='Validation cohort only; earlier test scores were already inspected'))
    (output/'index.html').write_text(HTML)
    print(__import__('json').dumps(dict(viewer=str(output/'index.html'),images=len(cases),families=len(selected))))


HTML='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Bora · tissue segment identification</title>
<style>body{font:16px system-ui;background:#f2f5f6;color:#142a35;margin:22px}h1{font-size:27px}p{max-width:1200px;line-height:1.5}button,select{font:inherit;padding:8px}table{border-collapse:collapse;background:white}td,th{padding:8px 12px;border:1px solid #ccc;text-align:right}td:first-child,th:first-child{text-align:left}#panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:12px}figure{margin:0;background:white;padding:10px}figcaption{font-weight:600;margin-bottom:5px}figure img{width:100%;cursor:crosshair}#ranks{position:sticky;top:0;background:#133548;color:white;padding:12px;z-index:5;min-height:42px;line-height:1.5}.swatch{display:inline-block;width:12px;height:12px;margin-right:4px}.key{display:inline-block;margin:6px 12px 6px 0}.detail{font-size:14px;font-weight:normal}a{color:#006693}</style>
<h1>Identify tissue segments after Bora</h1>
<p>Anonymous Virchow2 tile groups → unchanged algorithm 19 RGB conversion → Bora-AF regions → <b>one tissue identity and top-five shortlist per whole region</b>. Every classifier uses exactly the same frozen boundaries. Black outlines preserve separate region IDs even when two neighbors receive the same tissue label. Hover over a region to inspect its identity, area and all five methods’ rankings.</p>
<p id="scope">Exploratory segment identification results.</p><p>Expert masks do not determine the regions or RGB conversion. Scores ignore unannotated reference pixels; all H&E tissue still receives predictions. Segment scores use regions with ≥50% mapped annotation coverage. Pixel IoU also includes annotated pixels in mixed or sparsely annotated regions and errors from foreground removal. Top-five lists are ranks, not calibrated probabilities.</p>
<table id="scores"></table><p><button id="prev">Previous</button> <select id="case"></select> <button id="next">Next</button> <span id="info"></span></p>
<div id="ranks">Hover over any image to select a Bora segment.</div><p id="timing"></p><div id="panels"></div><div id="legend"></div>
<script>
let data,idx=0,ids,shown;const el=x=>document.getElementById(x),esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const names={R01:'Area-weighted tile vote',R02:'Region feature pooling',R03:'Region mean + dispersion',R04:'Region composition regression',R05:'Region + boundary context'};
fetch('data.json').then(r=>r.json()).then(d=>{data=d;el('scope').textContent=(d.split==='test'?'Exploratory follow-up test on previously inspected images. Configurations were selected on validation.':'Exploratory validation results used to select configurations.')+' Primary validation-selected classifier: '+d.selected_on_validation+'.';el('case').innerHTML=d.cases.map((c,i)=>`<option value="${i}">${esc(c.case)} · ${esc(c.dataset)}</option>`).join('');el('case').onchange=()=>select(+el('case').value);el('prev').onclick=()=>select((idx-1+d.cases.length)%d.cases.length);el('next').onclick=()=>select((idx+1)%d.cases.length);el('scores').innerHTML='<tr><th>Region classifier</th><th>Segment top 1</th><th>Segment top 5</th><th>Balanced accuracy</th><th>Native pixel IoU</th><th>Head ms / ROI</th></tr>'+d.models.map(m=>`<tr><td>${names[m.family]} (${esc(m.name)})</td><td>${(100*m.top1).toFixed(1)}%</td><td>${(100*m.top5).toFixed(1)}%</td><td>${(100*m.balanced_accuracy).toFixed(1)}%</td><td>${m.mean_native_pixel_iou.toFixed(3)}</td><td>${(1000*m.mean_head_seconds).toFixed(2)}</td></tr>`).join('');el('legend').innerHTML=d.roles.map((n,i)=>`<span class="key"><span class="swatch" style="background:rgb(${d.colours[i+1]})"></span>${esc(n.replaceAll('_',' '))}</span>`).join('');select(Math.max(0,d.cases.findIndex(c=>c.case===location.hash.slice(1))))});
function select(i){idx=i;location.hash=data.cases[i].case;show()}
function show(){const c=data.cases[idx];shown=c.case;ids=null;el('case').value=idx;el('info').textContent=`${idx+1}/${data.cases.length} · ${c.regions.length} whole regions · native ${c.native_shape_yx[1]} × ${c.native_shape_yx[0]} px · ${c.native_mpp_xy.join(' × ')} µm/px`;const t=c.geometry_timing;el('timing').textContent=`Cached-feature pipeline: grouping ${t.cluster_seconds.toFixed(3)} s; RGB ${t.conversion19_seconds.toFixed(3)} s; Bora ${t.bora_seconds.toFixed(3)} s; components + pooling ${t.component_and_feature_pooling_seconds.toFixed(3)} s; total ${t.total_cached_seconds.toFixed(3)} s. Virchow2 inference and head startup refit are excluded.`;const panels=[['Original H&E','original'],['Expert annotation · common tissue colours','reference'],['Input tile mask · anonymous groups','coarse'],['Unchanged RGB conversion 19','converted19'],['Bora region geometry · colours are region IDs','regions'],...c.models.map(m=>{let p=m.metrics.pixel_score.macro_iou;return [`${names[m.name.slice(0,3)]} · ${m.name}<br><span class="detail">pixel IoU ${p===null?'n/a':p.toFixed(3)} · segment top 1 ${(100*m.metrics.region_score.top1).toFixed(1)}% · head ${(1000*m.head_seconds).toFixed(2)} ms</span>`,m.name]})];el('panels').innerHTML=panels.map(([label,file])=>`<figure><figcaption>${label}</figcaption><a href="${c.case}/${file}.jpg" target="_blank"><img src="${c.case}/${file}.jpg" data-case="${c.case}"></a></figure>`).join('');document.querySelectorAll('figure img').forEach(im=>im.onmousemove=hover);const loader=new Image();loader.onload=()=>{if(shown!==c.case)return;const canvas=document.createElement('canvas');canvas.width=loader.width;canvas.height=loader.height;const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(loader,0,0);ids={pixels:ctx.getImageData(0,0,canvas.width,canvas.height).data,width:canvas.width,height:canvas.height}};loader.src=c.case+'/region_ids.png';el('ranks').textContent='Hover over any panel to inspect one frozen Bora tissue segment.'}
function hover(e){if(!ids||e.target.dataset.case!==shown)return;const c=data.cases[idx],b=e.target.getBoundingClientRect(),x=Math.min(ids.width-1,Math.max(0,Math.floor((e.clientX-b.left)/b.width*ids.width))),y=Math.min(ids.height-1,Math.max(0,Math.floor((e.clientY-b.top)/b.height*ids.height))),p=4*(y*ids.width+x),n=ids.pixels[p]+256*ids.pixels[p+1]+65536*ids.pixels[p+2];if(!n){el('ranks').textContent='Glass / excluded exterior background';return}const region=c.regions[n-1];el('ranks').innerHTML=`<b>Segment ${n} · ${region.area_pixels.toLocaleString()} native pixels · ${region.area_um2.toFixed(1)} µm² · anonymous group ${region.anonymous_group}</b><br>`+c.models.map(m=>`<b>${names[m.name.slice(0,3)]}</b>: `+m.top5[n-1].map((g,j)=>`${j+1}. ${esc(data.roles[g-1].replaceAll('_',' '))}`).join(' · ')).join('<br>')}
</script></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('registry-root','geometry','cycle','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.registry_root,args.geometry,args.cycle,args.output)


if __name__=='__main__':main()
