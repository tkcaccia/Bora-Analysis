"""Publish every validation image and inspectable top-five tile predictions."""
import argparse
import colorsys
from pathlib import Path
import json
import numpy as np
from PIL import Image
import tifffile
from .registry import load,save,sha


def palette(n):
    return np.array([[0,0,0]]+[list(np.array(colorsys.hsv_to_rgb((i*.618034)%1,.65,.95))*255)
                              for i in range(n)],dtype=np.uint8)


def overlay(image,mask,colours):
    out=image.copy();known=mask>0
    out[known]=np.round(.58*image[known]+.42*colours[mask[known]]).astype(np.uint8)
    return out


def run(root,cycles,output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    registry=load(root/'REGISTRY.json');roles=registry['ontology']['roles'];colours=palette(len(roles))
    models=[]
    for cycle in map(Path,cycles):
        results=load(cycle/'RESULTS.json');seen=set()
        for r in results['ranking']:
            key=r.get('method',r['name'].split('-')[0])
            if key in seen:continue
            seen.add(key)
            models.append(dict(cycle=str(cycle),**{k:v for k,v in r.items() if k!='cases'}))
    models.sort(key=lambda r:(-r['balanced_accuracy'],-r['top1']))
    models=models[:5]
    items=[]
    for row in registry['cases']:
        if row['split']!='validation':continue
        case=row['case'];folder=output/case;folder.mkdir(exist_ok=True)
        image=Image.open(row['original_path']).convert('RGB');working=image.size
        image.thumbnail((900,700));im=np.asarray(image);image.save(folder/'original.jpg',quality=88)
        ref=tifffile.imread(row['reference_path']);mapped=np.zeros(ref.shape,np.uint8)
        for raw,role in row['ontology_raw_to_role'].items():mapped[ref==int(raw)]=role
        truth=np.array(Image.fromarray(mapped).resize(image.size,Image.Resampling.NEAREST))
        Image.fromarray(overlay(im,truth,colours)).save(folder/'reference.jpg',quality=88)
        with np.load(root/'cases'/case/'units.npz') as u:boxes=u['boxes'];tissue=u['tissue_fraction']
        entry=dict(case=case,dataset=row['dataset'],mpp_xy=row['sample']['working_mpp_xy'],
            shape_yx=row['sample']['working_shape_yx'],boxes=boxes.tolist(),models=[])
        for model in models:
            with np.load(Path(model['cycle'])/model['name']/(case+'.npz')) as p:rank=p['top5'].copy()
            rank[tissue<.02]=0
            tilemap=np.zeros(working[::-1],np.uint8)
            for (x,y,w,h),g in zip(boxes,rank[:,0]):tilemap[y:y+h,x:x+w]=g
            small=np.array(Image.fromarray(tilemap).resize(image.size,Image.Resampling.NEAREST))
            name=model['name'];Image.fromarray(overlay(im,small,colours)).save(folder/(name+'.jpg'),quality=88)
            entry['models'].append(dict(name=name,top5=rank.tolist()))
        items.append(entry)
    save(output/'data.json',dict(cases=items,models=models,roles=roles,colours=colours.tolist(),split='validation',
        registry_sha256=sha(root/'REGISTRY.json'),test_annotations_opened=False))
    (output/'index.html').write_text(HTML)


HTML='''<!doctype html><meta charset="utf-8"><title>Bora supervised tissue validation</title>
<style>body{font:16px system-ui;margin:24px;background:#f5f5f5;color:#18232c}h1{font-size:26px}button,select{padding:9px;font-size:16px}#panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:15px}figure{margin:0;background:white;padding:10px}img{width:100%;cursor:crosshair}figcaption{font-weight:600;margin-bottom:6px}#ranks{position:sticky;top:0;background:#102c3e;color:white;padding:14px;z-index:1}table{border-collapse:collapse;background:white}td,th{padding:8px;border:1px solid #ccc;text-align:right}td:first-child,th:first-child{text-align:left}.key{display:inline-block;margin:6px 10px 6px 0}.swatch{display:inline-block;width:13px;height:13px;margin-right:5px}a{color:#00628c}</style>
<h1>Bora supervised tissue classification</h1>
<p>Validation images only. The best five distinct formulations are shown when available. Masks are 42% opaque. Hover over a prediction to inspect that tile’s ranked tissue choices. Top-five choices are model ranks, not calibrated probabilities. No Bora refinement has been applied in these classifier panels.</p>
<table id="scores"></table><p><button onclick="step(-1)">Previous</button> <select id="case"></select> <button onclick="step(1)">Next</button> <span id="info"></span></p>
<div id="ranks">Move over a prediction to see its top-five tissue choices.</div><div id="panels"></div><div id="legend"></div>
<script>
let data,idx=0;const el=x=>document.getElementById(x);function esc(s){return String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;')}
fetch('data.json').then(r=>r.json()).then(d=>{data=d;el('case').innerHTML=d.cases.map((r,i)=>`<option value="${i}">${r.case} · ${r.dataset}</option>`).join('');el('case').onchange=()=>{idx=+el('case').value;location.hash=d.cases[idx].case;show()};el('scores').innerHTML='<tr><th>Method</th><th>Top 1</th><th>Top 5</th><th>Balanced accuracy</th><th>Head ms per ROI</th></tr>'+d.models.map(m=>`<tr><td>${esc(m.name)}</td><td>${(m.top1*100).toFixed(1)}%</td><td>${(m.top5*100).toFixed(1)}%</td><td>${(m.balanced_accuracy*100).toFixed(1)}%</td><td>${(1000*(m.mean_head_seconds??m.mean_native_prediction_seconds)).toFixed(2)}</td></tr>`).join('');el('legend').innerHTML=d.roles.map((r,i)=>`<span class="key"><i class="swatch" style="background:rgb(${d.colours[i+1]})"></i>${i+1} ${esc(r.replaceAll('_',' '))}</span>`).join('');idx=Math.max(0,d.cases.findIndex(c=>c.case===location.hash.slice(1)));show()});
function step(n){idx=(idx+n+data.cases.length)%data.cases.length;location.hash=data.cases[idx].case;show()}
function show(){let c=data.cases[idx];el('case').value=idx;el('info').textContent=`${idx+1}/${data.cases.length} · ROI ${c.shape_yx[1]} × ${c.shape_yx[0]} pixels · ${c.mpp_xy.join(' × ')} µm per pixel · 45 µm tile extent`;let panels=[['H&E original','original',null],['Expert annotation with broad role mapping','reference',null],...c.models.map(m=>[m.name,m.name,m])];el('panels').innerHTML=panels.map(([label,file],i)=>`<figure><figcaption>${esc(label)}</figcaption><a href="${c.case}/${file}.jpg" target="_blank"><img id="im${i}" src="${c.case}/${file}.jpg"></a></figure>`).join('');panels.forEach(([label,file,m],i)=>{if(!m)return;el('im'+i).onmousemove=e=>{let box=e.target.getBoundingClientRect(),x=(e.clientX-box.left)/box.width*c.shape_yx[1],y=(e.clientY-box.top)/box.height*c.shape_yx[0],n=c.boxes.findIndex(([bx,by,w,h])=>x>=bx&&x<bx+w&&y>=by&&y<by+h);if(n>=0)el('ranks').textContent=m.name+' · tile '+n+' · '+m.top5[n].map((g,j)=>(j+1)+'. '+(g?data.roles[g-1].replaceAll('_',' '):'glass / no output')).join('   |   ')}})}
</script>'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cycles',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.cycles,a.output)


if __name__=='__main__':main()
