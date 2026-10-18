"""Native-pixel test gallery with ranked options and the five validation choices."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile
from .registry import load,save,sha
from .evaluate import unit_map
from .viewer import palette,overlay


def run(base,output):
    base,output=Path(base).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    registry=load(base/'registry-v1/REGISTRY.json');test=load(base/'test-v1/RESULTS.json')
    selection=load(base/'test-v1/TEST_SELECTION.json');available={s['name'] for s in selection['models']}
    names=[r['name'] for r in load(base/'validation-native-v1/RESULTS.json')['ranking'] if r['name'] in available][:5]
    roles=registry['ontology']['roles'];colours=palette(len(roles));rows={r['case']:r for r in registry['cases']}
    models=[r for name in names for r in test['ranking'] if r['name']==name];cases=[]
    for result in test['cases']:
        row=rows[result['case']];case=row['case'];folder=output/case;folder.mkdir(exist_ok=True)
        image=Image.open(row['original_path']).convert('RGB');image.thumbnail((900,700));im=np.asarray(image)
        image.save(folder/'original.jpg',quality=90)
        reference=tifffile.imread(row['reference_path']);truth=np.zeros(reference.shape,np.uint8)
        for raw,g in row['ontology_raw_to_role'].items():truth[reference==int(raw)]=g
        small=np.array(Image.fromarray(truth).resize(image.size,Image.Resampling.NEAREST))
        Image.fromarray(overlay(im,small,colours)).save(folder/'reference.jpg',quality=90)
        entry=dict(case=case,dataset=row['dataset'],mpp_xy=row['sample']['working_mpp_xy'],native_mpp_xy=row['sample']['native_mpp_xy'],
            shape_yx=row['sample']['working_shape_yx'],models=[],metrics=result['models'])
        for name in names:
            feature_root=base/('regions-v1' if name.startswith(('S08','S09')) else 'registry-v1')
            units,tf=unit_map(feature_root,row)
            with np.load(base/'test-v1'/name/(case+'.npz')) as p:ranks=p['top5'].copy()
            ranks[tf<.02]=0
            labels=tifffile.imread(base/'test-v1'/name/(case+'.tif'))
            sm=np.array(Image.fromarray(labels).resize(image.size,Image.Resampling.NEAREST))
            Image.fromarray(overlay(im,sm,colours)).save(folder/(name+'.jpg'),quality=90)
            ids=np.array(Image.fromarray(units+1).resize(image.size,Image.Resampling.NEAREST)).astype(np.uint32)
            encoded=np.stack([ids%256,(ids//256)%256,ids//65536],-1).astype(np.uint8)
            Image.fromarray(encoded).save(folder/(name+'-lookup.png'))
            entry['models'].append(dict(name=name,top5=ranks.tolist()))
        cases.append(entry)
    save(output/'data.json',dict(cases=cases,models=models,roles=roles,colours=colours.tolist(),
        choice='Five models selected by validation native-pixel IoU before test scoring; model order is not reselected from test outcomes.',
        test_results_sha256=sha(base/'test-v1/RESULTS.json')))
    (output/'index.html').write_text(HTML)


HTML='''<!doctype html><meta charset="utf-8"><title>Bora tissue test masks</title>
<style>body{font:16px system-ui;margin:24px;background:#f3f4f5;color:#172b38}h1{font-size:27px}button,select{padding:9px;font:inherit}#panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}figure{margin:0;padding:10px;background:white}img{width:100%;cursor:crosshair}figcaption{margin-bottom:8px;font-weight:600}small{display:block;font-weight:400}#options{position:sticky;top:0;background:#123344;color:white;padding:15px;z-index:2;min-height:45px}table{border-collapse:collapse;background:white}td,th{padding:8px;border:1px solid #ccc;text-align:right}td:first-child,th:first-child{text-align:left}.key{display:inline-block;margin:5px 10px 5px 0}.swatch{display:inline-block;width:13px;height:13px;margin-right:5px}a{color:#00668c}</style>
<h1>Bora ranked tissue predictions on held-out groups</h1>
<p>62 eligible test ROIs with at least two mapped expert tissue roles. All predictions were frozen before test annotations were decoded. Five formulations were chosen using validation mask IoU. Masks are 42% opaque; unannotated tissue is predicted but excluded from scores. Exterior glass is removed. Hover over a predicted segment to see its top-five tissue options.</p>
<p>These are supervised classifier outputs using square tiles or H&E superpixels. The controlled Bora-AF and Bora-VAF refinement comparison is available in the <a href="../controlled-viewer/index.html">refinement gallery</a>. Images had previously been inspected during Bora architecture development.</p>
<table id="scores"></table><p><button onclick="step(-1)">Previous</button> <select id="case"></select> <button onclick="step(1)">Next</button> <span id="info"></span></p>
<div id="options">Hover over a prediction to inspect its ranked tissue choices. Ranks are not calibrated probabilities.</div><div id="panels"></div><div id="legend"></div>
<script>
let data,idx=0;const el=x=>document.getElementById(x);function esc(s){return String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;')}function pc(v){return(v*100).toFixed(1)+'%'}
fetch('data.json').then(r=>r.json()).then(d=>{data=d;el('case').innerHTML=d.cases.map((c,i)=>`<option value="${i}">${c.case} · ${c.dataset}</option>`).join('');el('case').onchange=()=>{idx=+el('case').value;location.hash=d.cases[idx].case;show()};el('scores').innerHTML='<tr><th>Method</th><th>Mean mask IoU</th><th>Mean pixel top 1</th><th>Mean pixel top 5</th><th>Native head ms per ROI</th></tr>'+d.models.map(m=>`<tr><td>${esc(m.name)}</td><td>${pc(m.mean_native_pixel_macro_iou)}</td><td>${pc(m.mean_pixel_top1)}</td><td>${pc(m.mean_pixel_top5)}</td><td>${(m.mean_native_head_seconds*1000).toFixed(2)}</td></tr>`).join('');el('legend').innerHTML=d.roles.map((r,i)=>`<span class="key"><i class="swatch" style="background:rgb(${d.colours[i+1]})"></i>${i+1} ${esc(r.replaceAll('_',' '))}</span>`).join('');idx=Math.max(0,d.cases.findIndex(c=>c.case===location.hash.slice(1)));show()});
function step(n){idx=(idx+n+data.cases.length)%data.cases.length;location.hash=data.cases[idx].case;show()}
function show(){let c=data.cases[idx];el('case').value=idx;el('info').textContent=`${idx+1}/${data.cases.length} · ROI ${c.shape_yx[1]} × ${c.shape_yx[0]} working pixels · working ${c.mpp_xy.join(' × ')} µm/px · native ${c.native_mpp_xy.join(' × ')} µm/px`;let panels=[['Original H&E','original',null],['Expert annotation','reference',null],...c.models.map(m=>[m.name,m.name,m])];el('panels').innerHTML=panels.map(([label,file,m],i)=>{let s=c.metrics[file];return`<figure><figcaption>${esc(label)}${s?`<small>IoU ${pc(s.macro_iou)} · pixel top1 ${pc(s.pixel_top1)} · pixel top5 ${pc(s.pixel_top5)}</small>`:''}</figcaption><a href="${c.case}/${file}.jpg" target="_blank"><img id="im${i}" src="${c.case}/${file}.jpg"></a></figure>`}).join('');panels.forEach(([label,file,m],i)=>{if(!m)return;let canvas=document.createElement('canvas'),context=canvas.getContext('2d',{willReadFrequently:true}),lookup=new Image();lookup.onload=()=>{canvas.width=lookup.width;canvas.height=lookup.height;context.drawImage(lookup,0,0)};lookup.src=`${c.case}/${file}-lookup.png`;el('im'+i).onmousemove=e=>{if(!canvas.width)return;let rect=e.target.getBoundingClientRect(),x=Math.min(canvas.width-1,Math.floor((e.clientX-rect.left)/rect.width*canvas.width)),y=Math.min(canvas.height-1,Math.floor((e.clientY-rect.top)/rect.height*canvas.height)),p=context.getImageData(x,y,1,1).data,n=p[0]+256*p[1]+65536*p[2]-1;if(n>=0&&m.top5[n])el('options').textContent=file+' · segment '+n+' · '+m.top5[n].map((g,j)=>(j+1)+'. '+(g?data.roles[g-1].replaceAll('_',' '):'glass / no output')).join('   |   ')}})}
</script>'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.base,a.output)


if __name__=='__main__':main()
