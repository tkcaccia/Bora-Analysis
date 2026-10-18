"""Combine convergence-qualified bilateral results and publish all comparisons."""
import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image
import tifffile
from .registry import load,save,sha
from .viewer import overlay


def merge(base):
    base=Path(base);first=load(base/'controlled-comparators-v1/RESULTS.json')
    high=load(base/'controlled-comparators-fbs500-v1/RESULTS.json')
    follow=load(base/'controlled-comparators-fbs5000-followup-v1/RESULTS.json')
    replacements={r['case']:r for r in follow['cases']};higher={r['case']:r for r in high['cases']}
    rows=[]
    for original in first['cases']:
        row=dict(original);row['methods']={k:dict(v) for k,v in original['methods'].items() if k!='FBS-HE'}
        case=row['case'];source='controlled-comparators-fbs5000-followup-v1' if case in replacements else 'controlled-comparators-fbs500-v1'
        chosen=(replacements if case in replacements else higher)[case]['methods']['FBS-RGB']
        if any(c['info'] for c in chosen['details']['conjugate_gradient']):raise ValueError('Unconverged bilateral comparison remains')
        row['methods']['FBS-RGB']=dict(chosen,prediction_root=str(base/source))
        for name,entry in row['methods'].items():
            entry.setdefault('prediction_root',str(base/'controlled-comparators-v1'))
            p=Path(entry['prediction_root'])/'cases'/case/(name+'.tif')
            if sha(p)!=entry['mask_sha256']:raise ValueError('Refinement prediction changed')
        rows.append(row)
    groups=sorted({r['group'] for r in rows});rng=np.random.default_rng(314159);indices=rng.integers(0,len(groups),(2000,len(groups)))
    ranking=[]
    for name in rows[0]['methods']:
        entries=[r['methods'][name] for r in rows];times=np.array([r['median_seconds'] for r in entries]);values=np.array([r['score']['macro_iou'] for r in entries])
        ranking.append(dict(name=name,mean_iou=float(values.mean()),mean_seconds=float(times.mean()),over_one_second=int((times>1).sum()),
            runtime_adjusted_score=float(values.mean()-.05*np.maximum(times-1,0).mean()),
            convergence_qualified=name=='FBS-RGB'))
    comparisons=[]
    for a,b in [('Bora-AF','FBS-RGB'),('Bora-AF','TILES'),('Bora-VAF','Bora-AF')]:
        delta=np.array([np.mean([r['methods'][a]['score']['macro_iou']-r['methods'][b]['score']['macro_iou'] for r in rows if r['group']==g]) for g in groups])
        comparisons.append(dict(a=a,b=b,group_mean_iou_delta=float(delta.mean()),bootstrap95=np.percentile(delta[indices].mean(1),[2.5,97.5]).tolist()))
    record=dict(cases=rows,ranking=sorted(ranking,key=lambda r:-r['mean_iou']),paired_comparisons=comparisons,
        convergence='Every retained bilateral class solve has info0 at relative tolerance1e-5. Already-converged500-iteration outputs are identical under a5000-iteration ceiling; only six nonconverged cases required reruns.',
        claim_scope='39 controlled development ROIs; reference-conditioned conversion; cached CPU refinement, not full-pipeline throughput or comparison with an optimized bilateral implementation.',
        source_result_sha256={str(p.relative_to(base)):sha(p) for p in [base/'controlled-comparators-v1/RESULTS.json',base/'controlled-comparators-fbs500-v1/RESULTS.json',base/'controlled-comparators-fbs5000-followup-v1/RESULTS.json']})
    save(base/'CONTROLLED_FINAL.json',record);return record


def viewer(base,report):
    base=Path(base);output=base/'controlled-viewer';output.mkdir(exist_ok=True)
    registry=load(base/'registry-v1/REGISTRY.json');rows={r['case']:r for r in registry['cases']};cases=[]
    for record in report['cases']:
        row=rows[record['case']];case=row['case'];dst=output/case;dst.mkdir(exist_ok=True)
        image=Image.open(row['original_path']).convert('RGB');image.thumbnail((900,700));im=np.asarray(image)
        image.save(dst/'original.jpg',quality=90);colours=np.array([[0,0,0]]+row['sample']['palette'],np.uint8)
        src=Path(row['original_path']).parent
        for name,path in [('reference',row['reference_path']),('coarse',src/'coarse.tif')]+[(name,Path(v['prediction_root'])/'cases'/case/(name+'.tif')) for name,v in record['methods'].items() if name!='TILES']:
            mask=tifffile.imread(path);small=np.array(Image.fromarray(mask).resize(image.size,Image.Resampling.NEAREST))
            Image.fromarray(overlay(im,small,colours)).save(dst/(name+'.jpg'),quality=90)
        converted=Image.open(src/'ensemble/representation.png').convert('RGB');converted.thumbnail((900,700));converted.save(dst/'converted.jpg',quality=90)
        cases.append(dict(case=case,dataset=row['dataset'],classes=row['sample']['class_names'],colours=colours.tolist(),mpp_xy=row['sample']['working_mpp_xy'],
            metrics={name:dict(iou=v['score']['macro_iou'],seconds=v['median_seconds']) for name,v in record['methods'].items()}))
    save(output/'data.json',dict(cases=cases,ranking=report['ranking'],paired_comparisons=report['paired_comparisons']))
    (output/'index.html').write_text(HTML)
    (base/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Bora tissue study</title><style>body{font:20px system-ui;max-width:900px;margin:60px auto;line-height:1.6}a{color:#006080}</style><h1>Bora tissue study</h1><p><a href="viewer-test-v1/index.html">Held-out tissue predictions</a> —62 eligible test ROIs, five formulations selected on validation, expert masks and top-five tissue options.</p><p><a href="controlled-viewer/index.html">Bora-AF and Bora-VAF refinement comparison</a> —39 development ROIs, original H&E, expert annotation, tile input, unchanged converted RGB and refined masks.</p><p><a href="viewer-cycle-02/index.html">Exploratory classifier cycle viewer</a> —historical69-ROI validation pool before the stricter broad-role eligibility screen. Final comparisons use the eligible subset.</p>''')


HTML='''<!doctype html><meta charset="utf-8"><title>Bora controlled refinement comparison</title><style>body{font:16px system-ui;margin:24px;background:#f4f5f6;color:#182a36}h1{font-size:26px}table{border-collapse:collapse;background:white}td,th{border:1px solid #ccc;padding:8px;text-align:right}td:first-child,th:first-child{text-align:left}button,select{font:inherit;padding:9px}#panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}figure{margin:0;padding:10px;background:white}img{width:100%}figcaption{font-weight:600;margin-bottom:7px}small{display:block;font-weight:400}.key{display:inline-block;margin:6px 12px 6px 0}.swatch{display:inline-block;width:13px;height:13px;margin-right:5px}</style>
<h1>Bora controlled refinement comparison</h1><p>39 development ROIs. The same reference-derived tile input and unchanged #9 + #19 RGB are used throughout. This measures recovery from tile quantization. The bilateral solver uses the author’s reference notebook with every retained class solve converged to relative tolerance 1e-5. Runtime includes cached refinement and projection, with two CPU threads; it excludes encoder inference and RGB construction.</p><table id="scores"></table><p><button onclick="step(-1)">Previous</button> <select id="case"></select> <button onclick="step(1)">Next</button> <span id="info"></span></p><div id="legend"></div><div id="panels"></div>
<script>let data,idx=0;const el=x=>document.getElementById(x);function pc(v){return(v*100).toFixed(1)+'%'}function esc(s){return String(s).replaceAll('&','&amp;').replaceAll('<','&lt;')}
fetch('data.json').then(r=>r.json()).then(d=>{data=d;el('case').innerHTML=d.cases.map((r,i)=>`<option value="${i}">${r.case} · ${r.dataset}</option>`).join('');el('case').onchange=()=>{idx=+el('case').value;location.hash=d.cases[idx].case;show()};el('scores').innerHTML='<tr><th>Method</th><th>Mean IoU</th><th>Mean seconds</th><th>ROIs above 1 s</th></tr>'+d.ranking.map(m=>`<tr><td>${m.name}</td><td>${pc(m.mean_iou)}</td><td>${m.mean_seconds.toFixed(3)}</td><td>${m.over_one_second}</td></tr>`).join('');idx=Math.max(0,d.cases.findIndex(c=>c.case===location.hash.slice(1)));show()});function step(n){idx=(idx+n+data.cases.length)%data.cases.length;location.hash=data.cases[idx].case;show()}
function show(){let c=data.cases[idx];el('case').value=idx;el('info').textContent=`${idx+1}/${data.cases.length} · ${c.mpp_xy.join(' × ')} µm per working pixel`;el('legend').innerHTML=Object.entries(c.classes).map(([id,name])=>`<span class="key"><i class="swatch" style="background:rgb(${c.colours[id]})"></i>${id} ${esc(name)}</span>`).join('');let p=[['Original H&E','original'],['Expert annotation','reference'],['Input tile mask','coarse'],['Unchanged converted RGB','converted'],['Bora-AF','Bora-AF'],['Bora-VAF','Bora-VAF'],['Bilateral solver converged','FBS-RGB'],['Graph probability diffusion','GRAPH']];el('panels').innerHTML=p.map(([label,file])=>{let s=c.metrics[file==='coarse'?'TILES':file];return`<figure><figcaption>${label}${s?`<small>IoU ${pc(s.iou)} · ${s.seconds.toFixed(3)} s</small>`:''}</figcaption><a href="${c.case}/${file}.jpg" target="_blank"><img src="${c.case}/${file}.jpg"></a></figure>`}).join('')}
</script>'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);a=p.parse_args()
    report=merge(a.base);viewer(a.base,report)
    print(json.dumps(dict(ranking=report['ranking'],paired_comparisons=report['paired_comparisons'])))


if __name__=='__main__':main()
