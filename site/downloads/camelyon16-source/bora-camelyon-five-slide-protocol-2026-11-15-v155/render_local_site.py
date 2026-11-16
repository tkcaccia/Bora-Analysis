"""Build local-only matched CAM panels from the locked five-method score receipts."""
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA=Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
SITE=DATA/'worktrees/bora-analysis-four-benchmark-rights-hold-2026-11-14-v1/site/camelyon16'
HERE=Path(__file__).resolve().parent
FROZEN=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-15-v144/SCORING_FROZEN.json'
AGG=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-15-v153/AGGREGATE_VERIFIED.json'
AUDIT=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-15-v152/SCORE_AUDIT_VERIFIED.json'
CAL=DATA/'external-camelyon16-five-slide-2026-11-14-v24/calibrated'
NAMES={'he':'Original H&E','expert':'Official reference','bora':'Bora','unet':'Class-aware U-Net',
       'deeplabv3plus':'DeepLabV3+','medsam':'MedSAM with U-Net boxes',
       'vista_path':'VISTA-PATH generic tumor prompt'}
METHODS=('he','expert','bora','unet','deeplabv3plus','medsam','vista_path')
COLORS={1:np.array([82,200,112],dtype=np.float32),2:np.array([244,71,78],dtype=np.float32)}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def overlay(rgb,labels,method):
    result=rgb.astype(np.float32)
    if method in ('he',):return rgb
    if method in ('expert','bora','unet','deeplabv3plus'):
        mapping=((1,COLORS[1]),(2,COLORS[2]))
    else:
        mapping=((1,COLORS[2]),)
    for value,color in mapping:
        area=labels==value
        result[area]=.62*result[area]+.38*color
    return np.uint8(np.clip(result,0,255))


def centers(support):
    yy,xx=np.nonzero(support)
    assert len(yy)>0
    y0,y1=int(yy.min()),int(yy.max())
    x0,x1=int(xx.min()),int(xx.max())
    points=[]
    for gy in range(3):
        for gx in range(3):
            y=y0+(gy+.5)*(y1-y0)/3
            x=x0+(gx+.5)*(x1-x0)/3
            q=np.argmin((yy-y)**2+(xx-x)**2)
            cy,cx=int(yy[q]),int(xx[q])
            top=max(0,min(cy-128,support.shape[0]-256))
            left=max(0,min(cx-128,support.shape[1]-256))
            points.append((top,left))
    return points


def read_tiff(path,box,shape=None):
    with rasterio.open(path) as src:
        if box is None:
            h,w=shape
            return src.read(1,out_shape=(h,w),resampling=Resampling.nearest)
        top,left=box
        return src.read(1,window=Window(left,top,256,256))


def read_he(slide,box,shape=None):
    if box is None:return np.asarray(Image.open(CAL/slide/'he-overview.jpg').convert('RGB'))
    top,left=box
    with rasterio.open(CAL/slide/'calibrated-0p5-padded.tif') as src:
        arr=src.read((1,2,3),window=Window(left*8,top*8,2048,2048),
                     out_shape=(3,512,512),resampling=Resampling.bilinear)
    return np.moveaxis(arr,0,2)


def pct(value):return '—' if value is None else f'{100*value:.1f}%'


def render_slide(slide,freeze,aggregate):
    row=freeze['slides'][slide]
    support=np.memmap(row['support']['path'],dtype=np.uint8,mode='r',shape=tuple(row['shape_yx']))
    refinfo=json.loads(Path(row['reference_receipt_path']).read_text())
    ref=np.memmap(refinfo['aligned_path'],dtype=np.uint8,mode='r',shape=tuple(row['shape_yx']))
    score={m:json.loads((DATA/f'external-camelyon16-five-slide-2026-11-15-v144/test-scores/{m}/{slide}/SCORE_COMPLETE.json').read_text()) for m in row['methods']}
    views=[('overview',None)]+[(f'field-{i:02d}',box) for i,box in enumerate(centers(support),1)]
    out=SITE/'local-assets'/slide
    out.mkdir(parents=True,exist_ok=True)
    asset_rows=[]
    sections=[]
    for name,box in views:
        rgb=read_he(slide,box)
        rh,rw=rgb.shape[:2]
        panels=[]
        for method in METHODS:
            if method=='he':image=rgb
            else:
                if method=='expert':
                    m=ref if box is None else ref[box[0]:box[0]+256,box[1]:box[1]+256]
                    m=np.asarray(m)
                else:
                    mask=row['methods'][method]['shared_support']
                    m=read_tiff(mask['path'],box,shape=(rh,rw))
                if m.shape[:2]!=(rh,rw):m=cv2.resize(m,(rw,rh),interpolation=cv2.INTER_NEAREST)
                image=overlay(rgb,m,method)
            path=out/f'{name}-{method}.webp'
            Image.fromarray(image).save(path,'WEBP',quality=84)
            asset_rows.append({'slide':slide,'view':name,'method':method,
                               'path':str(path),'sha256':sha(path),'width':rw,'height':rh})
            label=NAMES[method]
            if method in score:
                value=score[method]['metrics']['shared_support']
                label+=(' · slide IoU '+pct(value['iou']) if value['iou'] is not None else
                        f" · slide false-positive tissue {value['false_positive_tissue_area_mm2']:.3f} mm²")
            panels.append(f'<figure class="unified-panel" data-method="{html.escape(method)}"><img loading="lazy" src="local-assets/{slide}/{name}-{method}.webp" alt="{html.escape(label)} on the matched {html.escape(name)} H&amp;E field" width="{rw}" height="{rh}"><figcaption>{html.escape(label)}</figcaption></figure>')
        heading='Full-slide overview' if box is None else f'Field {name[-2:]} · same coordinates in every panel'
        sections.append(f'<section class="unified-zoom" id="{name}"><h2>{heading}</h2><div class="unified-grid">{"".join(panels)}</div></section>')
    table=[]
    for method in row['methods']:
        m=score[method]['metrics'];v=m['shared_support'];n=m['native']
        table.append('<tr><th scope="row">'+html.escape(NAMES[method])+'</th>'+
                     '<td>'+pct(v['precision'])+'</td><td>'+pct(v['recall'])+'</td><td>'+pct(v['dice'])+'</td><td>'+pct(v['iou'])+'</td>'+
                     '<td>'+pct(n['iou'])+'</td><td>'+pct(v['boundary']['f1'])+'</td><td>'+str(v['lesions']['matched_components'])+'/'+str(v['lesions']['reference_components'])+'</td>'+
                     '<td>'+f"{v['false_positive_tissue_area_mm2']:.3f}"+'</td></tr>')
    known=next(iter(score.values()))['metrics']['scored_known_pixels']
    pos=next(iter(score.values()))['metrics']['shared_support']['reference_positive_pixels']
    outside=int(np.count_nonzero((ref>0)&(support==0)))
    subtitle=('positive lesion slide' if slide.startswith('tumor') else 'benign negative control')
    note=('The normal slide has no reference metastasis: IoU, Dice and lesion sensitivity are undefined. False-positive tissue area is the relevant endpoint.' if not pos else
          'The reference covers metastasis versus nonmetastatic tissue only; this is not an eight-class score.')
    body=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>{slide} · CAMELYON16 local Bora study</title><link rel="stylesheet" href="../viewer.css"><link rel="stylesheet" href="../benchmark-template.css"><link rel="stylesheet" href="../visium-hd/unified-panels.css"></head><body class="benchmark-page" data-collection="camelyon16"><a class="skip-link" href="#main-content">Skip to results</a><header class="site-header"><p class="eyebrow">BORA · LOCAL ANALYSIS ONLY</p><h1>CAMELYON16 · {slide}</h1><p class="site-intro">{subtitle}. This page is held locally; original slides and derived previews are not published online.</p><nav class="benchmark-sections" aria-label="Benchmark collections"><a href="../index.html">Benchmark 1 · ROIs</a><a href="../visium-hd/index.html">Benchmark 2 · WSIs</a><a href="index.html" aria-current="page">Benchmark 3 · CAMELYON16</a><a href="../visium/index.html">Benchmark 4 · Spatial transcriptomics</a></nav></header><main id="main-content"><p><a href="index.html">← Three-slide pilot</a></p><p class="note">Every method panel uses the same H&amp;E-derived display support. The reference panel shows all official code 1/2 pixels. The binary scores use those {known:,} known pixels; {outside:,} ({100*outside/known:.1f}%) lie outside the display support and remain in both score denominators. {pos:,} known pixels are metastasis. Native and shared-support results remain separate.</p><p class="note">Bora, U-Net and DeepLabV3+ were adapted using only tumor_093 and normal_093. MedSAM uses boxes from the adapted U-Net. VISTA-PATH uses a fixed broad “tumor” prompt, which is not metastasis-specific. KODAMA–Bora anonymous grouping was not fitted because 40,411 strictly admitted tiles across five slides exceed its 10,000-tile joint cap; it has no validated CAM tumor naming head.</p><section><h2>Binary agreement on official known pixels</h2><p>{note}</p><div class="table-wrap"><table><thead><tr><th>Method</th><th>Precision</th><th>Recall</th><th>Dice</th><th>IoU, shared support</th><th>IoU, native</th><th>Boundary F1 at 5 µm</th><th>Lesions matched</th><th>False-positive tissue, mm²</th></tr></thead><tbody>{''.join(table)}</tbody></table></div><p>Completion: five of five rendered methods for this slide. Boundary and lesion metrics have separate domains and are undefined where no reference lesion exists. This is a three-slide technical pilot; patient and center separation could not be verified.</p></section>{''.join(sections)}</main><footer class="site-footer">CAMELYON16 local CC0 analysis · no online publication · <a href="https://jamanetwork.com/journals/jama/article-abstract/2665774">CAMELYON16 challenge publication</a></footer></body></html>'''
    page=SITE/f'{slide}.html'
    page.write_text(body)
    return {'slide':slide,'page':str(page),'page_sha256':sha(page),'assets':asset_rows,
            'views':len(views),'method_panels_per_view':len(METHODS),'known_pixels':known,
            'known_pixels_outside_display_support':outside,'positive_pixels':pos}


def main():
    assert json.loads(AUDIT.read_text())['status']=='ALL_15_LOCAL_CAM_TEST_SCORES_INDEPENDENTLY_PIXEL_AUDITED'
    freeze=json.loads(FROZEN.read_text())
    agg=json.loads(AGG.read_text())
    rows=[render_slide(slide,freeze,agg) for slide in freeze['test_ids']]
    record={'status':'THREE_LOCAL_CAM_PAGES_RENDERED','scoring_freeze_sha256':sha(FROZEN),
            'score_audit_sha256':sha(AUDIT),'aggregate_sha256':sha(AGG),
            'rows':rows,'local_only':True,'online_publication':False}
    (HERE/'LOCAL_SITE_RENDERED.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':record['status'],'views':sum(r['views'] for r in rows),
                      'assets':sum(len(r['assets']) for r in rows)}))


if __name__=='__main__':main()
