"""Make a local, self-contained overview of the full-WSI experiment."""
import argparse
import colorsys
import html
import json
from pathlib import Path
import shutil
from collections import Counter

import cv2
import numpy as np
from PIL import Image,ImageDraw
from shapely.geometry import shape


PRED_COLORS={1:(42,143,231),2:(42,143,231),3:(255,156,56),4:(255,156,56),
             5:(143,117,205),6:(65,181,112),9:(228,77,97),12:(194,148,83),
             23:(110,168,220),26:(163,109,215)}
GROUP_COLORS={1:(79,185,233),2:(242,162,78),3:(186,117,209),4:(112,190,116),
              5:(232,121,125),6:(216,191,74)}
EXPERT_MAP={"invasive carcinoma":1,"intramucosal carcinoma":1,
            "intratumoral stroma":3,"desmoplastic submucosa":3,"normal lamina propria":3,
            "normal gland":6,"blood vessel":9,"blood vessel tumor-related":9,
            "muscularis propria":26,"muscularis mucosa":26}


def blend(image,labels,colors,alpha=.40):
    base=np.asarray(image.convert("RGB"),np.float32)
    for value,color in colors.items():
        where=labels==value
        if where.any():base[where]=(1-alpha)*base[where]+alpha*np.asarray(color)
    return Image.fromarray(np.rint(base).clip(0,255).astype(np.uint8))


def expert_outline(image,geojson,scale_x,scale_y,origin_x=0,origin_y=0):
    result=image.convert("RGBA")
    ink=Image.new("RGBA",result.size,(0,0,0,0));draw=ImageDraw.Draw(ink)
    for f in geojson["features"]:
        if f["geometry"]["type"] not in ("Polygon","MultiPolygon"):continue
        g=shape(f["geometry"])
        minx,miny,maxx,maxy=g.bounds
        if maxx<origin_x or maxy<origin_y or minx>origin_x+result.width/scale_x or miny>origin_y+result.height/scale_y:
            continue
        color=tuple(f.get("properties",{}).get("classification",{}).get("color") or [255,220,0])
        g=g.simplify(max(1.,1/min(scale_x,scale_y)))
        for poly in (list(g.geoms) if g.geom_type=="MultiPolygon" else [g]):
            if poly.geom_type!="Polygon":continue
            pts=[((x-origin_x)*scale_x,(y-origin_y)*scale_y) for x,y in poly.exterior.coords]
            if len(pts)>2:
                draw.polygon(pts,fill=(*color,88))
                draw.line(pts,fill=(*color,220),width=2)
                for ring in poly.interiors:
                    hole=[((x-origin_x)*scale_x,(y-origin_y)*scale_y) for x,y in ring.coords]
                    if len(hole)>2:draw.polygon(hole,fill=(0,0,0,0))
    return Image.alpha_composite(result,ink).convert("RGB")


def vector_outline(image,features,colors,scale_x,scale_y,origin_x,origin_y):
    base=image.convert("RGBA")
    ink=Image.new("RGBA",base.size,(0,0,0,0));draw=ImageDraw.Draw(ink)
    for feature in features:
        geometry=shape(feature["geometry"])
        minx,miny,maxx,maxy=geometry.bounds
        if (maxx<origin_x or maxy<origin_y or
                minx>origin_x+base.width/scale_x or miny>origin_y+base.height/scale_y):
            continue
        value=int(feature["properties"]["value"])
        color=colors.get(value,(255,220,0))
        geometry=geometry.simplify(max(1.,1/scale_x))
        for poly in (list(geometry.geoms) if geometry.geom_type=="MultiPolygon" else [geometry]):
            if poly.geom_type!="Polygon":continue
            for ring in [poly.exterior,*poly.interiors]:
                pts=[((x-origin_x)*scale_x,(y-origin_y)*scale_y) for x,y in ring.coords]
                if len(pts)>1:draw.line(pts,fill=(*color,230),width=2)
    return Image.alpha_composite(base,ink).convert("RGB")


def downsample_mask(mask,shape):
    h,w=shape
    ys=np.minimum(((np.arange(h)+.5)*mask.shape[0]/h).astype(np.int64),mask.shape[0]-1)
    xs=np.minimum(((np.arange(w)+.5)*mask.shape[1]/w).astype(np.int64),mask.shape[1]-1)
    return np.asarray(mask[np.ix_(ys,xs)])


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--thumbnail",type=Path,required=True)
    p.add_argument("--annotation",type=Path,required=True)
    p.add_argument("--geometry",type=Path,required=True)
    p.add_argument("--result",type=Path,required=True)
    p.add_argument("--global-readout",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    import pyvips
    from bora.wsi import WholeSlide,NativeWindow
    slide=WholeSlide(a.image)
    h,w=slide.shape_xy[1],slide.shape_xy[0]
    groups=np.memmap(a.geometry/"groups.native.dat",mode="r",dtype=np.uint8,shape=(h,w))
    mask=np.memmap(a.result/"classified.final.dat",mode="r",dtype=np.uint8,shape=(h,w))
    classifier=json.loads((a.result/"CLASSIFIED_COMPLETE.json").read_text())
    class_colors=dict(PRED_COLORS)
    present_values=(list(map(int,classifier["predicted_class_pixels"]))
                    if "predicted_class_pixels" in classifier else classifier["group_class_ids"])
    for value in present_values:
        if value and value not in class_colors:
            class_colors[value]=tuple(round(255*v) for v in
                colorsys.hsv_to_rgb((value*.61803398875)%1,.62,.92))
    expert=json.loads(a.annotation.read_text())
    legend_count=Counter()
    legend_color={}
    for feature in expert["features"]:
        category=feature.get("properties",{}).get("classification",{})
        name=category.get("name") or "(unlabelled)"
        legend_count[name]+=1
        legend_color[name]=category.get("color") or [255,220,0]
    thumb=Image.open(a.thumbnail).convert("RGB")
    thumb.save(a.output/"whole-original.jpg",quality=92)
    sx,sy=thumb.width/w,thumb.height/h
    expert_outline(thumb,expert,sx,sy).save(a.output/"whole-expert.jpg",quality=92)
    blend(thumb,downsample_mask(groups,thumb.size[::-1]),GROUP_COLORS).save(
        a.output/"whole-groups.jpg",quality=92)
    blend(thumb,downsample_mask(mask,thumb.size[::-1]),class_colors).save(
        a.output/"whole-bora.jpg",quality=92)
    sites=[("carcinoma-stroma",48000,12000,2048),
           ("upper-gland",56000,3500,2048),
           ("left-tumor",39000,9000,2048),
           ("lower-annotated",50500,21000,2048),
           ("scanner-strip",24000,41000,2048)]
    rows=[]
    for name,x,y,size in sites:
        window=NativeWindow(x,y,size,size,x,y,size,size)
        native,_,_=slide.read(window)
        original=Image.fromarray(native).resize((1024,1024),Image.Resampling.LANCZOS)
        pieces={"original":original,
            "expert":expert_outline(original,expert,.5,.5,x,y),
            "groups":blend(original,cv2.resize(np.asarray(groups[y:y+size,x:x+size]),
                                      (1024,1024),interpolation=cv2.INTER_NEAREST),GROUP_COLORS),
            "bora":blend(original,cv2.resize(np.asarray(mask[y:y+size,x:x+size]),
                                    (1024,1024),interpolation=cv2.INTER_NEAREST),class_colors)}
        ncols=(w+4095)//4096
        vector_features=[]
        for row in range(y//4096,(y+size-1)//4096+1):
            for col in range(x//4096,(x+size-1)//4096+1):
                vector_features.extend(json.loads((a.result/"vector-windows"/
                    f"{row*ncols+col:03d}.geojson").read_text()))
        pieces["geojson"]=vector_outline(original,vector_features,class_colors,.5,.5,x,y)
        for key,image in pieces.items():image.save(a.output/f"{name}-{key}.jpg",quality=91)
        rows.append((name,x,y))
    for label,source in (("conditional","roi_48000_12000"),
                         ("unsupervised","roi_48000_12000_unsupervised_k2")):
        p=a.geometry.parent/source/"bora/comparison.jpg"
        if p.exists():shutil.copy2(p,a.output/f"{label}-roi-comparison.jpg")
    for name in ("raw_readout_comparison.jpg","raw_seam_crop.jpg"):
        p=a.geometry.parent/name
        if p.exists():shutil.copy2(p,a.output/name)
    evaluation=json.loads((a.result/"EVALUATION_COMPLETE.json").read_text())
    vector_audit=json.loads((a.result/"VECTOR_COMPLETE.json").read_text())
    geometry_audit=json.loads((a.geometry/"GEOMETRY_COMPLETE.json").read_text())
    local_raw=json.loads((a.result/"EVALUATION_RAW.json").read_text())
    global_raw=(json.loads((a.global_readout/"EVALUATION_RAW.json").read_text())
                if a.global_readout else None)
    note=("The expert GeoJSON covers only part of this WSI. The full-slide clustering, Virchow2 "
          "encoding, Bora geometry and Neighbor context tissue readout did not read it. "
          "Bright, nearly neutral scanner/glass pixels and predictions in low-stain "
          "physical blocks were removed after Bora. "
          "Scores use only mapped, singly annotated pixels. The full-slide anonymous-group panel "
          "shows Bora-refined groups; the exact pre-refinement tile grid is shown in the regional comparison below.")
    css="""body{font:16px system-ui,sans-serif;background:#111827;color:#edf2f7;margin:0;padding:24px}
    h1,h2{margin:12px 0}p{max-width:1000px;line-height:1.5;color:#cbd5e1}
    .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;max-width:1700px}
    figure{margin:0;background:#1f2937;padding:10px;border-radius:10px}img{width:100%;height:auto;display:block}
    figcaption{font-weight:600;padding:7px 0 2px}table{border-collapse:collapse}td,th{border-bottom:1px solid #475569;padding:7px 12px;text-align:left}
    .wide{max-width:1500px;margin-top:18px}.wide img{max-width:100%}small{color:#aab6c8}
    @media(max-width:800px){.grid{grid-template-columns:1fr}}"""
    def grid(items):return '<div class="grid">'+''.join(
        f'<figure><img src="{html.escape(path)}"><figcaption>{html.escape(label)}</figcaption></figure>'
        for label,path in items)+'</div>'
    overview=grid([("Original 10x H&E","whole-original.jpg"),
                   ("Provided expert mask (partial coverage)","whole-expert.jpg"),
                   ("Anonymous groups after Bora-AF","whole-groups.jpg"),
                   ("Final Neighbor context tissue mask","whole-bora.jpg")])
    metrics=''.join(f'<tr><td>{html.escape(name)}</td><td>{m["reference_pixels"]:,}</td>'
        f'<td>{m["dice"]:.3f}</td><td>{m["iou"]:.3f}</td></tr>'
        for name,m in evaluation["metrics"].items() if m["reference_pixels"])
    expert_legend=''.join('<tr><td><span style="display:inline-block;width:1em;height:1em;'
        f'background:rgb({int(color[0])},{int(color[1])},{int(color[2])});margin-right:.5em"></span>'
        f'{html.escape(name)}</td><td>{legend_count[name]}</td><td>{"Scored" if name in EXPERT_MAP else "Excluded from score"}</td></tr>'
        for name,color in sorted(legend_color.items()))
    predicted_names=({int(value):classifier["class_names"][str(value)] for value in present_values}
                     if "class_names" in classifier else
                     {classifier["group_class_ids"][int(group)-1]:label
                      for group,label in classifier["group_class_names"].items()})
    predicted_legend=''.join('<tr><td><span style="display:inline-block;width:1em;height:1em;'
        f'background:rgb({color[0]},{color[1]},{color[2]});margin-right:.5em"></span>'
        f'{int(value)}</td><td>{html.escape(name)}</td></tr>'
        for value,name in sorted(predicted_names.items())
        for color in [class_colors[value]])
    sections=[]
    for name,x,y in rows:
        sections.append(f'<h2 id="{name}">{name.replace("-"," ").title()} · native ({x}, {y})</h2>'+grid([
            ("Original H&E",f"{name}-original.jpg"),
            ("Expert mask",f"{name}-expert.jpg"),
            ("Anonymous refined groups",f"{name}-groups.jpg"),
            ("Final classified mask",f"{name}-bora.jpg"),
            ("Compact GeoJSON contours",f"{name}-geojson.jpg")]))
    comparisons=''.join(f'<figure class="wide"><img src="{key}-roi-comparison.jpg"><figcaption>{label}</figcaption></figure>'
        for key,label in (("conditional","Expert-derived coarse tiles (conditional refinement)"),
                          ("unsupervised","Virchow2-derived coarse tiles (independent ROI)"))
        if (a.output/f"{key}-roi-comparison.jpg").exists())
    scale_check=''
    if global_raw:
        comparison_rows=''.join(f'<tr><td>{html.escape(name)}</td><td>{global_raw["metrics"][name]["dice"]:.3f}</td>'
                     f'<td>{local_raw["metrics"][name]["dice"]:.3f}</td></tr>'
                     for name in evaluation["evaluated_categories"])
        scale_check=(f'<h2>Whole-slide scale check</h2><p>The same fixed Neighbor context head '
            f'was applied to global anonymous groups and to groups within bounded native windows. '
            f'These exploratory readouts use the same {local_raw["evaluated_pixels"]:,} expert-mapped pixels; '
            f'the annotation was read only for scoring. Raw macro Dice: '
            f'{global_raw["macro_dice"]:.3f} global, {local_raw["macro_dice"]:.3f} window-local.</p>'
            f'<table><thead><tr><th>Tissue</th><th>Global Dice</th><th>Window-local Dice</th></tr></thead>'
            f'<tbody>{comparison_rows}</tbody></table>'
            f'<figure class="wide"><img src="raw_readout_comparison.jpg"><figcaption>Full-slide raw predictions</figcaption></figure>'
            f'<figure class="wide"><img src="raw_seam_crop.jpg"><figcaption>Raw predictions across a native window boundary</figcaption></figure>')
    body=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bora · Visium HD colon WSI</title><style>{css}</style></head><body>
    <h1>Bora on a full Visium HD colon WSI</h1>
    <p><a href="https://tkcaccia.github.io/Bora-Analysis/">Return to the tissue benchmark</a></p>
    <p>75,250 × 48,740 native pixels · 0.273812 × 0.273813 µm/pixel · Virchow2 at approximately 0.5 µm/pixel · RGB9+19 / Bora-AF / Neighbor context</p>
    <p>{html.escape(note)}</p>{overview}
    <p>Scale audit: {geometry_audit["processed"]} of {geometry_audit["total_windows"]} source windows completed;
    {vector_audit["windows"]} native vector windows exported. {geometry_audit["conversion_fallback_windows"]} locally homogeneous or undersized windows used a recorded coarse-palette RGB fallback. The native mask and GeoJSON were generated from the same final fitted coverage.</p>
    <details><summary>Expert annotation legend ({len(legend_color)} labels)</summary><table><thead><tr><th>Provided label</th><th>Features</th><th>Evaluation</th></tr></thead><tbody>{expert_legend}</tbody></table></details>
    <details open><summary>Predicted tissue legend</summary><table><thead><tr><th>Mask value</th><th>Predicted class</th></tr></thead><tbody>{predicted_legend}</tbody></table></details>
    {scale_check}
    <h2>Reference-restricted accuracy</h2><p>Macro Dice {evaluation["macro_dice"]:.3f}; {evaluation["evaluated_pixels"]:,} evaluated native pixels. Unannotated tissue is excluded.</p>
    <p>The fixed tissue head was trained before this slide and was not trained on its expert annotation. It predicts no normal-gland class here, so the zero gland Dice is a clear limitation of semantic transfer, despite successful whole-slide processing.</p>
    <table><thead><tr><th>Tissue</th><th>Annotated pixels</th><th>Dice</th><th>IoU</th></tr></thead><tbody>{metrics}</tbody></table>
    <p>Classification unit: {html.escape(classifier.get("classification_unit","whole-slide anonymous group"))}.</p>
    {''.join(sections)}
    <h2>Regional input-tile comparisons</h2>{comparisons}
    <p><small>Full-resolution OME mask and native GeoJSON remain in the Chiamaka result directory. Colors are overlays for review; GeoJSON stores the predicted tissue class of each polygon. Source H&amp;E: <a href="https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc-v4">10x Genomics Visium HD colorectal cancer dataset</a> (CC BY 4.0). The expert annotation was supplied for this study. <a href="https://github.com/tkcaccia/Bora-Analysis/tree/main/studies/visium-hd-colon-wsi-2026-10-19">Reproduction code and provenance</a>.</small></p>
    </body></html>'''
    (a.output/"index.html").write_text(body)
    print(json.dumps(dict(viewer=str(a.output/"index.html"),examples=rows,
                          assets=len(list(a.output.iterdir())))))


if __name__=="__main__":main()
