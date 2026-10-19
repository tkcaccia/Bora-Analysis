"""Reference-restricted evaluation of the full WSI, never inference input."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely import make_valid
from shapely.geometry import box,shape
from shapely.strtree import STRtree


NAMES = ["tumor","stroma","normal_gland","blood_vessel","muscle"]
ANNOTATION_TO_CATEGORY = {
    "invasive carcinoma":1,
    "intramucosal carcinoma":1,
    "intratumoral stroma":2,
    "desmoplastic submucosa":2,
    "normal lamina propria":2,
    "normal gland":3,
    "blood vessel":4,
    "blood vessel tumor-related":4,
    "muscularis propria":5,
    "muscularis mucosa":5,
}
PREDICTED_TO_CATEGORY = {1:1,2:1,3:2,4:2,6:3,9:4,26:5}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--annotation",type=Path,required=True)
    p.add_argument("--result",type=Path,required=True)
    p.add_argument("--mask-stage",choices=("raw","final"),default="final")
    a=p.parse_args()
    if a.mask_stage=="final" and not (a.result/"VECTOR_COMPLETE.json").exists():
        raise FileNotFoundError("Full-slide final mask is incomplete")
    if a.mask_stage=="raw" and not (a.result/"CLASSIFIED_COMPLETE.json").exists():
        raise FileNotFoundError("Full-slide classifier is incomplete")
    report=json.loads((a.result/"CLASSIFIED_COMPLETE.json").read_text())
    h,w=report["native_shape_yx"]
    predicted=np.memmap(a.result/("classified.final.dat" if a.mask_stage=="final" else
                                  "classified.raw.dat"),mode="r",dtype=np.uint8,shape=(h,w))
    data=json.loads(a.annotation.read_text())
    labels=Counter();included=[];classes=[];excluded_polygons=[];excluded_geometry=0
    for feature in data["features"]:
        name=feature.get("properties",{}).get("classification",{}).get("name")
        labels[name]+=1
        if feature["geometry"]["type"] not in ("Polygon","MultiPolygon"):
            excluded_geometry+=1;continue
        g=shape(feature["geometry"])
        if not g.is_valid:g=make_valid(g)
        if g.is_empty:continue
        if name in ANNOTATION_TO_CATEGORY:
            included.append(g);classes.append(ANNOTATION_TO_CATEGORY[name])
        else:
            excluded_polygons.append(g)
    tree=STRtree(included)
    excluded_tree=STRtree(excluded_polygons)
    stats=np.zeros((6,3),np.int64)
    confusion=np.zeros((6,256),np.int64)
    evaluated=ambiguous=unmapped_overlap=0
    predicted_lut=np.zeros(256,np.uint8)
    for k,v in PREDICTED_TO_CATEGORY.items():predicted_lut[k]=v
    for y in range(0,h,4096):
        for x in range(0,w,4096):
            y1,x1=min(h,y+4096),min(w,x+4096)
            at=tree.query(box(x,y,x1,y1))
            if not len(at):continue
            local=[[] for _ in range(6)]
            for i in at:local[classes[int(i)]].append((included[int(i)],1))
            reference=np.zeros((y1-y,x1-x),np.uint8)
            number=np.zeros_like(reference)
            transform=Affine(1,0,x,0,1,y)
            for category in range(1,6):
                if not local[category]:continue
                layer=rasterize(local[category],out_shape=reference.shape,
                                transform=transform,fill=0,dtype=np.uint8)
                reference[layer>0]=category
                number+=layer
            excluded_indices=excluded_tree.query(box(x,y,x1,y1))
            if len(excluded_indices):
                excluded=rasterize(((excluded_polygons[int(i)],1) for i in excluded_indices),
                    out_shape=reference.shape,transform=transform,fill=0,dtype=np.uint8)>0
            else:
                excluded=np.zeros_like(reference,dtype=bool)
            valid=number==1
            valid &= reference>0
            unmapped_overlap+=int(np.count_nonzero(valid&excluded))
            valid &= ~excluded
            evaluated+=int(valid.sum())
            ambiguous+=int((number>1).sum())
            raw_pred=np.asarray(predicted[y:y1,x:x1])
            pred=predicted_lut[raw_pred]
            for category in range(1,6):
                a_mask=(reference==category)&valid
                p_mask=(pred==category)&valid
                if a_mask.any():
                    confusion[category]+=np.bincount(raw_pred[a_mask],minlength=256)
                stats[category,0]+=np.count_nonzero(a_mask&p_mask)
                stats[category,1]+=np.count_nonzero(~a_mask&p_mask)
                stats[category,2]+=np.count_nonzero(a_mask&~p_mask)
    metrics={};present=[]
    for category,name in enumerate(NAMES,1):
        tp,fp,fn=map(int,stats[category])
        if tp+fn:present.append(name)
        metrics[name]=dict(reference_pixels=tp+fn,true_positive=tp,false_positive=fp,
            false_negative=fn,dice=2*tp/max(1,2*tp+fp+fn),iou=tp/max(1,tp+fp+fn))
    result=dict(image_native_shape_yx=[h,w],annotation=str(a.annotation),
        mask_stage=a.mask_stage,
        annotation_feature_count=len(data["features"]),annotation_labels=dict(labels),
        polygon_features_mapped=len(included),non_polygon_features_ignored=excluded_geometry,
        polygon_features_unmapped=len(excluded_polygons),
        annotation_to_category=ANNOTATION_TO_CATEGORY,predicted_to_category=PREDICTED_TO_CATEGORY,
        score_restriction="Mapped native pixels with one mapped category and no overlapping unmapped polygon; all unannotated tissue excluded",
        annotation_used_by_inference=False,evaluated_pixels=evaluated,
        overlapping_ambiguous_pixels_excluded=ambiguous,
        mapped_pixels_overlapping_unmapped_polygons_excluded=unmapped_overlap,
        reference_to_predicted_class_pixels={NAMES[i-1]:{str(k):int(v) for k,v in
            enumerate(confusion[i]) if v} for i in range(1,6)},
        metrics=metrics,macro_dice=float(np.mean([metrics[n]["dice"] for n in present])),
        macro_iou=float(np.mean([metrics[n]["iou"] for n in present])),
        evaluated_categories=present)
    if a.mask_stage=="final" and (a.result/"BACKGROUND_COMPLETE.json").exists():
        background=json.loads((a.result/"BACKGROUND_COMPLETE.json").read_text())
        result["post_bora_background_rule"]=background["rule"]
        result["post_bora_background_removed_pixels"]=background["removed_pixels"]
    (a.result/("EVALUATION_COMPLETE.json" if a.mask_stage=="final" else
               "EVALUATION_RAW.json")).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(evaluated_pixels=evaluated,macro_dice=result["macro_dice"],
                          categories={n:metrics[n]["dice"] for n in present}),indent=2))


if __name__=="__main__":main()
