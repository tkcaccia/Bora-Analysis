"""Score only expert-labelled pixels and render the conditional ROI experiment."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tifffile


COLORS = {1: (35, 185, 245), 2: (247, 155, 40)}


def score(reference, prediction, valid):
    values = {}
    for label in (1, 2):
        a = (reference == label) & valid
        b = (prediction == label) & valid
        tp = int(np.count_nonzero(a & b))
        fp = int(np.count_nonzero(~a & b))
        fn = int(np.count_nonzero(a & ~b))
        values[str(label)] = dict(dice=2*tp/max(1, 2*tp+fp+fn),
                                  iou=tp/max(1, tp+fp+fn),
                                  reference_pixels=int(a.sum()),predicted_pixels=int(b.sum()))
    values["macro_dice"] = float(np.mean([values[k]["dice"] for k in ("1", "2")]))
    values["macro_iou"] = float(np.mean([values[k]["iou"] for k in ("1", "2")]))
    values["pixel_accuracy"] = float(np.mean(reference[valid] == prediction[valid]))
    values["evaluated_pixels"] = int(valid.sum())
    return values


def overlay(image, labels, alpha=.35):
    result = image.astype(np.float32).copy()
    for label, color in COLORS.items():
        mask = labels == label
        result[mask] = (1-alpha)*result[mask]+alpha*np.asarray(color)
        edge = cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_GRADIENT,
                                np.ones((3,3),np.uint8)) > 0
        result[edge] = color
    return Image.fromarray(np.rint(result).clip(0,255).astype(np.uint8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--prediction", type=Path, required=True)
    p.add_argument("--reference-case", type=Path)
    a = p.parse_args()
    source = tifffile.imread(a.case/"original.tif")
    reference = tifffile.imread((a.reference_case or a.case)/"reference.native.tif")
    predicted = tifffile.imread(a.prediction/"mask.ome.tif")
    coarse = tifffile.imread(a.case/"coarse.tif")
    tissue = tifffile.imread(a.case/"native_tissue.tif") > 0
    if reference.shape != predicted.shape or source.shape[:2] != coarse.shape:
        raise ValueError("Reference, prediction or input geometry mismatch")
    semantic_prediction = np.zeros_like(predicted)
    semantic_prediction[predicted == 1] = 1  # tumor
    semantic_prediction[predicted == 3] = 2  # stroma
    coarse_native = cv2.resize(coarse,reference.shape[::-1],interpolation=cv2.INTER_NEAREST)
    coarse_native[~tissue] = 0
    valid = reference > 0
    conditional = a.reference_case is None
    result = dict(score_scope="Expert-annotated carcinoma and two stromal subclasses only",
        excluded="Unannotated tissue, other expert labels and background",
        reference_used_to_construct_coarse_input=conditional,
        independent_tissue_discovery_score=not conditional,
        native_shape_yx=list(reference.shape),
        baseline=score(reference,coarse_native,valid) if conditional else None,
        bora=score(reference,semantic_prediction,valid))
    if not conditional:
        mappings = [(1,2),(2,1)]
        candidates = []
        for mapping in mappings:
            proposed=np.zeros_like(coarse_native)
            proposed[coarse_native==1]=mapping[0]
            proposed[coarse_native==2]=mapping[1]
            candidates.append(score(reference,proposed,valid))
        result["posthoc_coarse_label_match_upper_bound"] = max(candidates,key=lambda z:z["macro_dice"])
        result["posthoc_match_not_used_by_bora"] = True
    (a.prediction/"conditional_metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    layers = [source,
        np.asarray(overlay(source,cv2.resize(reference,source.shape[1::-1],interpolation=cv2.INTER_NEAREST))),
        np.asarray(overlay(source,coarse)),
        np.asarray(overlay(source,cv2.resize(semantic_prediction,source.shape[1::-1],interpolation=cv2.INTER_NEAREST)))]
    labels = ["Original H&E", "Expert annotation", "Input tile mask", "Bora refined mask"]
    gap = 12
    canvas = Image.new("RGB",(2*source.shape[1]+3*gap,2*source.shape[0]+3*gap+60),"white")
    draw = ImageDraw.Draw(canvas)
    for i,(label,im) in enumerate(zip(labels,layers)):
        row,col = divmod(i,2)
        x=gap+col*(source.shape[1]+gap);y=gap+row*(source.shape[0]+gap+30)
        draw.text((x,y),label,fill="black")
        canvas.paste(Image.fromarray(im),(x,y+24))
    canvas.thumbnail((2200,2200),Image.Resampling.LANCZOS)
    canvas.save(a.prediction/"comparison.jpg",quality=91)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
