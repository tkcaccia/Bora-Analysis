"""Preview two fixed Neighbor context readouts before any vector export."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


COLORS={1:(42,143,231),2:(42,143,231),3:(255,156,56),4:(255,156,56),
        5:(143,117,205),6:(65,181,112),9:(228,77,97),12:(194,148,83),26:(163,109,215)}


def sample(mask,shape):
    h,w=shape
    yy=np.minimum(((np.arange(h)+.5)*mask.shape[0]/h).astype(np.int64),mask.shape[0]-1)
    xx=np.minimum(((np.arange(w)+.5)*mask.shape[1]/w).astype(np.int64),mask.shape[1]-1)
    return np.asarray(mask[np.ix_(yy,xx)])


def panel(image,path):
    report=json.loads((path/"CLASSIFIED_COMPLETE.json").read_text())
    mask=np.memmap(path/"classified.raw.dat",mode="r",dtype=np.uint8,
                   shape=report["native_shape_yx"])
    labels=sample(mask,image.size[::-1]);result=np.asarray(image,np.float32).copy()
    for value in np.unique(labels):
        if value==0:continue
        color=np.asarray(COLORS.get(int(value),(130,185,185)),np.float32)
        result[labels==value]=.6*result[labels==value]+.4*color
    return Image.fromarray(np.rint(result).clip(0,255).astype(np.uint8))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--thumbnail",type=Path,required=True)
    parser.add_argument("--global-readout",type=Path,required=True)
    parser.add_argument("--window-readout",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    original=Image.open(args.thumbnail).convert("RGB")
    left=panel(original,args.global_readout)
    right=panel(original,args.window_readout)
    canvas=Image.new("RGB",(2*original.width,original.height+40),(255,255,255))
    canvas.paste(left,(0,40));canvas.paste(right,(original.width,40))
    draw=ImageDraw.Draw(canvas)
    draw.text((10,10),"One class per WSI-wide anonymous group",fill=(0,0,0))
    draw.text((original.width+10,10),"One class per anonymous group within each WSI window",fill=(0,0,0))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(args.output,quality=90)
    print(args.output)


if __name__=="__main__":main()
