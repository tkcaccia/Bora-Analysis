"""Remove bright neutral scanner/glass pixels from the final WSI mask.

Virchow2, anonymous grouping and Bora-AF have already analyzed the complete
detected tissue. This CPU pass removes a stain-free scanner band after Bora,
without reading the expert annotation. It is checkpointed by native core.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np
import pyvips
from bora.wsi import low_stain_background, ome_mpp


def atomic_json(path,data):
    temporary=path.with_suffix(path.suffix+".tmp")
    with temporary.open("w") as stream:
        json.dump(data,stream,indent=2,allow_nan=False)
        stream.write("\n");stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--image",type=Path,required=True)
    parser.add_argument("--result",type=Path,required=True)
    parser.add_argument("--core",type=int,default=4096)
    args=parser.parse_args()
    if args.core<1:raise ValueError("Positive native core required")
    if (args.result/"BACKGROUND_COMPLETE.json").exists():
        raise FileExistsError("Completed background pass already exists")
    if not (args.result/"CLEANED_COMPLETE.json").exists():
        raise FileNotFoundError("Native mask cleanup is incomplete")
    report=json.loads((args.result/"CLASSIFIED_COMPLETE.json").read_text())
    h,w=report["native_shape_yx"]
    shape_xy,mpp=ome_mpp(args.image)
    if shape_xy!=(w,h) or not np.allclose(mpp,report["native_mpp_xy"],rtol=0,atol=1e-9):
        raise ValueError("Source image and final mask geometry/calibration disagree")
    image=pyvips.Image.new_from_file(str(args.image),access="random")
    if image.width!=w or image.height!=h or image.bands!=3:
        raise ValueError("libvips and mask geometry disagree")
    source=np.memmap(args.result/"classified.cleaned.dat",mode="r",dtype=np.uint8,shape=(h,w))
    target_path=args.result/"classified.background.dat"
    if target_path.exists():
        if target_path.stat().st_size!=h*w:raise ValueError("Invalid partial background raster")
    else:
        with target_path.open("wb") as stream:stream.truncate(h*w)
    target=np.memmap(target_path,mode="r+",dtype=np.uint8,shape=(h,w))
    checkpoint_dir=args.result/"background-windows"
    checkpoint_dir.mkdir(exist_ok=True)
    totals=Counter();removed_by_class=Counter();started=monotonic();index=0
    for y in range(0,h,args.core):
        for x in range(0,w,args.core):
            cw,ch=min(args.core,w-x),min(args.core,h-y)
            checkpoint=checkpoint_dir/f"{index:03d}.json"
            if checkpoint.exists():
                record=json.loads(checkpoint.read_text())
                if record["core_xywh"]!=[x,y,cw,ch]:raise ValueError("Mismatched background checkpoint")
            else:
                native=np.ndarray((ch,cw,3),dtype=np.uint8,
                    buffer=image.crop(x,y,cw,ch).write_to_memory()).copy()
                before=np.asarray(source[y:y+ch,x:x+cw])
                drop=low_stain_background(native)&(before>0)
                after=before.copy();after[drop]=0
                target[y:y+ch,x:x+cw]=after;target.flush()
                classes,amounts=np.unique(before[drop],return_counts=True)
                record={"index":index,"core_xywh":[x,y,cw,ch],
                        "positive_before":int(np.count_nonzero(before)),
                        "removed_pixels":int(np.count_nonzero(drop)),
                        "removed_by_class":{str(int(k)):int(v) for k,v in zip(classes,amounts)},
                        "expert_annotation_read":False}
                atomic_json(checkpoint,record)
                print(json.dumps({"background_window":index,"removed":record["removed_pixels"]}),flush=True)
            totals["positive_before"]+=record["positive_before"]
            totals["removed_pixels"]+=record["removed_pixels"]
            removed_by_class.update({int(k):v for k,v in record["removed_by_class"].items()})
            index+=1
    summary={"source_image":str(args.image),"source_size_bytes":args.image.stat().st_size,
             "native_shape_yx":[h,w],"native_mpp_xy":mpp.tolist(),
             "windows":index,"window_core_pixels":args.core,
             "rule":"HSV saturation <=10/255 and minimum RGB >210; applied only to predicted positive pixels after Bora",
             "positive_before":totals["positive_before"],"removed_pixels":totals["removed_pixels"],
             "removed_by_class":{str(k):v for k,v in sorted(removed_by_class.items())},
             "removed_fraction_of_positive":totals["removed_pixels"]/max(1,totals["positive_before"]),
             "expert_annotation_read":False,"seconds":monotonic()-started}
    atomic_json(args.result/"BACKGROUND_COMPLETE.json",summary)
    print(json.dumps({"stage":"background_complete",**summary}),flush=True)


if __name__=="__main__":main()
