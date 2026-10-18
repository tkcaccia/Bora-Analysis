"""Command line interface for Bora-AF and Bora-VAF on calibrated ROIs."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
import tifffile
from threadpoolctl import threadpool_limits
from .appearance import refine_appearance


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('image','evidence-rgb','coarse','tissue','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--method',choices=['af','vaf'],default='af');p.add_argument('--mpp-xy',type=float,nargs=2,required=True)
    p.add_argument('--tile-pixels',type=float,default=90);p.add_argument('--output-shape-yx',type=int,nargs=2)
    p.add_argument('--output-tissue',type=Path);a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.json').exists():raise ValueError('Choose a new output path')
    he=np.asarray(Image.open(a.image).convert('RGB'));rgb=np.asarray(Image.open(a.evidence_rgb).convert('RGB'))
    coarse=tifffile.imread(a.coarse);tissue=tifffile.imread(a.tissue)>0
    native=tifffile.imread(a.output_tissue)>0 if a.output_tissue else None
    cv2.setNumThreads(2)
    with threadpool_limits(limits=2):
        began=time.monotonic();mask,details=refine_appearance(rgb,coarse,tissue,a.mpp_xy,he=he,method=a.method,
            tile_size=a.tile_pixels,output_shape_yx=a.output_shape_yx,output_tissue=native)
        seconds=time.monotonic()-began
    a.output.parent.mkdir(parents=True,exist_ok=True);tifffile.imwrite(a.output,mask,compression='deflate')
    files={name:str(getattr(a,name)) for name in ('image','evidence_rgb','coarse','tissue')}
    if a.output_tissue:files['output_tissue']=str(a.output_tissue)
    report=dict(details=details,refinement_seconds=seconds,threads=2,input_mpp_xy=a.mpp_xy,
        input_sha256={name:digest(path) for name,path in files.items()},output_sha256=digest(a.output),
        timing_excludes='File loading, encoder inference, evidence-RGB construction and writing')
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
