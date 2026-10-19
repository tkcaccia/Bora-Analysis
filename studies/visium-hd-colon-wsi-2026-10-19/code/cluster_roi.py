"""Make anonymous coarse tiles from Virchow2 features, without expert labels."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tifffile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    (a.output/"features").mkdir()
    for name in ("original.tif","tissue.tif","native_tissue.tif"):
        os.link(a.source/name,a.output/name)
    for name in ("base.npy","scale_1.npy","scale_2.npy","FEATURES_COMPLETE.json"):
        os.link(a.source/"features"/name,a.output/"features"/name)
    image = tifffile.imread(a.source/"original.tif")
    tissue = tifffile.imread(a.source/"tissue.tif") > 0
    tokens = np.load(a.source/"features/base.npy",mmap_mode="r")
    h,w=image.shape[:2]
    cy=(np.arange(tokens.shape[0])+.5)*14
    cx=(np.arange(tokens.shape[1])+.5)*14
    records=[];vectors=[]
    for y in range(0,h,90):
        for x in range(0,w,90):
            y1,x1=min(y+90,h),min(x+90,w)
            if min(y1-y,x1-x)<72 or tissue[y:y1,x:x1].mean()<.5:
                continue
            yi=np.flatnonzero((cy>=y+22)&(cy<y1-22))
            xi=np.flatnonzero((cx>=x+22)&(cx<x1-22))
            if len(yi) and len(xi):
                vectors.append(np.asarray(tokens[np.ix_(yi,xi)],np.float32).mean((0,1)))
                records.append((y,x,y1,x1))
    if len(vectors)<16:
        raise ValueError("Fewer than 16 eligible tissue tiles")
    z=StandardScaler().fit_transform(np.asarray(vectors))
    z=PCA(n_components=min(24,len(z)-1),random_state=43).fit_transform(z)
    groups=KMeans(n_clusters=2,n_init=20,random_state=43).fit_predict(z)+1
    coarse=np.zeros((h,w),np.uint8)
    for (y,x,y1,x1),group in zip(records,groups):coarse[y:y1,x:x1]=group
    tifffile.imwrite(a.output/"coarse.tif",coarse)
    spec=json.loads((a.source/"case.json").read_text())
    spec["case"]="visium_hd_colon_roi_48000_12000_unsupervised_k2"
    (a.output/"case.json").write_text(json.dumps(spec,indent=2)+"\n")
    report=dict(method="Virchow2 tile-mean StandardScaler/PCA24/KMeans(k=2)",
        tile_size_working_pixels=90,seed=43,expert_annotation_read=False,
        eligible_tiles=len(records),groups={str(i):int(np.sum(groups==i)) for i in (1,2)})
    (a.output/"clustering.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":main()
