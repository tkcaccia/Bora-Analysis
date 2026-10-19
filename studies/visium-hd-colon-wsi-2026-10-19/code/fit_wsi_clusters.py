"""Fit one global anonymous Virchow2 tile clustering without expert labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def source_identity(path):
    path=Path(path)
    h=hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(1024*1024))
        f.seek(max(0,path.stat().st_size-1024*1024))
        h.update(f.read(1024*1024))
    return dict(size_bytes=path.stat().st_size,edge_sha256=h.hexdigest())


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--thumbnail",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--bora-source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    sys.path.insert(0,str(a.bora_source))
    sys.path.insert(0,str(a.source))
    import torch
    from bora.wsi import WholeSlide,NativeWindow
    from experiments.virchow2_inverse.gpu_ideas import load_model,normalize
    slide=WholeSlide(a.image)
    thumb=np.asarray(Image.open(a.thumbnail).convert("RGB"))
    if thumb.shape[1]/thumb.shape[0] != slide.shape_xy[0]/slide.shape_xy[1]:
        if abs(thumb.shape[1]/thumb.shape[0]-slide.shape_xy[0]/slide.shape_xy[1])>.01:
            raise ValueError("Thumbnail aspect ratio differs from WSI")
    support=np.min(thumb,axis=2)<220
    coords=np.argwhere(support)
    rng=np.random.default_rng(20261019)
    order=rng.permutation(len(coords))
    chosen=[]
    occupied=np.zeros(support.shape,bool)
    for index in order:
        ty,tx=map(int,coords[index])
        if occupied[max(0,ty-16):ty+17,max(0,tx-16):tx+17].any():continue
        chosen.append((ty,tx))
        occupied[max(0,ty-16):ty+17,max(0,tx-16):tx+17]=True
        if len(chosen)==300:break
    if len(chosen)<100:raise ValueError("Insufficient spatially separated tissue samples")
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.4)
    model=load_model(a.model)
    window_native=round(224*.5/float(np.mean(slide.native_mpp_xy)))
    patches=[];centres=[]
    for ty,tx in chosen:
        x=round((tx+.5)*slide.shape_xy[0]/thumb.shape[1]-window_native/2)
        y=round((ty+.5)*slide.shape_xy[1]/thumb.shape[0]-window_native/2)
        x=int(np.clip(x,0,slide.shape_xy[0]-window_native))
        y=int(np.clip(y,0,slide.shape_xy[1]-window_native))
        window=NativeWindow(x,y,window_native,window_native,x,y,window_native,window_native)
        _,working,mpp=slide.read(window)
        if np.any(np.abs(mpp-.5)>.01):raise ValueError("Uncalibrated sampled Virchow2 window")
        patches.append(cv2.resize(working,(224,224),interpolation=cv2.INTER_AREA))
        centres.append((x+window_native//2,y+window_native//2))
    values=[]
    with torch.inference_mode():
        for start in range(0,len(patches),4):
            batch=np.stack(patches[start:start+4]).transpose(0,3,1,2).copy()
            x=torch.from_numpy(batch).cuda().float()/255
            with torch.autocast("cuda",dtype=torch.float16):
                tokens=model(normalize(x))[:,5:]
            values.extend(tokens.float().mean(1).cpu().numpy())
    values=np.asarray(values,np.float32)
    if values.shape!=(len(chosen),1280) or not np.isfinite(values).all():
        raise ValueError("Invalid full-channel Virchow2 sample embeddings")
    scaler=StandardScaler().fit(values)
    z=scaler.transform(values)
    pca=PCA(n_components=24,random_state=43).fit(z)
    projected=pca.transform(z)
    kmeans=KMeans(n_clusters=6,n_init=20,random_state=43).fit(projected)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/"global_clusters.npz",
        scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,
        pca_mean=pca.mean_,pca_components=pca.components_,centers=kmeans.cluster_centers_)
    report=dict(image=str(a.image),image_identity=source_identity(a.image),
        native_shape_xy=slide.shape_xy,native_mpp_xy=slide.native_mpp_xy.tolist(),
        target_mpp=.5,window_native_px=window_native,window_working_px=224,
        sampled_patch_centres_native_xy=centres,n_samples=len(chosen),
        sample_vectors_shape=list(values.shape),n_clusters=6,
        group_counts=np.bincount(kmeans.labels_,minlength=6).tolist(),
        expert_annotation_read=False,
        procedure="Global spatial sample; full Virchow2 tile embedding; StandardScaler; PCA24; KMeans6")
    (a.output/"CLUSTERS_COMPLETE.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(stage="complete",samples=len(chosen),groups=report["group_counts"])),flush=True)


if __name__=="__main__":main()
