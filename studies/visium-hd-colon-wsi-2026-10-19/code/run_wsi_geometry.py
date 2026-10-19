"""Checkpointed full-slide Virchow2 -> RGB9+19 -> Bora-AF geometry pass.

This is one fail-fast Chiamaka GPU process. There are no child jobs or retries.
Each completed native core is atomically checkpointed before the next is read.
The expert GeoJSON is neither opened nor accepted as an argument here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import monotonic

import cv2
import numpy as np
from PIL import Image
from scipy import sparse


PALETTE = np.asarray([[79,185,233],[242,162,78],[186,117,209],
                      [112,190,116],[232,121,125],[216,191,74]],np.uint8)


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w") as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def clusters(values,model):
    z=(values-model["scaler_mean"])/model["scaler_scale"]
    p=(z-model["pca_mean"])@model["pca_components"].T
    diff=p[:,None,:]-model["centers"][None,:,:]
    return np.argmin(np.sum(diff*diff,axis=2),axis=1).astype(np.uint8)+1


def make_coarse(tokens,tissue,global_model):
    h,w=tissue.shape
    cy=(np.arange(tokens.shape[0])+.5)*14
    cx=(np.arange(tokens.shape[1])+.5)*14
    coarse=np.zeros((h,w),np.uint8)
    points=[];bounds=[]
    for y in range(0,h,90):
        for x in range(0,w,90):
            y1,x1=min(h,y+90),min(w,x+90)
            if min(y1-y,x1-x)<72 or tissue[y:y1,x:x1].mean()<.02:
                continue
            yi=np.flatnonzero((cy>=y+22)&(cy<y1-22))
            xi=np.flatnonzero((cx>=x+22)&(cx<x1-22))
            if len(yi) and len(xi):
                points.append(np.asarray(tokens[np.ix_(yi,xi)],np.float32).mean((0,1)))
                bounds.append((y,y1,x,x1))
    if not points:
        # A small tissue fragment may fall only in partial edge tiles. The
        # slide-wide KMeans can still identify it from the nearest token.
        ys,xs=np.nonzero(tissue)
        if not len(ys):return coarse,0
        ty,tx=int(np.median(ys)),int(np.median(xs))
        gy,gx=min(ty//14,tokens.shape[0]-1),min(tx//14,tokens.shape[1]-1)
        group=int(clusters(np.asarray(tokens[gy,gx],np.float32)[None,:],global_model)[0])
        coarse[:]=group
        return coarse,1
    groups=clusters(np.asarray(points,np.float32),global_model)
    for (y,y1,x,x1),group in zip(bounds,groups):coarse[y:y1,x:x1]=group
    return coarse,len(points)


def exact_core_pool(tokens,core_groups,window,working_shape,n_groups=6):
    """Native-pixel weighted token pooling, without retaining a WSI token grid."""
    h,w=core_groups.shape;gh,gw=tokens.shape[:2]
    y=(np.arange(h)+window.core_y-window.y+.5)*working_shape[0]/window.height/14
    x=(np.arange(w)+window.core_x-window.x+.5)*working_shape[1]/window.width/14
    ty=np.minimum(y.astype(np.int32),gh-1)
    tx=np.minimum(x.astype(np.int32),gw-1)
    token_ids=ty[:,None]*gw+tx[None,:]
    valid=core_groups>0
    weights=sparse.coo_matrix((np.ones(int(valid.sum()),np.float32),
        (core_groups[valid].astype(np.int32)-1,token_ids[valid])),
        shape=(n_groups,gh*gw)).tocsr()
    mass=np.asarray(weights.sum(1)).ravel().astype(np.float64)
    sums=np.asarray(weights@np.asarray(tokens,np.float32).reshape(-1,1280),np.float64)
    return sums,mass


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--global-clusters",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--feature-source",type=Path,required=True)
    p.add_argument("--bora-source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--indices",type=int,nargs="*")
    p.add_argument("--deadline-seconds",type=int,default=3300)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    sys.path.insert(0,str(a.bora_source));sys.path.insert(0,str(a.feature_source))
    import torch
    from bora.wsi import WholeSlide
    from bora.pipeline import Case
    from bora.conversion import convert_rgb
    from bora.appearance import refine_appearance
    from experiments.virchow2_inverse.gpu_ideas import load_model
    from experiments.virchow2_multi.run import encode
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.4)
    cv2.setNumThreads(2)
    slide=WholeSlide(a.image)
    if not np.allclose(slide.native_mpp_xy,[.27381215039690188,.2738132569747595],atol=1e-8):
        raise ValueError("Unexpected slide calibration")
    windows=list(slide.windows(4096,512))
    global_model=dict(np.load(a.global_clusters,allow_pickle=False))
    if global_model["centers"].shape[0]!=6:raise ValueError("Expected six global groups")
    manifest=dict(image=str(a.image),image_size_bytes=a.image.stat().st_size,
        native_shape_xy=slide.shape_xy,native_mpp_xy=slide.native_mpp_xy.tolist(),
        global_clusters_sha256=sha256(a.global_clusters),checkpoint_sha256=sha256(a.model/"model.safetensors"),
        script_sha256=sha256(__file__),core_pixels=4096,halo_pixels=512,
        total_windows=len(windows),limit=a.limit,indices=a.indices,
        expert_annotation_read=False,all_tissue_encoded=True,
        low_support_policy="explicit coarse-palette fallback only for <16 tiles or one group")
    atomic_json(a.output/"RUN_MANIFEST.json",manifest)
    # The sparse on-disk array is never read into process memory as a WSI.
    mask=np.memmap(a.output/"groups.native.dat",mode="w+",dtype=np.uint8,
                   shape=(slide.shape_xy[1],slide.shape_xy[0]))
    mask[:]=0;mask.flush()
    model=load_model(a.model)
    group_sums=np.zeros((6,1280),np.float64)
    group_mass=np.zeros(6,np.float64)
    started=monotonic();processed=0;skipped=0;fallback=0
    with threadpool_limits(limits=2):
        for index,window in enumerate(windows):
            if a.limit and index>=a.limit:break
            if a.indices is not None and index not in a.indices:continue
            if monotonic()-started>a.deadline_seconds-120:
                atomic_json(a.output/"STOPPED_AT_CHECKPOINT.json",dict(next_index=index,
                    processed=processed,skipped=skipped,reason="self deadline with 120-second margin"))
                break
            t=monotonic()
            native,working,mpp=slide.read(window)
            support_native=np.min(native,axis=2)<235
            core=window.core_slices
            if not np.any(support_native[core]):
                skipped+=1
                atomic_json(a.output/f"window-{index:03d}.json",dict(index=index,status="background",
                    native_core_xywh=[window.core_x,window.core_y,window.core_width,window.core_height],
                    seconds=monotonic()-t))
                print(json.dumps(dict(index=index,status="background",seconds=round(monotonic()-t,2))),flush=True)
                continue
            support=cv2.resize(support_native.astype(np.uint8),working.shape[1::-1],
                               interpolation=cv2.INTER_AREA)>0
            tokens,base_meta=encode(model,working)
            expected=tuple((np.asarray(working.shape[:2])+13)//14)
            if tokens.shape!=(*expected,1280) or not np.isfinite(tokens).all():
                raise ValueError(f"Window {index} base token grid invalid")
            coarse,eligible=make_coarse(tokens,support,global_model)
            if not np.any(coarse):raise ValueError(f"Window {index} lacks a coarse group")
            variants={}
            for name,factor in (("scale_1",2),("scale_2",4)):
                scaled=np.asarray(Image.fromarray(working).resize(
                    (round(working.shape[1]/factor),round(working.shape[0]/factor)),
                    Image.Resampling.LANCZOS))
                if min(scaled.shape[:2])<224:
                    scaled=np.pad(scaled,((0,max(0,224-scaled.shape[0])),
                                          (0,max(0,224-scaled.shape[1])),(0,0)),mode="reflect")
                values,_=encode(model,scaled)
                variants[name]=dict(values=values,shape_yx=list(scaled.shape[:2]))
            case=Case(working,coarse,support,tokens,mpp,slide.native_mpp_xy,support_native,
                      palette=PALETTE,variants=variants)
            case.validate(minimum_tiles=1,minimum_groups=1)
            rgb,conversion=convert_rgb(tokens,coarse,PALETTE,variants,allow_local_fallback=True)
            if conversion["conversion"]!="RGB9+19":fallback+=1
            regions,appearance=refine_appearance(rgb,coarse,support,mpp,he=working,
                output_shape_yx=support_native.shape,output_tissue=support_native)
            local_ids=np.unique(regions[regions>0])
            native_core=regions[core]
            sums,mass=exact_core_pool(tokens,native_core,window,working.shape[:2])
            group_sums+=sums;group_mass+=mass
            mask[window.core_y:window.core_y+window.core_height,
                 window.core_x:window.core_x+window.core_width]=native_core
            mask.flush()
            record=dict(index=index,status="complete",
                native_window_xywh=[window.x,window.y,window.width,window.height],
                native_core_xywh=[window.core_x,window.core_y,window.core_width,window.core_height],
                working_shape_yx=list(working.shape[:2]),working_mpp_xy=mpp.tolist(),
                eligible_tiles=eligible,groups=local_ids.tolist(),
                mass=mass.tolist(),sums=sums.tolist(),
                conversion=conversion,appearance=appearance,
                seconds=monotonic()-t)
            atomic_json(a.output/f"window-{index:03d}.json",record)
            processed+=1
            print(json.dumps(dict(index=index,status="complete",eligible_tiles=eligible,
                groups=local_ids.tolist(),seconds=round(monotonic()-t,2),fallback=fallback)),flush=True)
            del native,working,support_native,support,tokens,coarse,variants,case,rgb,regions,native_core
    mask.flush()
    summary=dict(**manifest,processed=processed,background_windows=skipped,
        conversion_fallback_windows=fallback,group_mass=group_mass.tolist(),
        group_sums=group_sums.tolist(),wall_seconds=monotonic()-started,
        all_windows_completed=processed+skipped==len(windows))
    atomic_json(a.output/("GEOMETRY_COMPLETE.json" if summary["all_windows_completed"]
                          else "PILOT_COMPLETE.json"),summary)
    print(json.dumps(dict(stage="complete",processed=processed,background=skipped,
        fallback=fallback,all_windows_completed=summary["all_windows_completed"],
        wall_seconds=summary["wall_seconds"])),flush=True)


if __name__=="__main__":main()
