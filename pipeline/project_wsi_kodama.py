#!/usr/bin/env python3
"""Extend nonspatial KODAMA-Louvain sample groups to every WSI tile.

KODAMA has no required out-of-sample transform here. A fixed 7-neighbor
feature-space vote transfers its anonymous sample groups; neither expert
labels nor tile coordinates enter the vote. Tile coordinates are used only to
rasterize the assigned groups for Bora.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20), b''):
            h.update(block)
    return h.hexdigest()


def project(pooled_dir:Path, input_dir:Path, cluster_dir:Path, output:Path)->dict:
    pooled_dir=pooled_dir.resolve();input_dir=input_dir.resolve();cluster_dir=cluster_dir.resolve();output=output.resolve()
    if output.exists():raise FileExistsError(output)
    pooled_receipt=json.loads((pooled_dir/'POOLED_COMPLETE.json').read_text())
    input_receipt=json.loads((input_dir/'INPUT_COMPLETE.json').read_text())
    group_receipt=json.loads((cluster_dir/'CLUSTERS_COMPLETE.json').read_text())
    if input_receipt['pooled_receipt_sha256']!=sha256(pooled_dir/'POOLED_COMPLETE.json'):
        raise ValueError('KODAMA input belongs to different pooled features')
    if group_receipt['spatial_coordinates_used'] or group_receipt['expert_annotation_used']:
        raise ValueError('KODAMA and clustering must exclude spatial/expert inputs')
    for name,expected in input_receipt['files'].items():
        if sha256(input_dir/name)!=expected:raise ValueError('KODAMA input checksum differs: '+name)
    sample=np.load(input_dir/'sample_indices.npy',allow_pickle=False)
    all_pca=np.load(input_dir/'all_pca.npy',mmap_mode='r',allow_pickle=False)
    valid=np.load(pooled_dir/'valid.npy',allow_pickle=False)
    table=np.load(pooled_dir/'tile_table.npy',mmap_mode='r',allow_pickle=False)
    rows=pd.read_csv(cluster_dir/'sample_clusters.csv')
    labels=rows['group'].to_numpy(dtype=np.int32)
    if (len(sample)!=len(labels) or all_pca.shape!=(pooled_receipt['grid_tiles'],30)
            or len(valid)!=len(all_pca) or table.shape!=(len(all_pca),5)
            or len(np.unique(labels))<2 or np.any(labels<1) or np.any(labels>255)):
        raise ValueError('KODAMA labels and WSI tile features disagree')
    x=np.asarray(all_pca[sample],np.float32)
    with threadpool_limits(limits=2):
        model=KNeighborsClassifier(n_neighbors=7,weights='distance',n_jobs=2).fit(x,labels)
        if min(np.bincount(labels)[1:])>=5:
            cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=20261027)
            held=cross_val_predict(KNeighborsClassifier(n_neighbors=7,weights='distance',n_jobs=2),
                                   x,labels,cv=cv,n_jobs=1)
            cv_accuracy=float(accuracy_score(labels,held))
            cv_balanced=float(balanced_accuracy_score(labels,held))
        else:
            cv_accuracy=cv_balanced=None
        ids=np.flatnonzero(valid)
        projected=np.zeros(len(valid),np.uint8)
        confidence=np.zeros(len(valid),np.float32)
        for start in range(0,len(ids),2048):
            tile_ids=ids[start:start+2048]
            proba=model.predict_proba(np.asarray(all_pca[tile_ids],np.float32))
            projected[tile_ids]=model.classes_[np.argmax(proba,axis=1)].astype(np.uint8)
            confidence[tile_ids]=np.max(proba,axis=1).astype(np.float32)
    width,height=map(int,pooled_receipt['native_tile_wh'])
    xs=np.asarray(table[ids,1],np.int64)
    ys=np.asarray(table[ids,2],np.int64)
    cols=xs//width; grid_width=int(cols.max()+1)
    rows_idx=ys//height; grid_height=int(rows_idx.max()+1)
    if grid_width*grid_height!=len(valid) or not np.array_equal(rows_idx*grid_width+cols,ids):
        raise ValueError('Feature tile table is not a complete row-major WSI grid')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.wsi-kodama-projection-',dir=output.parent) as tmp:
        folder=Path(tmp)
        np.save(folder/'tile_groups.npy',projected.reshape(grid_height,grid_width))
        np.save(folder/'tile_confidence.npy',confidence.reshape(grid_height,grid_width))
        counts=np.bincount(projected[ids],minlength=int(labels.max())+1)[1:]
        record=dict(format='bora-wsi-kodama-projection-v1',
                    source_image_sha256=pooled_receipt['image_sha256'],
                    pooled_receipt_sha256=sha256(pooled_dir/'POOLED_COMPLETE.json'),
                    kodama_input_sha256=sha256(input_dir/'INPUT_COMPLETE.json'),
                    cluster_receipt_sha256=sha256(cluster_dir/'CLUSTERS_COMPLETE.json'),
                    sample_clusters_sha256=sha256(cluster_dir/'sample_clusters.csv'),
                    feature_space='Virchow2 pooled tile features, standardized PCA30',
                    out_of_sample_assignment='7-neighbor inverse-distance vote in PCA30',
                    spatial_coordinates_used_for_clustering_or_assignment=False,
                    expert_annotation_used=False,
                    sample_count=len(sample),tile_count=int(valid.sum()),
                    tile_grid_shape_yx=[grid_height,grid_width],
                    native_tile_wh=[width,height],
                    groups=len(np.unique(labels)),group_counts=counts.tolist(),
                    sample_extension_cv_accuracy=cv_accuracy,
                    sample_extension_cv_balanced_accuracy=cv_balanced,
                    median_tile_vote_confidence=float(np.median(confidence[ids])),
                    files={name:sha256(folder/name) for name in ('tile_groups.npy','tile_confidence.npy')})
        (folder/'PROJECTED_COMPLETE.json').write_text(json.dumps(record,indent=2)+'\n')
        os.replace(folder,output)
    return record


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pooled',type=Path,required=True)
    p.add_argument('--kodama-input',type=Path,required=True)
    p.add_argument('--clusters',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(project(a.pooled,a.kodama_input,a.clusters,a.output),indent=2))

if __name__=='__main__':main()
