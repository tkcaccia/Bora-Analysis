"""Freeze annotation-free Bora regions and their full Virchow2 evidence.

Coarse grouping uses unsupervised physical-tile features. The existing #19
RGB conversion is fitted to those anonymous groups, then Bora-AF refines them.
Semantic tissue identity is assigned in a later stage to connected regions.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
from scipy import sparse,ndimage as ndi
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
import tifffile
from bora.appearance import refine_appearance
from experiments.virchow2_multi.methods import Context,run_method
from experiments.bora_supervised.registry import load,save,sha


@dataclass(frozen=True)
class InferenceCase:
    case: str
    image: str
    image_sha256: str
    tokens: str
    tokens_sha256: str
    tile_shards: str
    tissue_working: str
    tissue_native: str
    working_mpp_xy: tuple
    native_mpp_xy: tuple
    native_shape_yx: tuple


def sanitize(row,registry_root):
    """The geometry worker receives no reference path, class names or old RGB."""
    return InferenceCase(row['case'],row['original_path'],row['original_sha256'],
        row['feature']['path'],row['feature']['sha256'],str(Path(registry_root)/'cases'/row['case']),
        row['tissue_support_path'],str(Path(row['original_path']).parent/'tissue.native.tif'),
        tuple(row['sample']['working_mpp_xy']),tuple(row['sample']['native_mpp_xy']),tuple(row['sample']['native_shape_yx']))


def components(mask):
    regions=np.zeros(mask.shape,np.int32);groups=[];offset=0
    for g in np.unique(mask):
        if g==0:continue
        cc,n=ndi.label(mask==g,np.ones((3,3)));at=cc>0;regions[at]=cc[at]+offset
        groups.extend([int(g)]*n);offset+=n
    return regions,np.asarray(groups,np.uint8)


def grouping_observations(boxes,tissue_fraction,mpp_xy,tile_um=45.):
    """Fit groups only on substantial physical fields; predict every edge tile.

    A high tissue fraction in a clipped 11x1 pixel corner does not make that
    corner a complete microscopy observation. Match the existing conversion's
    80% per-axis edge admission, in physical rather than nominal pixel units.
    """
    dimensions_um=np.asarray(boxes)[:,2:4]*np.asarray(mpp_xy)
    return (np.asarray(tissue_fraction)>=.5)&np.all(dimensions_um>=.8*tile_um,axis=1)


def overlap(regions,source_ids,n_columns):
    """Exact native-pixel area overlap; region0 (glass) is never a prediction unit."""
    valid=regions.ravel()>0
    return sparse.coo_matrix((np.ones(int(valid.sum())),
        (regions.ravel()[valid]-1,source_ids.ravel()[valid])),
        shape=(int(regions.max()),n_columns)).tocsr()


def token_evidence(tokens,regions,working_shape):
    h,w=regions.shape;gh,gw,columns=tokens.shape
    yy=np.minimum(((np.arange(h)+.5)*working_shape[0]/h/14).astype(int),gh-1)
    xx=np.minimum(((np.arange(w)+.5)*working_shape[1]/w/14).astype(int),gw-1)
    index=yy[:,None]*gw+xx[None,:]
    weights=overlap(regions,index,gh*gw);mass=np.asarray(weights.sum(1)).ravel()
    x=np.asarray(tokens,dtype=np.float32).reshape(-1,columns)
    mean=(weights@x)/mass[:,None]
    variance=np.maximum((weights@(x*x))/mass[:,None]-mean*mean,0)
    return mean.astype(np.float32),np.sqrt(variance).astype(np.float32),mass,weights


def freeze(case,output):
    output=Path(output);folder=output/'cases'/case.case;done=folder/'GEOMETRY_FROZEN.json'
    if done.exists():
        record=load(done)
        for name,h in record['files_sha256'].items():
            if sha(folder/name)!=h:raise ValueError('Frozen region geometry changed')
        return record
    folder.mkdir(parents=True,exist_ok=True);began=time.monotonic()
    if sha(case.image)!=case.image_sha256 or sha(case.tokens)!=case.tokens_sha256:raise ValueError('Inference input changed')
    he=np.asarray(Image.open(case.image).convert('RGB'));tokens=np.load(case.tokens,mmap_mode='r')
    if tokens.ndim!=3 or tokens.shape[-1]!=1280:raise ValueError('Full unselected Virchow2 tokens required')
    tissue=tifffile.imread(case.tissue_working)>0;native_tissue=tifffile.imread(case.tissue_native)>0
    shard=Path(case.tile_shards);features=np.load(shard/'mean.npy').astype(np.float64)
    with np.load(shard/'units.npz') as u:boxes=u['boxes'];fraction=u['tissue_fraction']
    known=grouping_observations(boxes,fraction,case.working_mpp_xy)
    if known.sum()<16:raise ValueError('Fewer than16 supported physical coarse tiles')
    clustering_started=time.monotonic()
    pca=PCA(n_components=min(16,int(known.sum())-1),svd_solver='full').fit(features[known])
    retained=pca.explained_variance_>max(float(pca.explained_variance_[0])*1e-4,1e-8)
    if not retained.any():raise ValueError('No nonconstant embedding directions for tissue grouping')
    z=pca.transform(features)[:,retained]/np.sqrt(pca.explained_variance_[retained])
    model=KMeans(n_clusters=4,n_init=10,random_state=314159,algorithm='lloyd').fit(z[known])
    assignment=model.predict(z).astype(np.uint8)+1
    coarse=np.zeros(he.shape[:2],np.uint8);unit_ids=np.empty(he.shape[:2],np.int32)
    for i,((x,y,w,h),g) in enumerate(zip(boxes,assignment)):
        coarse[y:y+h,x:x+w]=g;unit_ids[y:y+h,x:x+w]=i
    coarse[~tissue]=0;cluster_seconds=time.monotonic()-clustering_started
    if len(np.unique(coarse[coarse>0]))<2:raise ValueError('Anonymous grouping collapsed to one group')
    palette=np.array([[220,45,65],[55,165,75],[55,110,230],[200,100,220]],np.uint8)
    conversion_started=time.monotonic()
    context=Context(he,coarse,tokens,palette,analysis_support=tissue)
    rgb,_,details,_,_=run_method(19,context,{},return_probabilities=True)
    conversion_seconds=time.monotonic()-conversion_started
    refinement_started=time.monotonic()
    refined,solver=refine_appearance(rgb,coarse,tissue,case.working_mpp_xy,he=he,method='af',
        output_shape_yx=case.native_shape_yx,output_tissue=native_tissue)
    bora_seconds=time.monotonic()-refinement_started
    pooling_started=time.monotonic();regions,groups=components(refined)
    if regions.max()==0:raise ValueError('Native background removal left no tissue region')
    mean,std,area,token_weights=token_evidence(tokens,regions,he.shape[:2])
    native_units=cv2.resize(unit_ids,case.native_shape_yx[::-1],interpolation=cv2.INTER_NEAREST)
    tile_weights=overlap(regions,native_units,len(boxes))
    # Neither semantic classification nor annotation access occurs here.
    tifffile.imwrite(folder/'regions.native.tif',regions,compression='deflate')
    tifffile.imwrite(folder/'groups.native.tif',refined,compression='deflate')
    tifffile.imwrite(folder/'coarse.working.tif',coarse,compression='deflate')
    Image.fromarray(rgb).save(folder/'converted19.png')
    np.save(folder/'mean.npy',mean);np.save(folder/'std.npy',std)
    sparse.save_npz(folder/'tile_overlap.npz',tile_weights);sparse.save_npz(folder/'token_overlap.npz',token_weights)
    bboxes=[]
    for index,slices in enumerate(ndi.find_objects(regions),1):
        if slices is None:raise ValueError('Region IDs are not contiguous')
        ys,xs=slices;bboxes.append([xs.start,ys.start,xs.stop-xs.start,ys.stop-ys.start])
    save(folder/'REGIONS.json',dict(regions=[dict(id=i+1,anonymous_group=int(g),area_pixels=int(a),
        area_um2=float(a*np.prod(case.native_mpp_xy)),bbox_native_xywh=b) for i,(g,a,b) in enumerate(zip(groups,area,bboxes))],
        prediction_unit='Connected Bora region; every positive native tissue pixel belongs to exactly one region',
        full_unselected_feature_columns=tokens.shape[-1],semantic_labels_assigned=False))
    files=['regions.native.tif','groups.native.tif','coarse.working.tif','converted19.png','mean.npy','std.npy','tile_overlap.npz','token_overlap.npz','REGIONS.json']
    record=dict(case=case.case,regions=len(mean),fine_annotation_read=False,old_reference_derived_coarse_or_rgb_read=False,
        grouping='Within-image PCA up to16 nondegenerate components, whitening, fixed4-cluster KMeans; all grouping is unsupervised',
        rgb_conversion='Unchanged algorithm19; its coarse classes are anonymous unsupervised groups, not reference-derived tissue labels',
        selected_conversion_dimensions=context.selected.tolist(),conversion_details=details,bora_details=solver,
        timing=dict(cluster_seconds=cluster_seconds,conversion19_seconds=conversion_seconds,bora_seconds=bora_seconds,
                    component_and_feature_pooling_seconds=time.monotonic()-pooling_started,total_cached_seconds=time.monotonic()-began),
        files_sha256={f:sha(folder/f) for f in files},input_image_sha256=case.image_sha256,input_tokens_sha256=case.tokens_sha256,
        geometry_code_sha256=sha(Path(__file__)))
    record['grouping_observations']=dict(admitted=int(known.sum()),all_tiles=len(boxes),
        minimum_tissue_fraction=.5,minimum_extent_each_axis_um=36.,
        clipped_tiles_predicted=True)
    save(done,record);print(__import__('json').dumps(dict(frozen=case.case,regions=len(mean),seconds=record['timing']['total_cached_seconds'])),flush=True)
    return record


def run(registry_root,output,cases=None):
    registry_root,output=Path(registry_root).resolve(),Path(output).resolve();registry=load(registry_root/'REGISTRY.json')
    selected=[r for r in registry['cases'] if cases is None or r['case'] in cases]
    protocol=dict(task='Identify each tissue segment after Bora, using one top1 and top5 per frozen connected region',
        source_registry_sha256=sha(registry_root/'REGISTRY.json'),cases=[r['case'] for r in selected],
        geometry_annotation_free=True,conversion=19,coarse_groups=4,physical_coarse_tile_um=45,
        grouping_minimum_extent_each_axis_um=36.,clipped_tiles_predicted=True,
        primary_refiner='Bora-AF',threads=2,
        no_reference_conditioned_rgb_cache=True,reference_paths_given_to_worker=False,
        old_test_reuse='The previous tile/superpixel test was already scored. This follow-up retains train-group separation but is an exploratory reused-test analysis.',
        planned_region_classifiers=['Area-weighted vote of native tile ranks','Pooled full-feature region PLS-LDA',
                                   'Region mean and feature dispersion PLS-LDA','Region composition PLS regression'],
        inference_unit='Connected native Bora region, never a tile or a superpixel',
        source_sha256=sha(Path(__file__)),conversion_source_sha256=sha(__import__('inspect').getfile(run_method)),
        refiner_source_sha256=sha(__import__('inspect').getfile(refine_appearance)))
    path=output/'PROTOCOL.json'
    if path.exists() and load(path)!=protocol:raise ValueError('Use a new geometry protocol version')
    save(path,protocol)
    records=[freeze(sanitize(row,registry_root),output) for row in selected]
    save(output/'ALL_GEOMETRY_FROZEN.json',dict(protocol_sha256=sha(path),fine_annotations_read=False,
        regions=sum(r['regions'] for r in records),cases={r['case']:sha(output/'cases'/r['case']/'GEOMETRY_FROZEN.json') for r in records}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registry-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',nargs='+');a=p.parse_args();cv2.setNumThreads(2)
    with threadpool_limits(limits=2):run(a.registry_root,a.output,a.cases)


if __name__=='__main__':main()
