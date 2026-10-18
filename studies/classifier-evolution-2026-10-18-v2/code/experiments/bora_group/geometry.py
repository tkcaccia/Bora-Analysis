"""Preserve the previous expert-derived tiling and RGB; anonymize IDs only."""
import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi,sparse
import tifffile
from threadpoolctl import threadpool_limits
from bora.appearance import refine_appearance
from experiments.bora_supervised.registry import load,save,sha
from experiments.bora_supervised.units import tile_boxes
from experiments.bora_segment.coarse_example import anonymize
from experiments.bora_segment.geometry import token_evidence,overlap

UNIT='Whole anonymous Bora group, including all disconnected pieces'


@dataclass(frozen=True)
class GroupCase:
    case:str
    image:str
    image_sha256:str
    coarse:str
    coarse_sha256:str
    rgb:str
    rgb_sha256:str
    tokens:str
    tokens_sha256:str
    tissue_working:str
    tissue_native:str
    working_mpp_xy:tuple
    native_mpp_xy:tuple
    native_shape_yx:tuple


def sanitize(row):
    src=Path(row['original_path']).parent;inputs=load(src/'INPUTS_FROZEN.json')['files_sha256']
    return GroupCase(row['case'],row['original_path'],row['original_sha256'],str(src/'coarse.tif'),
        row['sample']['coarse_sha256'],str(src/'ensemble/representation.png'),inputs['ensemble/representation.png'],
        row['feature']['path'],row['feature']['sha256'],row['tissue_support_path'],str(src/'tissue.native.tif'),
        tuple(row['sample']['working_mpp_xy']),tuple(row['sample']['native_mpp_xy']),tuple(row['sample']['native_shape_yx']))


def compact_groups(anonymous_mask):
    """Keep disconnected pieces together; omit an eliminated output group."""
    surviving=np.unique(anonymous_mask[anonymous_mask>0]).astype(int)
    lookup=np.zeros(int(anonymous_mask.max())+1,np.int32)
    lookup[surviving]=np.arange(1,len(surviving)+1)
    return lookup[anonymous_mask],surviving


def freeze(case,output):
    output=Path(output);folder=output/'cases'/case.case;done=folder/'GEOMETRY_FROZEN.json'
    if done.exists():
        previous=load(done)
        for name,h in previous['files_sha256'].items():
            if sha(folder/name)!=h:raise ValueError('Frozen group geometry changed')
        return previous
    folder.mkdir(parents=True,exist_ok=True);began=time.monotonic()
    for file,h in [(case.image,case.image_sha256),(case.coarse,case.coarse_sha256),(case.rgb,case.rgb_sha256),(case.tokens,case.tokens_sha256)]:
        if sha(file)!=h:raise ValueError('Previous benchmark input changed')
    he=np.asarray(Image.open(case.image).convert('RGB'));coarse=tifffile.imread(case.coarse)
    rgb=np.asarray(Image.open(case.rgb).convert('RGB'));anonymous,restore=anonymize(coarse,case.image_sha256)
    tissue=tifffile.imread(case.tissue_working)>0;native_tissue=tifffile.imread(case.tissue_native)>0
    before=time.monotonic();refined,details=refine_appearance(rgb,anonymous,tissue,case.working_mpp_xy,he=he,
        method='af',output_shape_yx=case.native_shape_yx,output_tissue=native_tissue)
    bora_seconds=time.monotonic()-before
    # Independent original-ID solve verifies that only the names changed.
    original,_=refine_appearance(rgb,coarse,tissue,case.working_mpp_xy,he=he,method='af',
        output_shape_yx=case.native_shape_yx,output_tissue=native_tissue)
    different=int(np.count_nonzero(restore[refined]!=original))
    if different:raise ValueError('Anonymization changed Bora geometry: '+str(different)+'pixels')
    pool_started=time.monotonic();regions,surviving=compact_groups(refined)
    if not len(surviving):raise ValueError('No refined tissue group')
    tokens=np.load(case.tokens,mmap_mode='r')
    if tokens.ndim!=3 or tokens.shape[-1]!=1280:raise ValueError('Full unselected Virchow2 spatial tokens required')
    mean,std,mass,token_weights=token_evidence(tokens,regions,he.shape[:2])
    boxes=tile_boxes(he.shape[:2],case.working_mpp_xy);tile_ids=np.zeros(he.shape[:2],np.int32)
    for i,(x,y,w,h) in enumerate(boxes):tile_ids[y:y+h,x:x+w]=i
    native_ids=cv2.resize(tile_ids,case.native_shape_yx[::-1],interpolation=cv2.INTER_NEAREST)
    tile_weights=overlap(regions,native_ids,len(boxes))
    tifffile.imwrite(folder/'regions.native.tif',regions,compression='deflate')
    tifffile.imwrite(folder/'groups.native.tif',refined,compression='deflate')
    tifffile.imwrite(folder/'coarse.working.tif',anonymous,compression='deflate')
    shutil.copyfile(case.rgb,folder/'converted.png')
    np.save(folder/'mean.npy',mean);np.save(folder/'std.npy',std)
    sparse.save_npz(folder/'tile_overlap.npz',tile_weights);sparse.save_npz(folder/'token_overlap.npz',token_weights)
    information=[]
    for i,(g,area) in enumerate(zip(surviving,mass),1):
        yy,xx=np.where(regions==i)
        information.append(dict(id=i,anonymous_group=int(g),area_pixels=int(area),area_um2=float(area*np.prod(case.native_mpp_xy)),
            bbox_native_xywh=[int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)],
            disconnected_pieces=int(ndi.label(regions==i,np.ones((3,3)))[1])))
    save(folder/'REGIONS.json',dict(regions=information,prediction_unit=UNIT,semantic_labels_assigned=False,
        full_unselected_feature_columns=1280,connected_components_are_not_separate_classification_units=True))
    # Numeric reversible key is segregated from predictor feature files.
    key=output/'evaluation_keys'/(case.case+'.json')
    save(key,dict(case=case.case,source_class_by_output_group=[int(restore[g]) for g in surviving],
        input_source_ids=restore[1:].tolist(),input_anonymous_ids=list(range(1,len(restore))),
        eliminated_anonymous_ids=sorted(set(range(1,len(restore)))-set(surviving.tolist()))))
    files=['regions.native.tif','groups.native.tif','coarse.working.tif','converted.png','mean.npy','std.npy','tile_overlap.npz','token_overlap.npz','REGIONS.json']
    record=dict(case=case.case,regions=len(surviving),prediction_unit=UNIT,geometry_mode='previous expert-derived coarse groups with deidentified IDs',
        input_groups=len(restore)-1,eliminated_groups=len(restore)-1-len(surviving),
        new_clustering_used=False,previous_tile_assignments_preserved_exactly=True,
        coarse_reference_derived=True,rgb_reference_conditioned=True,rgb_recomputed=False,rgb_preview_file='converted.png',
        fine_annotation_read=False,old_reference_derived_coarse_or_rgb_read=True,
        original_vs_anonymous_bora_differing_pixels=different,evaluation_key_sha256=sha(key),
        native_mpp_xy=list(case.native_mpp_xy),working_mpp_xy=list(case.working_mpp_xy),native_shape_yx=list(case.native_shape_yx),
        input_coarse_sha256=case.coarse_sha256,input_rgb_sha256=case.rgb_sha256,
        input_image_sha256=case.image_sha256,input_tokens_sha256=case.tokens_sha256,bora_details=details,
        timing=dict(cluster_seconds=0.,conversion19_seconds=0.,bora_seconds=bora_seconds,
            component_and_feature_pooling_seconds=time.monotonic()-pool_started,
            total_cached_seconds=None,anonymization_parity_audit_excluded=True),
        files_sha256={name:sha(folder/name) for name in files},geometry_code_sha256=sha(Path(__file__)))
    # Cost is the measured refinement plus feature/output preparation. Exclude
    # the second solve, which is solely an anonymization invariance audit.
    record['timing']['total_cached_seconds']=bora_seconds+record['timing']['component_and_feature_pooling_seconds']
    record['audit_and_io_wall_seconds']=time.monotonic()-began
    save(done,record);print(__import__('json').dumps(dict(frozen=case.case,input_groups=len(restore)-1,output_groups=len(surviving),identical_geometry=different==0)),flush=True)
    return record


def run(registry_root,output,cases=None):
    registry_root,output=Path(registry_root),Path(output);registry=load(registry_root/'REGISTRY.json')
    rows=[r for r in registry['cases'] if cases is None or r['case'] in cases]
    protocol=dict(task='Expert-derived tile masks, deidentified IDs, unchanged9+19 RGB, Bora-AF, supervised whole-group identification',
        prediction_unit=UNIT,source_registry_sha256=sha(registry_root/'REGISTRY.json'),cases=[r['case'] for r in rows],
        input_tile_assignments='Exact previous benchmark assignments; no KMeans or new grouping',
        input_rgb='Exact previous frozen ensemble bytes; no new RGB fitting',source_names_hidden_from_head=True,
        model_features='Full unselected Virchow2 features; no source class ID, A/B ID, RGB palette or reference vocabulary',
        controlled_reference_derived_benchmark=True,autonomous_segmentation_claim=False,threads=2,
        code_sha256=sha(Path(__file__)),old_test_reuse=True)
    path=output/'PROTOCOL.json'
    if path.exists() and load(path)!=protocol:raise ValueError('Use a new immutable group geometry version')
    save(path,protocol);records=[freeze(sanitize(r),output) for r in rows]
    save(output/'ALL_GEOMETRY_FROZEN.json',dict(prediction_unit=UNIT,regions=sum(r['regions'] for r in records),
        cases={r['case']:sha(output/'cases'/r['case']/'GEOMETRY_FROZEN.json') for r in rows},
        fine_annotations_read=False,coarse_reference_derived=True,new_clustering_used=False,
        all_tile_partitions_identical=True,all_rgb_bytes_unchanged=True,all_anonymous_bora_geometry_identical=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('registry-root','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cases',nargs='+');a=p.parse_args();cv2.setNumThreads(2)
    with threadpool_limits(limits=2):run(a.registry_root,a.output,a.cases)


if __name__=='__main__':main()
