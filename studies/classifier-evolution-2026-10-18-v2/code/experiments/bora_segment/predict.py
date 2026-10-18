"""Identify frozen Bora tissue segments without accepting an annotation input."""
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
import tifffile
from threadpoolctl import threadpool_limits
from experiments.bora_supervised.registry import load,save,sha
from experiments.bora_supervised.predict import load_model
from .train import region_features
from .labels import semantic_mask


def validate_prediction_unit(record,metadata):
    """Match head to frozen units and disclose reference-derived coarse inputs."""
    if record['fine_annotation_read']:
        raise ValueError('Geometry must freeze without reading fine annotations')
    unit=metadata['training_unit']
    if unit=='Whole anonymous Bora group, including all disconnected pieces':
        if record.get('prediction_unit')!=unit:
            raise ValueError('Whole-group geometry is required for this head')
        if record.get('new_clustering_used') or not record.get('previous_tile_assignments_preserved_exactly'):
            raise ValueError('Whole-group inputs must preserve the previous tiling')
        return unit
    if unit!='Connected Bora region':
        raise ValueError('A tissue group or region classifier is required')
    if record['old_reference_derived_coarse_or_rgb_read']:
        raise ValueError('This historical region head requires annotation-free geometry')
    return unit


def run(model,geometry_case,output,mpp_xy):
    model,geometry_case,output=map(Path,(model,geometry_case,output))
    if output.exists():raise ValueError('Preserve previous prediction')
    record=load(geometry_case/'GEOMETRY_FROZEN.json')
    for name,expected in record['files_sha256'].items():
        if sha(geometry_case/name)!=expected:raise ValueError('Frozen region geometry or evidence changed')
    predict,metadata,startup=load_model(model)
    unit=validate_prediction_unit(record,metadata)
    mpp=np.asarray(mpp_xy,dtype=float)
    if mpp.shape!=(2,) or not np.isfinite(mpp).all() or np.any(mpp<=0):raise ValueError('Positive native pixel spacing required')
    info=load(geometry_case/'REGIONS.json')['regions']
    if not all(np.isclose(r['area_pixels']*np.prod(mpp),r['area_um2']) for r in info):
        raise ValueError('Pixel spacing differs from frozen native region geometry')
    began=time.monotonic();x=region_features(geometry_case.parent.parent,geometry_case.name,metadata['feature'],mpp)
    feature_seconds=time.monotonic()-began
    predict(x);times=[]
    for _ in range(5):
        began=time.monotonic();ranks=predict(x);times.append(time.monotonic()-began)
    regions=tifffile.imread(geometry_case/'regions.native.tif');mask=semantic_mask(regions,ranks);output.mkdir(parents=True)
    tifffile.imwrite(output/'top1.native.tif',mask,compression='deflate')
    np.savez_compressed(output/'ranked_regions.npz',top5=ranks)
    roles=metadata['ontology']['roles']
    save(output/'tissue_segments.json',dict(model=metadata['name'],prediction_unit=unit,
        region_geometry_sha256=sha(geometry_case/'regions.native.tif'),geometry_unchanged=True,annotation_input_used=False,
        coarse_reference_derived=record.get('coarse_reference_derived',False),
        rgb_reference_conditioned=record.get('rgb_reference_conditioned',False),
        native_backend=metadata['engine'],startup_refit_seconds=startup,
        feature_loading_and_context_seconds=feature_seconds,warm_head_seconds=float(np.median(times)),
        ranks_are_calibrated_probabilities=False,
        segments=[dict(**item,top1_id=int(rank[0]),top1_name=roles[rank[0]-1],
            top5=[dict(rank=j+1,id=int(g),name=roles[g-1]) for j,g in enumerate(rank)]) for item,rank in zip(info,ranks)]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','geometry-case','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--native-mpp-xy',type=float,nargs=2,required=True);a=p.parse_args();cv2.setNumThreads(2)
    with threadpool_limits(limits=2):run(a.model,a.geometry_case,a.output,a.native_mpp_xy)


if __name__=='__main__':main()
