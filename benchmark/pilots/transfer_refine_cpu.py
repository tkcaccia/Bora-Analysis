"""Refine one frozen source-predicted TIGER tile map without target labels."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
import tifffile


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--registry', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    from bora.pipeline import Case, prepare, export_mask

    if a.output.exists():
        raise FileExistsError(a.output)
    root = a.root
    features = root/'features/tiger_15/approved-one-case'
    complete = json.loads((features/'FEATURES_COMPLETE.json').read_text())
    frozen = json.loads((root/'tiles/PREDICTIONS_FROZEN.json').read_text())
    plan = json.loads((root/'inputs/tiger_15/PLAN.json').read_text())
    row = next(r for r in json.loads(a.registry.read_text())['cases'] if r['case']=='tiger_15')
    assert row['split'] != 'train' and row['group'] != frozen['source_group']
    assert complete['case']=='tiger_15' and not complete['reference_read']
    assert plan['samples'][0]['target_reference_read'] is False
    assert len(plan['samples'][0]['palette']) == 2
    sample = plan['samples'][0]
    image = np.asarray(Image.open(root/'inputs/tiger_15/cases/tiger_15/original.png').convert('RGB'))
    coarse = tifffile.imread(root/'inputs/tiger_15/cases/tiger_15/coarse.tif').astype(np.uint8)
    tissue = tifffile.imread(row['tissue_support_path']) > 0
    assert tissue.shape==coarse.shape==image.shape[:2]
    base = np.load(features/'base.npy', mmap_mode='r', allow_pickle=False)
    variants={}
    for name in ('scale_1','scale_2'):
        meta=complete['files'][name]
        variants[name]=dict(values=np.load(features/(name+'.npy'),mmap_mode='r',allow_pickle=False),
                            shape_yx=meta['image_shape_yx'])
    case=Case(image,coarse,tissue,base,np.asarray(sample['working_mpp_xy']),
              np.asarray(sample['native_mpp_xy']),tissue,
              palette=np.asarray(sample['palette']),variants=variants,
              provenance=dict(source_training_image=frozen['source_case'],
                              source_training_group=frozen['source_group'],
                              predicted_tile_sha256=next(r['tile_file_sha256'] for r in frozen['records'] if r['case']=='tiger_15'),
                              fine_annotations_read=False))
    start=perf_counter()
    prepared=prepare(case)
    regions=prepared['regions']
    predicted=np.zeros(regions.shape,np.uint8)
    group_readout=[]
    for group in range(1,int(regions.max())+1):
        pix=(regions==group)&tissue
        counts=np.bincount(coarse[pix],minlength=3)
        tile_group=int(np.argmax(counts[1:])+1)
        role={1:1,2:3}[tile_group]
        predicted[pix]=role
        group_readout.append(dict(group=group,source_predicted_tile_group=tile_group,
                                  predicted_role=role,area_pixels=int(pix.sum()),
                                  tile_votes=counts.tolist()))
    a.output.mkdir(parents=True)
    Image.fromarray(prepared['rgb']).save(a.output/'converted-rgb.png')
    tifffile.imwrite(a.output/'groups.tif',regions,compression='deflate')
    tifffile.imwrite(a.output/'classified-before-export.tif',predicted,compression='deflate')
    report=export_mask(predicted,a.output,np.asarray(sample['native_mpp_xy']),
                       {1:'tumor',3:'stroma'})
    # export_mask is the standard Neighbor-context path; this pilot substitutes
    # an area-majority tile readout and must state that provenance explicitly.
    geojson_path=a.output/'mask.geojson'
    geojson=json.loads(geojson_path.read_text())
    for feature in geojson['features']:
        feature['properties']['tissue_identity_source']='Source-trained fastPLS tile predictions; area-majority per refined group'
    temporary=geojson_path.with_suffix('.geojson.tmp')
    temporary.write_text(json.dumps(geojson,indent=2)+'\n')
    os.replace(temporary,geojson_path)
    report.update(case='tiger_15',source_case=frozen['source_case'],
                  segment_classification='Area-majority of source-trained fastPLS tile predictions within each refined group',
                  group_readout=group_readout,elapsed_cpu_seconds=perf_counter()-start,
                  reference_read=False,appearance=prepared['details']['appearance'],
                  conversion=prepared['details']['conversion'])
    (a.output/'REFINEMENT_COMPLETE.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('case','elapsed_cpu_seconds','segment_classification')}))


if __name__=='__main__':
    main()
