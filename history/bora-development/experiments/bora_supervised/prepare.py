"""Create immutable feature shards and TRAIN/VALIDATION target shards only.

Test annotations are not opened here. Their paths/hashes stay in the registry
for the separate final evaluator after all candidate predictions are frozen.
"""
import argparse
from collections import Counter
from pathlib import Path
import time
import numpy as np
from PIL import Image
import tifffile

from .registry import load,save,sha
from .units import tile_boxes,pool_tokens,region_targets


def prepare_case(root,row,ontology):
    root=Path(root);dst=root/'cases'/row['case'];done=dst/'PREPARED.json'
    if done.exists():
        r=load(done)
        if r['registry_sha256']!=sha(root/'REGISTRY.json'):raise ValueError('Changed registry')
        for name,h in r['files_sha256'].items():
            if sha(dst/name)!=h:raise ValueError('Prepared shard changed')
        return r
    began=time.monotonic();dst.mkdir(parents=True,exist_ok=True)
    feature=row['feature'];sample=row['sample']
    for p,h in [(feature['path'],feature['sha256']),
                (row['tissue_support_path'],row['tissue_support_sha256'])]:
        if sha(p)!=h:raise ValueError('Frozen input changed: '+p)
    tokens=np.load(feature['path'],mmap_mode='r')
    if tuple(tokens.shape)!=tuple(feature['shape']):raise ValueError('Feature geometry changed')
    boxes=tile_boxes(sample['working_shape_yx'],sample['working_mpp_xy'])
    mean,centre=pool_tokens(tokens,boxes,feature['patch_px'])
    if not np.isfinite(mean).all() or not np.isfinite(centre).all():raise ValueError('Nonfinite predictors')
    tissue=tifffile.imread(row['tissue_support_path'])>0
    if tuple(tissue.shape)!=tuple(sample['working_shape_yx']):raise ValueError('Tissue geometry mismatch')
    tissue_fraction=np.array([tissue[y:y+h,x:x+w].mean() for x,y,w,h in boxes],dtype=np.float32)
    np.save(dst/'mean.npy',mean);np.save(dst/'centre.npy',centre)
    np.savez(dst/'units.npz',boxes=boxes,tissue_fraction=tissue_fraction)
    annotated=False
    if row['split']!='test':
        if sha(row['reference_path'])!=row['reference_sha256']:raise ValueError('Annotation changed')
        reference=tifffile.imread(row['reference_path'])
        fractions,coverage=region_targets(reference,boxes,sample['working_shape_yx'],
            row['ontology_raw_to_role'],len(ontology['roles']))
        np.savez(dst/'targets.npz',fractions=fractions,coverage=coverage)
        annotated=True
    files=['mean.npy','centre.npy','units.npz']+(['targets.npz'] if annotated else [])
    record=dict(case=row['case'],split=row['split'],group=row['group'],units=len(boxes),
        annotated_targets_created=annotated,reference_decoded=annotated,
        feature_columns=1280,physical_tile_um=45.,
        registry_sha256=sha(root/'REGISTRY.json'),
        files_sha256={n:sha(dst/n) for n in files},seconds=time.monotonic()-began,
        feature_contract='Full1280 frozen spatial tokens; tile means weight token-grid-cell overlap. Token receptive field is about112um, not the7um grid spacing.',
        original_coarse_or_selected_rgb_read=False)
    save(done,record)
    print(__import__('json').dumps(dict(case=row['case'],split=row['split'],units=len(boxes),targets=annotated)),flush=True)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();registry=load(a.root/'REGISTRY.json')
    records=[prepare_case(a.root,r,registry['ontology']) for r in registry['cases']]
    tests=[r for r in records if r['split']=='test']
    if any(r['reference_decoded'] for r in tests):raise ValueError('Test annotation read during preparation')
    save(a.root/'PREPARATION_COMPLETE.json',dict(cases=len(records),units=sum(r['units'] for r in records),
        split_counts=dict(Counter(r['split'] for r in records)),test_annotations_decoded=False,
        original_coarse_or_selected_rgb_read=False,registry_sha256=sha(a.root/'REGISTRY.json'),
        case_preparation_sha256={r['case']:sha(a.root/'cases'/r['case']/'PREPARED.json') for r in records}))


if __name__=='__main__':main()
