"""Attach training labels to already frozen Bora segments; keep test sealed."""
import argparse
from pathlib import Path
import numpy as np
import tifffile
from experiments.bora_supervised.registry import load,save,sha


def target_counts(regions,roles,n_roles):
    count=int(regions.max());width=n_roles+1
    counts=np.bincount(regions.ravel().astype(np.int64)*width+roles.ravel(),minlength=(count+1)*width).reshape(count+1,width)[1:]
    area=counts.sum(1);annotated=counts[:,1:].sum(1)
    fractions=counts[:,1:]/np.maximum(annotated[:,None],1)
    return fractions.astype(np.float32),(annotated/np.maximum(area,1)).astype(np.float32)


def run(registry_root,geometry,output):
    registry_root,geometry,output=map(lambda p:Path(p).resolve(),(registry_root,geometry,output))
    registry=load(registry_root/'REGISTRY.json');frozen=load(geometry/'ALL_GEOMETRY_FROZEN.json')
    if set(frozen['cases'])!={r['case'] for r in registry['cases']}:raise ValueError('Freeze all geometry before attaching any labels')
    protocol=dict(registry_sha256=sha(registry_root/'REGISTRY.json'),geometry_frozen_sha256=sha(geometry/'ALL_GEOMETRY_FROZEN.json'),
        label_unit='A connected native Bora segment',test_annotations_decoded=False,
        coverage='At least50percent of region area has a mapped annotation for a training or scored unit; ignored pixels never become stroma.',
        geometry_or_rgb_refitted_after_labels=False,code_sha256=sha(Path(__file__)))
    if (output/'PROTOCOL.json').exists() and load(output/'PROTOCOL.json')!=protocol:raise ValueError('Use new target version')
    if (output/'TARGETS_FROZEN.json').exists():
        previous=load(output/'TARGETS_FROZEN.json')
        if previous['protocol_sha256']!=sha(output/'PROTOCOL.json'):raise ValueError('Target protocol changed')
        for case,record in previous['cases'].items():
            if sha(output/'cases'/case/'targets.npz')!=record['sha256']:raise ValueError('Target shard changed')
        return
    save(output/'PROTOCOL.json',protocol);records={}
    for row in registry['cases']:
        case=row['case'];src=geometry/'cases'/case
        if sha(src/'GEOMETRY_FROZEN.json')!=frozen['cases'][case]:raise ValueError('Geometry manifest changed')
        record=load(src/'GEOMETRY_FROZEN.json')
        path=src/'regions.native.tif'
        if sha(path)!=record['files_sha256']['regions.native.tif']:raise ValueError('Region boundaries changed')
        if row['split']=='test':
            if (output/'cases'/case/'targets.npz').exists():raise ValueError('Test target unexpectedly prepared')
            continue
        if sha(row['reference_path'])!=row['reference_sha256']:raise ValueError('Training/validation reference changed')
        native=tifffile.imread(row['reference_path']);roles=np.zeros(native.shape,np.uint8)
        for code,g in row['ontology_raw_to_role'].items():roles[native==int(code)]=g
        regions=tifffile.imread(path)
        if regions.shape!=roles.shape:raise ValueError('Native annotation/region geometry differs')
        fractions,coverage=target_counts(regions,roles,len(registry['ontology']['roles']))
        dst=output/'cases'/case;dst.mkdir(parents=True,exist_ok=True)
        np.savez(dst/'targets.npz',fractions=fractions,coverage=coverage)
        records[case]=dict(split=row['split'],regions=len(fractions),sha256=sha(dst/'targets.npz'))
    save(output/'TARGETS_FROZEN.json',dict(protocol_sha256=sha(output/'PROTOCOL.json'),cases=records,
        test_annotations_decoded=False,geometry_unchanged=True))
    print(__import__('json').dumps(dict(labelled_cases=len(records),test_targets_prepared=False)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('registry-root','geometry','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.registry_root,a.geometry,a.output)


if __name__=='__main__':main()
