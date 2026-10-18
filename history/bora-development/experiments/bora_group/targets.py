"""Reveal original group identities only after anonymized geometry freezes."""
import argparse
from pathlib import Path
import numpy as np
import tifffile
from experiments.bora_supervised.registry import load,save,sha
from experiments.bora_segment.targets import target_counts
from .geometry import UNIT


def identity_labels(row,geometry):
    geometry=Path(geometry);folder=geometry/'cases'/row['case'];record=load(folder/'GEOMETRY_FROZEN.json')
    key=geometry/'evaluation_keys'/(row['case']+'.json')
    if sha(key)!=record['evaluation_key_sha256']:raise ValueError('Anonymization answer key changed')
    source=load(key)['source_class_by_output_group']
    return np.asarray([row['ontology_raw_to_role'].get(str(c),0) for c in source],np.uint8)


def run(registry_root,geometry,output):
    registry_root,geometry,output=map(Path,(registry_root,geometry,output))
    registry=load(registry_root/'REGISTRY.json');frozen=load(geometry/'ALL_GEOMETRY_FROZEN.json')
    if set(frozen['cases'])!={r['case'] for r in registry['cases']}:raise ValueError('All geometry must freeze before labels attach')
    protocol=dict(prediction_unit=UNIT,registry_sha256=sha(registry_root/'REGISTRY.json'),
        geometry_sha256=sha(geometry/'ALL_GEOMETRY_FROZEN.json'),test_annotations_decoded=False,
        identity_target='Expert tissue identity that generated the original coarse group; revealed only to training/evaluation.',
        fractions='Mapped expert pixel composition within the refined group, used by the composition candidate and quality diagnostics.',
        group_eligibility='Mapped source identity and at least one mapped expert pixel. Unannotated pixels never enter pixel quality.',
        code_sha256=sha(Path(__file__)))
    path=output/'PROTOCOL.json'
    if path.exists() and load(path)!=protocol:raise ValueError('Use a new target version')
    save(path,protocol);records={}
    for row in registry['cases']:
        case=row['case'];dst=output/'cases'/case
        if row['split']=='test':
            if (dst/'targets.npz').exists():raise ValueError('Test targets must remain unprepared')
            continue
        folder=geometry/'cases'/case
        if sha(folder/'GEOMETRY_FROZEN.json')!=frozen['cases'][case]:raise ValueError('Geometry changed')
        record=load(folder/'GEOMETRY_FROZEN.json')
        if sha(folder/'regions.native.tif')!=record['files_sha256']['regions.native.tif']:raise ValueError('Group boundaries changed')
        if sha(row['reference_path'])!=row['reference_sha256']:raise ValueError('Expert reference changed')
        regions=tifffile.imread(folder/'regions.native.tif');raw=tifffile.imread(row['reference_path']);truth=np.zeros(raw.shape,np.uint8)
        for code,role in row['ontology_raw_to_role'].items():truth[raw==int(code)]=role
        fractions,coverage=target_counts(regions,truth,len(registry['ontology']['roles']));identity=identity_labels(row,geometry)
        if len(identity)!=len(fractions):raise ValueError('One hidden identity per whole refined group required')
        dst.mkdir(parents=True,exist_ok=True);np.savez(dst/'targets.npz',fractions=fractions,coverage=coverage,identity=identity)
        records[case]=dict(split=row['split'],regions=len(identity),sha256=sha(dst/'targets.npz'))
    save(output/'TARGETS_FROZEN.json',dict(protocol_sha256=sha(path),prediction_unit=UNIT,cases=records,
        test_annotations_decoded=False,test_identity_keys_decoded=False,geometry_unchanged=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('registry-root','geometry','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.registry_root,a.geometry,a.output)


if __name__=='__main__':main()
