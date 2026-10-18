"""Append calibrated acquisitions without changing any historical input or fold."""
import argparse
from pathlib import Path
import os, shutil
from collections import Counter
from threadpoolctl import threadpool_limits
from experiments.bora_supervised.registry import load,save,sha,build
from experiments.bora_supervised.prepare import prepare_case
from experiments.bora_extension.conversions import transfer_native_files,prepare_case as convert_case
from experiments.bora_group import geometry,targets


def link_tree(source,destination):
    """Hard links preserve immutable artifacts without duplicating native arrays."""
    source,destination=Path(source),Path(destination)
    for path in source.rglob('*'):
        relative=path.relative_to(source);out=destination/relative
        if path.is_dir():out.mkdir(parents=True,exist_ok=True)
        elif not out.exists():
            out.parent.mkdir(parents=True,exist_ok=True)
            try:os.link(path,out)
            except OSError:shutil.copy2(path,out)


def run(study,old_registry,old_geometry,data):
    study,old_registry,old_geometry,data=map(Path,(study,old_registry,old_geometry,data))
    sources=[study/'acquisition-v2'/n for n in ('gcss','ignite')]
    old=load(old_registry/'REGISTRY.json');inputs=study/'inputs';samples=[r['sample'] for r in old['cases']]
    for row in old['cases']:
        dst=inputs/'cases'/row['case'];dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():dst.symlink_to(Path(row['original_path']).parent,target_is_directory=True)
    for source in sources:
        for sample in load(source/'PLAN.json')['samples']:
            transfer_native_files(source,inputs,sample)
            convert_case((inputs,study/'features-new',sample));samples.append(sample)
    save(inputs/'PLAN.json',dict(samples=samples,previous_registry_sha256=sha(old_registry/'REGISTRY.json'),
         added_source_plan_sha256={str(s):sha(s/'PLAN.json') for s in sources}))
    ontology=old['ontology'];known=set(ontology['mapping'])|set(ontology['excluded'])
    unknown={name for s in samples for name in s['class_names'].values()}-known
    # Provider-defined keratinization is a distinct role; never merge silently.
    for name in sorted(unknown):
        if name.lower()!='keratinization':raise ValueError('Unreviewed additional ontology label: '+name)
        ontology['roles'].append('keratinized_tissue');ontology['mapping'][name]='keratinized_tissue'
    save(study/'ontology.json',ontology)
    registry=study/'registry-v2'
    if not (registry/'REGISTRY.json').exists():build(inputs,data/'results',registry,old_registry,study/'ontology.json')
    current=load(registry/'REGISTRY.json');geo=study/'geometry'
    if not (geo/'ALL_GEOMETRY_FROZEN.json').exists():
        link_tree(old_geometry/'cases',geo/'cases');link_tree(old_geometry/'evaluation_keys',geo/'evaluation_keys')
        geometry.run(registry,geo)
    targets.run(registry,geo,study/'targets')
    # Tile features are descriptor inputs only. Copy immutable old predictors,
    # then rebuild non-test targets under the extended ontology when necessary.
    for row in current['cases']:
        src=old_registry/'cases'/row['case'];dst=registry/'cases'/row['case']
        if src.exists() and not (dst/'PREPARED.json').exists():
            dst.mkdir(parents=True,exist_ok=True)
            for name in ('mean.npy','centre.npy','units.npz'):
                if not (dst/name).exists():os.link(src/name,dst/name)
            record=load(src/'PREPARED.json')
            if row['split']!='test':
                import numpy as np
                with np.load(src/'targets.npz') as t:
                    fractions=np.pad(t['fractions'],((0,0),(0,len(ontology['roles'])-t['fractions'].shape[1])))
                    np.savez(dst/'targets.npz',fractions=fractions,coverage=t['coverage'])
            record.update(registry_sha256=sha(registry/'REGISTRY.json'),
                files_sha256={n:sha(dst/n) for n in record['files_sha256']},reused_immutable_predictors=True)
            save(dst/'PREPARED.json',record)
        else:prepare_case(registry,row,ontology)
    save(study/'PREPARATION_COMPLETE.json',dict(cases=len(current['cases']),dataset_counts=current['dataset_counts'],
        split_counts=current['split_counts'],added_cases=len(samples)-len(old['cases']),
        fixed_old_geometry_and_folds=True,registry_sha256=sha(registry/'REGISTRY.json')))


def main():
    p=argparse.ArgumentParser()
    for name in ('study','old-registry','old-geometry','data'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=2):run(a.study,a.old_registry,a.old_geometry,a.data)

if __name__=='__main__':main()
