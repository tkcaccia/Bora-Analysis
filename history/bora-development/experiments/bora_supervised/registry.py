"""Build an appendable, leakage-audited index without decoding fine masks.

Dataset versions are immutable. New versions preserve existing case identities,
label IDs and group assignments; new groups use a stable hash partition.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ONTOLOGY=Path(__file__).with_name('ontology.json')
FEATURE_DIRECTORIES=('virchow2-multicase-2026-10-16',
    'bora-extension-features-2026-10-18-v1',
    'bora-tissue-features-160-2026-10-18-v1',
    'bora-puma-features-40-2026-10-18-v1')


def load(path):return json.loads(Path(path).read_text())


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def tissue_ids(names,ontology):
    ids={name:i+1 for i,name in enumerate(ontology['roles'])}
    result={}
    for code,name in names.items():
        if name in ontology['mapping']:result[str(code)]=ids[ontology['mapping'][name]]
        elif name in ontology['excluded']:result[str(code)]=0
        else:raise ValueError('Unreviewed tissue label: '+name)
    return result


def validate_ontology_extension(old,new):
    """New roles may only be appended; existing meanings must remain fixed."""
    n=len(old['roles'])
    if new['roles'][:n]!=old['roles'] or len(set(new['roles']))!=len(new['roles']):
        raise ValueError('Ontology extension changed or duplicated stable role IDs')
    for key in ('mapping','excluded'):
        if any(new[key].get(name)!=meaning for name,meaning in old[key].items()):
            raise ValueError('Ontology extension changed an existing label meaning')
    if set(new['mapping'])&set(new['excluded']):raise ValueError('A source label cannot be both mapped and excluded')
    if any(role not in new['roles'] for role in new['mapping'].values()):
        raise ValueError('Mapping names an unknown role')


def stable_split(group,seed=314159):
    fraction=int(hashlib.sha256(f'bora-supervised-v1:{seed}:{group}'.encode()).hexdigest()[:16],16)/16**16
    return 'train' if fraction<.70 else 'validation' if fraction<.85 else 'test'


def official_split(sample):
    source=sample.get('source',{})
    values=[source.get('split'),source.get('provider_split')]
    for value in values:
        value=str(value).lower()
        if value in ('test','testing','test_set'):return 'test'
        if value in ('val','valid','validation','validation_set'):return 'validation'
    return None


def source_group(sample):
    text=json.dumps({k:sample.get(k) for k in ('group','source')})
    match=re.search(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}',text.upper())
    return 'TCGA:'+match.group() if match else sample['dataset']+':'+str(sample['group'])


def assign_groups(samples,seed=314159):
    """Join known donors and exact source-image duplicates before partitioning."""
    parent={r['case']:r['case'] for r in samples}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    seen={}
    for row in samples:
        identity=row.get('source',{}).get('image_sha256')
        for key in [source_group(row)]+(['image:'+identity] if identity else []):
            if key in seen:union(row['case'],seen[key])
            else:seen[key]=row['case']
    clusters={}
    for row in samples:clusters.setdefault(find(row['case']),[]).append(row)
    result={}
    for rows in clusters.values():
        groups=sorted({source_group(r) for r in rows})
        group=groups[0]
        forced={official_split(r) for r in rows}-{None}
        split='test' if 'test' in forced else 'validation' if 'validation' in forced else stable_split(group,seed)
        for row in rows:result[row['case']]=dict(group=group,split=split,linked_source_groups=groups,
            forced_provider_splits=sorted(forced))
    return result


def find_cache(results,sample,feature_entry=None):
    candidates=[]
    directories=([Path(feature_entry['path']).parent] if feature_entry else [])
    directories.extend(Path(results)/name/'cases'/sample['case']/'features' for name in FEATURE_DIRECTORIES)
    seen=set()
    for directory in directories:
        if directory.resolve() in seen:continue
        seen.add(directory.resolve())
        if not (directory/'base.npy').exists() or not (directory/'READY.json').exists():continue
        meta=load(directory/'READY.json')
        if meta.get('image_sha256')==sample['original_sha256']:candidates.append((directory,meta))
    if len(candidates)!=1:raise ValueError('Expected one matching full-feature cache: '+sample['case']+'; found '+str(len(candidates)))
    directory,meta=candidates[0]
    if feature_entry and meta['sha256']!=feature_entry['sha256']:
        raise ValueError('Base-feature hash differs from frozen input provenance')
    array=np.load(directory/'base.npy',mmap_mode='r')
    h,w=sample['working_shape_yx'];patch=meta.get('patch_px')
    if (array.ndim!=3 or array.shape[-1]!=1280 or patch!=14
            or array.shape[0]*patch<h or array.shape[1]*patch<w
            or meta.get('channels',1280)!=1280):raise ValueError('Full-token geometry/channel mismatch')
    return dict(path=str(directory/'base.npy'),sha256=meta['sha256'],shape=list(array.shape),
        dtype=str(array.dtype),metadata_path=str(directory/'READY.json'),metadata_sha256=sha(directory/'READY.json'),
        channels=1280,patch_px=patch,window_px=meta['window_px'],
        physical_patch_um_xy=[patch*v for v in sample['working_mpp_xy']],
        physical_window_um_xy=[meta['window_px']*v for v in sample['working_mpp_xy']],
        selected_channels_used=False)


def build(source,results,root,previous=None,ontology_path=ONTOLOGY):
    source,root=Path(source),Path(root)
    if (root/'REGISTRY.json').exists():raise ValueError('Create a new immutable dataset version')
    plan=load(source/'PLAN.json');ontology=load(ontology_path);samples=plan['samples']
    validate_ontology_extension(ontology,ontology)
    if len({s['case'] for s in samples})!=len(samples):raise ValueError('Duplicate case IDs')
    assigned=assign_groups(samples)
    rows=[]
    for sample in samples:
        if sample.get('working_mpp_xy') is None or min(sample['working_mpp_xy'])<=0:
            raise ValueError('Physical calibration required before inclusion')
        case=sample['case'];folder=source/'cases'/case
        frozen=load(folder/'INPUTS_FROZEN.json')
        files=frozen['files_sha256']
        if files['original.png']!=sample['original_sha256']:raise ValueError('Image identity mismatch')
        rows.append(dict(case=case,dataset=sample['dataset'],**assigned[case],sample=sample,
            original_path=str(folder/'original.png'),original_sha256=files['original.png'],
            reference_path=str(folder/'reference.native.tif'),reference_sha256=sample['reference_native_sha256'],
            tissue_support_path=str(folder/'tissue.working.tif'),tissue_support_sha256=files['tissue.working.tif'],
            ontology_raw_to_role=tissue_ids(sample['class_names'],ontology),
            feature=find_cache(results,sample,frozen.get('feature_cache',{}).get('base'))))
    if previous:
        old=load(Path(previous)/'REGISTRY.json');new={r['case']:r for r in rows}
        validate_ontology_extension(old['ontology'],ontology)
        for row in old['cases']:
            current=new.get(row['case'])
            stable=('case','dataset','group','split','linked_source_groups','forced_provider_splits',
                    'sample','original_sha256','reference_sha256','tissue_support_sha256',
                    'ontology_raw_to_role','feature')
            if current is None or any(current[k]!=row[k] for k in stable):
                raise ValueError('Extension changed an existing case or split')
            # Keep original immutable shard/input locations when an extended
            # source plan lives in a new directory.
            current.update(row)
    record=dict(version='bora-supervised-tissues-v1',source=str(source),source_plan_sha256=sha(source/'PLAN.json'),
        ontology=ontology,ontology_sha256=sha(ontology_path),cases=rows,
        split_counts=dict(Counter(r['split'] for r in rows)),dataset_counts=dict(Counter(r['dataset'] for r in rows)),
        partition='Stable source-group hash70/15/15; exact provider split tokens reserved; development-tagged source releases repartitioned; exact image duplicates linked before assignment',
        group_caveat='TCGA case identifiers joined across datasets. Other provider groups may be slides, cores or ROIs, not verified patients. Unknown cross-source links remain a limitation.',
        reuse_caveat='All469 images were previously inspected during refinement development; this split is new for the supervised head, not an independent validation of the chosen Bora architecture.',
        fine_masks_decoded=False,coarse_masks_read=False,selected_feature_maps_read=False,
        upstream_encoder_revision='3158645804b69e3f3bc4439d4116edddf0840a72',
        feature_contract='Unselected1280-dimensional spatial token grids only; no per-image reference-driven channel selection or converted RGB enters this supervised experiment.',
        previous_registry_sha256=sha(Path(previous)/'REGISTRY.json') if previous else None,
        registry_code_sha256=sha(Path(__file__)),
        created_at_utc=datetime.now(timezone.utc).isoformat())
    save(root/'REGISTRY.json',record)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','results','root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--previous',type=Path)
    p.add_argument('--ontology',type=Path,default=ONTOLOGY)
    a=p.parse_args();r=build(a.source,a.results,a.root,a.previous,a.ontology)
    print(json.dumps(dict(cases=len(r['cases']),splits=r['split_counts'],fine_masks_decoded=False)))


if __name__=='__main__':main()
