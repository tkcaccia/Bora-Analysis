"""Version a requested cohort/cleanup change without mutating a live protocol."""
import argparse,shutil,os
from pathlib import Path
from collections import Counter
from experiments.bora_supervised.registry import load,save,sha
from .data import evaluation_cases


def run(previous,study):
    previous,study=Path(previous),Path(study)
    if not (previous/'STOPPED_FOR_COHORT_REVISION.json').exists():raise ValueError('Stop and audit the old study before revising')
    if list((study/'cycles').glob('*/CYCLE_COMPLETE.json')):raise ValueError('Revision already evaluated')
    study.mkdir(parents=True,exist_ok=True)
    # Copy code, never hard-link mutable code into the archived partial run.
    shutil.copytree(previous/'code',study/'code',dirs_exist_ok=True)
    for name in ('registry-v2','geometry','descriptors','targets'):
        dst=study/name
        if not dst.exists():dst.symlink_to(previous/name,target_is_directory=True)
    for name in ('PREPARATION_COMPLETE.json','PREPROCESSING_COMPLETE.json','CLASSIFIER_INPUTS_VERIFIED.json','VECTOR_PROFILE_APPROVED.json'):
        shutil.copy2(previous/name,study/name)
    policy=dict(excluded_datasets=['puma'],scope='All eligible previously reserved validation and test images from other datasets',
        training_splits_unchanged=True,excluded_evaluation_images_never_promoted_to_training=True,
        development_evaluation=True,native_polygon_and_raster_minimum_area=20,strictly_less_than_threshold=True)
    save(study/'EVALUATION_POLICY.json',policy)
    reg=load(study/'registry-v2/REGISTRY.json');rows={r['case']:r for r in reg['cases']};cases=evaluation_cases(rows,policy)
    transferred=[]
    for old in sorted((previous/'cycles/cycle-01').glob('*/PREDICTIONS.json')):
        manifest=load(old);records=[r for r in manifest['predictions'] if r['case'] in cases]
        if {r['case'] for r in records}!=set(cases):raise ValueError('Incomplete source predictor')
        dst=study/'cycles/cycle-01'/old.parent.name;dst.mkdir(parents=True,exist_ok=True)
        for record in records:
            src=old.parent/(record['case']+'.npz');out=dst/src.name
            if sha(src)!=record['sha256']:raise ValueError('Source prediction changed')
            if not out.exists():os.link(src,out)
        manifest['predictions']=records
        # Measured fit/warm prediction costs are unchanged. Fine-mask results,
        # selection, and old vectors are deliberately not transferred.
        save(dst/'PREDICTIONS.json',manifest);transferred.append(manifest['id'])
    summary=dict(previous_study=str(previous),previous_registry_sha256=sha(previous/'registry-v2/REGISTRY.json'),
        registry_unchanged=sha(study/'registry-v2/REGISTRY.json')==sha(previous/'registry-v2/REGISTRY.json'),
        cases=cases,images=len(cases),dataset_counts=dict(Counter(rows[c]['dataset'] for c in cases)),
        split_counts=dict(Counter(rows[c]['split'] for c in cases)),puma_evaluation_images=0,
        excluded_puma=[c for c,r in rows.items() if r['split']!='train' and r['dataset']=='puma'],
        transferred_frozen_predictors=transferred,old_scores_or_vectors_reused=False,
        prediction_reuse_reason='Exact same training rows, features, labels, fitted recipes and image inputs; evaluation cohort and output cleanup changed only')
    save(study/'COHORT_REVISION.json',summary)
    save(study/'STATUS_NOTE.json',dict(current_phase='Revised cohort ready for fresh geometry export and scoring',
        cycles_completed=0,images=len(cases),dataset_counts=summary['dataset_counts'],split_counts=summary['split_counts'],
        excluded_dataset='puma',previous_partial_study_preserved=str(previous),continuous_polygon_area_checked=True))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--study',type=Path,required=True)
    a=p.parse_args();print(run(a.previous,a.study))
