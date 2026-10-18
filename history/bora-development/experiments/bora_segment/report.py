"""Summarize frozen segment evidence and stage costs without new fitting."""
import argparse
from pathlib import Path
import numpy as np
from experiments.bora_supervised.registry import load,save,sha


def run(registry_root,geometry):
    registry_root,geometry=Path(registry_root),Path(geometry);registry=load(registry_root/'REGISTRY.json')
    complete=load(geometry/'CYCLE_COMPLETE.json');records={}
    for split,folder in [('validation','classifiers-validation-v1'),('followup_test','classifiers-followup-test-v1')]:
        result=load(geometry/folder/'RESULTS.json');seen=set();families=[]
        eligible_cases={c['case'] for c in result['ranking'][0]['cases']}
        timings=[load(geometry/'cases'/case/'GEOMETRY_FROZEN.json')['timing'] for case in sorted(eligible_cases)]
        for candidate in result['ranking']:
            if candidate['family'] in seen:continue
            seen.add(candidate['family']);model=load(geometry/folder/candidate['name']/'PREDICTIONS.json')
            rows={x['case']:x for x in model['predictions']}
            sums=[]
            for case in sorted(eligible_cases):
                base=load(geometry/'cases'/case/'GEOMETRY_FROZEN.json')['timing']['total_cached_seconds']
                p=rows[case];sums.append(base+p['head_seconds']+p.get('feature_loading_and_context_seconds',0))
            by_dataset={d:float(np.mean([c['pixel_score']['macro_iou'] for c in candidate['cases'] if c['dataset']==d]))
                for d in sorted({c['dataset'] for c in candidate['cases']})}
            families.append(dict(**{k:v for k,v in candidate.items() if k!='cases'},
                mean_stage_sum_cached_seconds=float(np.mean(sums)),stage_sum_over_one_second=int(np.count_nonzero(np.asarray(sums)>1)),
                runtime_penalized_pixel_iou=float(candidate['mean_native_pixel_iou']-.05*np.maximum(np.asarray(sums)-1,0).mean()),
                mean_region_oracle_pixel_accuracy=float(np.mean([c['oracle_majority_region_pixel_accuracy'] for c in candidate['cases']])),
                per_dataset_pixel_iou=by_dataset))
        records[split]=dict(cases=len(eligible_cases),models=families,selected_on_validation=result['selected_segment_classifier'],
            mean_geometry_stage_seconds={k:float(np.mean([t[k] for t in timings])) for k in timings[0]},
            oracle_note='Perfect majority-label assignment on these frozen regions; retrospective upper bound, not an available prediction.')
    all_frozen=load(geometry/'ALL_GEOMETRY_FROZEN.json')
    save(geometry/'REPORT.json',dict(task='One semantic tissue ranking per frozen connected Bora segment',
        registry_sha256=sha(registry_root/'REGISTRY.json'),geometry_sha256=sha(geometry/'ALL_GEOMETRY_FROZEN.json'),
        total_cases=len(all_frozen['cases']),total_regions=all_frozen['regions'],results=records,
        timing_scope='Stage sums from cached tokens, including geometry input/output work, then recorded evidence loading/context and warm head. Not a single end-to-end stopwatch; excludes encoder inference and startup refit.',
        prior_test_reuse_disclosed=True,provider_partition_correction='See SOURCE_SPLIT_AUDIT.json; PESO provider test release reused as development.',
        original_cycle_receipt=complete))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registry-root',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True)
    a=p.parse_args();run(a.registry_root,a.geometry)


if __name__=='__main__':main()
