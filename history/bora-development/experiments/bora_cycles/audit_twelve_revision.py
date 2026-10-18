"""Compare the two frozen cycle12 engineering pilots; never rescore or predict.

This is distinct from pilot/full repeatability: Q12 has an explicit repair and
Q15 an implementation-only port. Neither permits other native outputs to change.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import tifffile
from .audit_repeatability import frozen_run,REPORT_ONLY
from .paired_eight import SCORE_CORE
from .run import digest,save,verify


def compare_case(new,old,a,b,first,second,methods):
    if a['sample']!=b['sample']:raise ValueError('Case geometry/ontology changed')
    for field in ('input_manifest_sha256','rgb_sha256'):
        if first[field]!=second[field]:raise ValueError('Frozen input differs')
    counts=dict(unchanged_native_masks=0,unchanged_accuracy_cores=0,native_supports=0,proposal_arrays=0)
    changed=0
    for key in methods:
        aa,bb=a['methods'][key],b['methods'][key]
        if aa['status']!='ok' or bb['status']!='ok':raise ValueError('A method failed')
        x,y=[tifffile.imread(folder/key/'native.tif',maxworkers=2) for folder in (new,old)]
        if x.shape!=y.shape:raise ValueError('Native geometry differs')
        if key=='Q12':changed=int(np.count_nonzero(x!=y))
        else:
            if not np.array_equal(x,y):raise ValueError('Unchanged native mask differs: '+key)
            if any(aa['score'][field]!=bb['score'][field] for field in SCORE_CORE):
                raise ValueError('Unchanged accuracy core differs: '+key)
            counts['unchanged_native_masks']+=1;counts['unchanged_accuracy_cores']+=1
        support=key+'/support.native.tif'
        if (support in first['files_sha256'])!=(support in second['files_sha256']):raise ValueError('Support contract differs')
        if support in first['files_sha256']:
            if not np.array_equal(tifffile.imread(new/support,maxworkers=2),tifffile.imread(old/support,maxworkers=2)):
                raise ValueError('Native support differs: '+key)
            counts['native_supports']+=1
    for name in ('NOVELTY.npz','RARITY.npz'):
        if (name in first['files_sha256'])!=(name in second['files_sha256']):raise ValueError('Proposal contract differs')
        if name not in first['files_sha256']:continue
        with np.load(new/name) as x,np.load(old/name) as y:
            if x.files!=y.files or any(not np.array_equal(x[k],y[k]) for k in x.files):raise ValueError('Proposal arrays differ')
            counts['proposal_arrays']+=len(x.files)
    return counts,changed


def compare(new,old):
    new,old=Path(new),Path(old);p,f,rows=frozen_run(new);q,g,previous=frozen_run(old)
    if (not p['pilot'] or not q['pilot'] or p['cycle']!=12 or q['cycle']!=12
            or len(p['cases'])!=24 or len(p['methods'])!=23 or p['cases']!=q['cases']
            or p['methods']!=q['methods'] or 'Q12' not in p['methods']
            or any(p.get(k)!=q.get(k) for k in ('input_plan_sha256','primary_conversion','benchmark_scope','study_protocol','lineage'))):
        raise ValueError('Matching24-case/23-method cycle12 pilot contracts required')
    changed_sources=sorted(k for k in set(p['source_hashes'])|set(q['source_hashes']) if p['source_hashes'].get(k)!=q['source_hashes'].get(k))
    allowed=REPORT_ONLY|{'experiments/bora_cycles/twelve_regions.py','experiments/bora_cycles/preflight_twelve.sh','cpp/evidence/kernels.cpp','cpp/evidence/libbora_evidence.so'}
    if any(k not in allowed and not (k.startswith('experiments/bora_cycles/') and k.endswith('.md')) for k in changed_sources):
        raise ValueError('Undeclared prediction source changed')
    totals=dict(unchanged_native_masks=0,unchanged_accuracy_cores=0,native_supports=0,proposal_arrays=0);records=[]
    for case in p['cases']:
        a,b=new/'cases'/case,old/'cases'/case
        if digest(a/'PREDICTIONS.json')!=f['manifests'][case] or digest(b/'PREDICTIONS.json')!=g['manifests'][case]:raise ValueError('Prediction manifest changed')
        first,second=[json.loads((folder/'PREDICTIONS.json').read_text()) for folder in (a,b)]
        verify(a,first['files_sha256']);verify(b,second['files_sha256'])
        counts,changed=compare_case(a,b,rows[case],previous[case],first,second,p['methods'])
        for k,v in counts.items():totals[k]+=v
        records.append(dict(case=case,q12_changed_native_pixels=changed,
            q12_iou_delta=rows[case]['methods']['Q12']['score']['macro_iou']-previous[case]['methods']['Q12']['score']['macro_iou']))
    if totals['unchanged_native_masks']!=528 or totals['unchanged_accuracy_cores']!=528 or totals['native_supports']!=120:
        raise ValueError('Incomplete engineering revision comparison')
    timing={}
    for key in ('Q15','J01','C17','B23'):
        x=np.array([rows[c]['methods'][key]['median_seconds'] for c in p['cases']]);y=np.array([previous[c]['methods'][key]['median_seconds'] for c in q['cases']])
        timing[key]=dict(cases=24,old_mean_seconds=float(y.mean()),new_mean_seconds=float(x.mean()),old_median_seconds=float(np.median(y)),new_median_seconds=float(np.median(x)),old_over_one_second=int((y>1).sum()),new_over_one_second=int((x>1).sum()))
    return dict(cycle=12,pilot_versions=[1,2],cases=24,**totals,changed_sources=changed_sources,records=records,
        q12_cases_with_changed_pixels=sum(r['q12_changed_native_pixels']>0 for r in records),timing_observations=timing,
        timing_caveat='Concurrent image jobs, changing work scheduling and no isolated controlled timing experiment. These are observed cached-Bora times, not causal or end-to-end speedup estimates.',
        old_protocol_sha256=digest(old/'PROTOCOL.json'),new_protocol_sha256=digest(new/'PROTOCOL.json'),
        old_results_sha256=digest(old/'RESULTS.json'),new_results_sha256=digest(new/'RESULTS.json'),
        comparison_source_sha256=digest(Path(__file__)),predictions_modified=False,scores_modified=False,visual_reviews_created=False,survivors_selected=False,
        interpretation='All22 non-Q12 predictions/accuracy cores and all owned supports match over this24-case development pilot only. Q12 repair is explicit. This does not prove full-cohort repeatability, anatomical validity or generalization.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('new','old','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve prior revision audit')
    result=compare(args.new,args.old);save(args.output,result)
    print(json.dumps({k:result[k] for k in ('cases','unchanged_native_masks','unchanged_accuracy_cores','native_supports','q12_cases_with_changed_pixels','timing_observations')}))


if __name__=='__main__':main()
