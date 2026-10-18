"""Multiclass benchmarking requires at least two distinct mapped tissue roles.

Single-role examples remain valid training examples. Existing exploratory
all-validation reports are retained; this creates explicitly versioned scores.
"""
import argparse
from pathlib import Path
import numpy as np
from .registry import load,save,sha
from .cycle_two import score_predictions


def eligible(row):
    return len(set(row['ontology_raw_to_role'].values())-{0})>=2


def rescore(root,cycle):
    root,cycle=Path(root),Path(cycle);registry=load(root/'REGISTRY.json')
    subset=dict(registry,cases=[r for r in registry['cases'] if eligible(r)])
    original=load(cycle/'RESULTS.json');scores=[]
    for result in original['ranking']:
        frozen=load(cycle/result['name']/'PREDICTIONS.json')
        s={k:v for k,v in result.items() if k!='cases'}
        s.update(score_predictions(root,subset,cycle/result['name'],frozen['classes'],frozen['predictions']))
        scores.append(s)
    scores.sort(key=lambda r:(-r['balanced_accuracy'],-r['top1']))
    out=dict(ranking=scores,criterion='At least two distinct mapped expert tissue roles per benchmark ROI. Existing physical calibration/size/tile-count eligibility remains in force.',
        validation_cases=[r['case'] for r in subset['cases'] if r['split']=='validation'],
        excluded_validation=[dict(case=r['case'],reason='Fewer than two distinct mapped expert tissue roles') for r in registry['cases'] if r['split']=='validation' and not eligible(r)],
        original_all_validation_results_sha256=sha(cycle/'RESULTS.json'),test_annotations_read=False,
        selected_validation_configuration=scores[0]['name'],source_sha256=sha(Path(__file__)))
    save(cycle/'ELIGIBLE_RESULTS.json',out)
    print(__import__('json').dumps(dict(cycle=str(cycle),eligible_validation=len(out['validation_cases']),best={k:v for k,v in scores[0].items() if k not in ('cases','recall_by_role')})),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--cycle',type=Path,required=True)
    a=p.parse_args();rescore(a.root,a.cycle)


if __name__=='__main__':main()
