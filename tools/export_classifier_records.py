"""Export completed-cycle candidate recipes and metrics, omitting private paths.

Reads saved JSON only. Does not resume a study or execute any candidate.
"""
import argparse
import json
from pathlib import Path


def export(study, destination):
    destination.mkdir(parents=True, exist_ok=True)
    exported = []
    for cycle in sorted((study / 'cycles').glob('cycle-*')):
        if not (cycle / 'CYCLE_COMPLETE.json').exists():
            continue
        result = json.loads((cycle / 'RESULTS.json').read_text())
        records = []
        for row in result['ranking']:
            row = {k: v for k, v in row.items() if k != 'cases'}
            records.append(row)
        payload = dict(generation=result['generation'], ranking=records,
                       selected=result['selected'],
                       development_test_reuse=result['development_test_reuse'])
        target = destination / cycle.name
        target.mkdir(exist_ok=True)
        (target / 'SUMMARY.json').write_text(json.dumps(payload, indent=2) + '\n')
        candidates = json.loads((cycle / 'CANDIDATES.json').read_text())
        (target / 'CANDIDATES.json').write_text(json.dumps(candidates, indent=2) + '\n')
        exported.append(cycle.name)
    return exported


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(export(args.study, args.output)))
