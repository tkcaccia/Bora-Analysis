"""Prespecified small breast extension with distinct source groups at 56 px."""
from __future__ import annotations

import json
from pathlib import Path

from benchmark import run_bora_pilot as pilot

ROOT = pilot.DATA_ROOT / 'results' / 'bora-breast-extension-2026-10-16'


def eligible(row: dict) -> bool:
    return (row['width'] * row['height'] <= 3_000_000 and
            len([v for v in row['raw_labels'] if v not in row['ignore_labels']]) >= 2)


def main() -> None:
    selection=[]
    for dataset in ('bcss','tiger'):
        rows=[json.loads(line) for line in
              (pilot.DATA_ROOT/'prepared'/f'{dataset}.jsonl').open()]
        if dataset == 'bcss':
            rows=[r for r in rows if eligible(r) and r['id'] != pilot.SAMPLES['bcss']]
        else:
            rows=[r for r in rows if eligible(r) and r.get('source_subset') == 'tissue-cells'
                  and r['width'] >= 1000 and r['height'] >= 1000
                  and r['group_id'] != '114S']
            first={}
            for row in rows:
                first.setdefault(row['group_id'],row)
            rows=list(first.values())[:4]
        for row in rows:
            selection.append({'dataset':dataset,'id':row['id'],'group_id':row['group_id'],
                              'width':row['width'],'height':row['height']})
    ROOT.mkdir(exist_ok=True)
    (ROOT/'selection.json').write_text(json.dumps(selection,indent=2))
    for item in selection:
        pilot.SAMPLES[item['dataset']]=item['id']
        pilot.run_sample(item['dataset'], ROOT/item['group_id'], 'adaptive-wand',
                         sizes=(56,), errors=(0,))

if __name__=='__main__':main()
