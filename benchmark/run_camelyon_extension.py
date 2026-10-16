"""Run the three inventoried native CAMELYON16 tumour-window pairs.

These 4096-pixel windows are sampled from three complete acquired slides. The
run does not measure full-slide throughput or independent normal annotations.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmark import run_bora_pilot as pilot


ROOT = pilot.DATA_ROOT / 'results/bora-camelyon-windows-2026-10-16'


def main() -> None:
    rows = [json.loads(line) for line in
            (pilot.DATA_ROOT / 'prepared/camelyon16-subset.jsonl').open()]
    if len(rows) != 3 or len({row['group_id'] for row in rows}) != 3:
        raise ValueError('Expected three different source slides in frozen inventory')
    ROOT.mkdir(parents=True, exist_ok=True)
    selection = [{'id': row['id'], 'group_id': row['group_id'],
                  'image': row['image'], 'mask': row['mask']} for row in rows]
    path = ROOT / 'selection.json'
    if path.exists() and json.loads(path.read_text()) != selection:
        raise ValueError('Existing CAMELYON selection differs')
    path.write_text(json.dumps(selection, indent=2) + '\n')
    for row in rows:
        pilot.SAMPLES['camelyon16-subset'] = row['id']
        pilot.run_sample('camelyon16-subset', ROOT / row['group_id'],
                         'adaptive-wand', sizes=(56, 90, 216), errors=(0,))


if __name__ == '__main__':
    main()
