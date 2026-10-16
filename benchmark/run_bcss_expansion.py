"""Resumable BCSS source-case extension at three tile widths and zero tile error."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark import run_bora_pilot as pilot


DEFAULT_OUTPUT = pilot.DATA_ROOT / 'results' / 'bora-bcss-expansion-2026-10-16'
PREVIOUS_GROUPS = {'TCGA-A7-A4SE', 'TCGA-BH-A42U',
                   'TCGA-GM-A2DI', 'TCGA-OL-A5RW'}


def select(count: int) -> list[dict]:
    rows = [json.loads(line) for line in
            (pilot.DATA_ROOT / 'prepared/bcss.jsonl').open()]
    eligible = [row for row in rows
                if row['group_id'] not in PREVIOUS_GROUPS
                and row['width'] * row['height'] <= 5_000_000
                and len(set(row['raw_labels']) - set(row['ignore_labels'])) >= 2]
    return eligible[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--groups', type=int, default=8)
    args = parser.parse_args()
    if args.groups < 1:
        parser.error('--groups must be positive')
    rows = select(args.groups)
    if len(rows) != args.groups:
        raise ValueError(f'Only {len(rows)} eligible unused BCSS source groups available')
    args.output.mkdir(parents=True, exist_ok=True)
    selection_file = args.output / 'selection.json'
    selection = [{'dataset': 'bcss', 'id': r['id'], 'group_id': r['group_id'],
                  'width': r['width'], 'height': r['height']} for r in rows]
    if selection_file.exists() and json.loads(selection_file.read_text()) != selection:
        raise ValueError('Existing selection differs; use a separate output directory')
    selection_file.write_text(json.dumps(selection, indent=2) + '\n')
    for row in rows:
        pilot.SAMPLES['bcss'] = row['id']
        pilot.run_sample('bcss', args.output / row['group_id'], 'adaptive-wand',
                         sizes=(56, 90, 216), errors=(0,))


if __name__ == '__main__':
    main()
