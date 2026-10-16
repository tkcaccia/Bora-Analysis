"""Resumable, source-group-aware TIGER extension at three tile widths.

This is a descriptive extension of the frozen adaptive-wand configuration.
Selection uses the acquired inventory and excludes groups already evaluated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark import run_bora_pilot as pilot


DEFAULT_OUTPUT = pilot.DATA_ROOT / 'results' / 'bora-tiger-expansion-2026-10-16'


def select(count: int) -> list[dict]:
    inventory = pilot.DATA_ROOT / 'prepared/tiger.jsonl'
    rows = [json.loads(line) for line in inventory.open()]
    previous = pilot.DATA_ROOT / 'results/bora-breast-extension-2026-10-16/selection.json'
    excluded = {'114S'}
    if previous.exists():
        excluded.update(row['group_id'] for row in json.loads(previous.read_text())
                        if row['dataset'] == 'tiger')
    selected: dict[str, dict] = {}
    for row in rows:
        if (row.get('source_subset') != 'tissue-cells'
                or row['group_id'] in excluded
                or row['width'] * row['height'] > 3_000_000
                or min(row['width'], row['height']) < 1000):
            continue
        valid = set(row['raw_labels']) - set(row['ignore_labels'])
        if len(valid) < 2:
            continue
        selected.setdefault(row['group_id'], row)
    return list(selected.values())[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--groups', type=int, default=10)
    args = parser.parse_args()
    if args.groups < 1:
        parser.error('--groups must be positive')
    rows = select(args.groups)
    if len(rows) != args.groups:
        raise ValueError(f'Only {len(rows)} distinct eligible source groups available')
    args.output.mkdir(parents=True, exist_ok=True)
    selection_file = args.output / 'selection.json'
    selection = [{'dataset': 'tiger', 'id': r['id'], 'group_id': r['group_id'],
                  'width': r['width'], 'height': r['height'],
                  'source_subset': r['source_subset']} for r in rows]
    if selection_file.exists() and json.loads(selection_file.read_text()) != selection:
        raise ValueError('Existing selection differs; use a separate output directory')
    selection_file.write_text(json.dumps(selection, indent=2) + '\n')
    for row in rows:
        pilot.SAMPLES['tiger'] = row['id']
        pilot.run_sample('tiger', args.output / row['group_id'], 'adaptive-wand',
                         sizes=(56, 90, 216), errors=(0,))


if __name__ == '__main__':
    main()
