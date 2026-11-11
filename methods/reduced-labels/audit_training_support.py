#!/usr/bin/env python3
"""Count full-resolution expert pixels and independent sources in frozen training only."""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
HERE = Path(__file__).resolve().parent
INPUTS = ROOT / 'migration/benchmark1-supervised-virchow2-response-2026-11-10-v1/INPUTS_FROZEN.json'
SAMPLE = ROOT / 'migration/benchmark1-supervised-strategy-search-2026-11-10-v1/SAMPLE_COMPLETE.json'
HARM = ROOT / 'benchmark/methods/benchmark1/tissue_label_harmonization.json'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = json.loads(INPUTS.read_text())
    sample = json.loads(SAMPLE.read_text())
    harmonization = json.loads(HARM.read_text())['mapping']
    class_names = manifest['classes']
    role_to_id = {name: int(idx) for idx, name in class_names.items()}
    source_by_case = dict(zip(sample['case_ids'], sample['source_groups']))
    counts = {int(i): {'name': name, 'pixels': 0, 'images': set(), 'sources': set(),
                       'datasets': collections.Counter()} for i, name in class_names.items()}
    ignored = collections.Counter()
    unknown = collections.Counter()
    train = [row for row in manifest['rows'] if row['split'] == 'train']
    assert len(train) == 397 and len(source_by_case) == 397
    for row in train:
        case, dataset = row['case'], row['dataset']
        assert case in source_by_case
        path = Path(row['expert_mask_path'])
        assert digest(path) == row['expert_mask_sha256'], case
        with Image.open(path) as im:
            mask = np.asarray(im)
        assert list(mask.shape) == row['expert_mask_shape_yx'], case
        values, pixel_counts = np.unique(mask, return_counts=True)
        for value, n in zip(values.tolist(), pixel_counts.tolist()):
            if value in (0, 255):
                ignored[(dataset, value, None)] += n
                continue
            entry = harmonization[dataset].get(str(value))
            if entry is None:
                unknown[(dataset, value)] += n
                continue
            role = entry.get('canonical_role')
            if entry.get('use') != 'score' or role not in role_to_id:
                ignored[(dataset, value, role)] += n
                continue
            rec = counts[role_to_id[role]]
            rec['pixels'] += n
            rec['images'].add(case)
            rec['sources'].add(source_by_case[case])
            rec['datasets'][dataset] += n
    assert not unknown, unknown
    result = {'status': 'TRAINING_SUPPORT_AUDITED', 'training_images': 397,
              'validation_masks_opened': 0, 'manifest_sha256': digest(INPUTS),
              'sample_receipt_sha256': digest(SAMPLE), 'harmonization_sha256': digest(HARM),
              'classes': {str(i): {'name': rec['name'], 'pixels': rec['pixels'],
                                   'images': len(rec['images']), 'source_groups': len(rec['sources']),
                                   'datasets': dict(rec['datasets'])}
                          for i, rec in counts.items()},
              'ignored_source_codes': {str(k): v for k, v in ignored.items()}}
    path = HERE / 'TRAINING_SUPPORT.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(path), 'classes': result['classes']}, indent=2))


if __name__ == '__main__':
    main()
