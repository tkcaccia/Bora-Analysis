#!/usr/bin/env python3
"""Freeze a broad tissue ontology using Benchmark 1 training data alone."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SUPPORT = HERE / 'TRAINING_SUPPORT.json'
SAMPLE = Path('/mnt/sata_ssd/ChatGPT/Bora/migration/benchmark1-supervised-strategy-search-2026-11-10-v1/SAMPLE_COMPLETE.json')

# Parent merges are anatomical/semantic, not chosen from validation scores.
# IDs 16,18,20,22,23 have no defensible parent among the broad targets.
PARENTS = {
    'tumor': (1, 2),
    'stroma': (3, 4),
    'necrosis': (5,),
    'epithelium': (6, 7, 15, 17, 24),
    'blood': (8,),
    'vessel': (9, 10),
    'adipose': (11,),
    'immune_infiltrate': (12, 13, 14, 19, 21),
}
MIN_SOURCE_GROUPS = 20
MIN_PIXELS = 500_000


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    output = HERE / 'ONTOLOGY_FROZEN.json'
    if output.exists():
        raise FileExistsError(output)
    support, sample = json.loads(SUPPORT.read_text()), json.loads(SAMPLE.read_text())
    assert support['training_images'] == sample['training_cases'] == 397
    assert support['validation_masks_opened'] == sample['validation_cases_opened'] == 0
    labels = np.load(sample['index_path'])['label']
    cases = np.load(sample['index_path'])['case_index']
    source_groups = np.asarray(sample['source_groups'])
    assert len(source_groups) == len(sample['case_ids']) == 397
    old_to_new = {str(i): None for i in range(1, 25)}
    classes = {}
    for new_id, (name, children) in enumerate(PARENTS.items(), 1):
        for old_id in children:
            assert old_to_new[str(old_id)] is None
            old_to_new[str(old_id)] = new_id
        pixels = sum(support['classes'][str(i)]['pixels'] for i in children)
        sources = len(set(source_groups[np.unique(cases[np.isin(labels, children)])]))
        assert sources >= MIN_SOURCE_GROUPS and pixels >= MIN_PIXELS, (name, sources, pixels)
        classes[str(new_id)] = {'name': name, 'original_ids': list(children),
                                'training_full_mask_pixels': pixels,
                                'training_source_groups_at_least': sources}
    retained = sum(c['training_full_mask_pixels'] for c in classes.values())
    original = sum(c['pixels'] for c in support['classes'].values())
    excluded = [support['classes'][str(i)] | {'original_id': i}
                for i in range(1, 25) if old_to_new[str(i)] is None]
    result = {'status': 'TRAINING_ONLY_EIGHT_CLASS_ONTOLOGY_FROZEN',
              'selection_basis': 'semantic parent merges followed by >=20 independent training source groups and >=500000 full-resolution expert pixels per reported class; no validation data inspected',
              'training_images': 397, 'validation_masks_opened': 0,
              'minimum_source_groups': MIN_SOURCE_GROUPS, 'minimum_expert_pixels': MIN_PIXELS,
              'classes': classes, 'old_to_new': old_to_new,
              'excluded_original_classes': excluded,
              'retained_annotated_pixel_fraction': retained / original,
              'retained_annotated_pixels': retained, 'all_original_annotated_pixels': original,
              'evaluation_rule': 'Map source expert codes to original canonical roles, then to eight classes; pixels from five unrepresented leaf roles and unlabeled/background provider codes are unscored. Apply identical mapping to every method prediction. Report scored coverage with every metric. Never count excluded pixels as background.',
              'no_whole_slide_expert_accuracy': True,
              'pm016_excluded': True, 'beetle_excluded': True,
              'training_support_sha256': sha(SUPPORT),
              'sample_receipt_sha256': sha(SAMPLE),
              'source_sha256': sha(Path(__file__))}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'class_names': [v['name'] for v in classes.values()],
                      'excluded': [c['name'] for c in excluded],
                      'retained_fraction': retained / original}, indent=2))


if __name__ == '__main__':
    main()
