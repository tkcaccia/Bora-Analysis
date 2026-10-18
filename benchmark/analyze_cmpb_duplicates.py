"""Screen calibrated fields for exact and perceptually similar cross-split images.

This is a triage aid, not proof of patient independence or absence of overlap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn


def phash(path: Path) -> int:
    with Image.open(path) as image:
        square = image.convert('L').resize((32, 32), Image.Resampling.BOX)
    coefficients = dctn(np.asarray(square, np.float32), norm='ortho')[:8, :8].reshape(-1)
    bits = coefficients[1:] > np.median(coefficients[1:])
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry', type=Path, required=True)
    p.add_argument('--evaluation-cases', type=Path, required=True,
                   help='JSON with a cases array of selected evaluation IDs')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    registry = json.loads(a.registry.read_text())['cases']
    chosen = {r['case'] for r in json.loads(a.evaluation_cases.read_text())['cases']}
    rows = []
    for r in registry:
        if r['split'] != 'train' and r['case'] not in chosen:
            continue
        path = Path(r['original_path'])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == r['original_sha256'], r['case']
        rows.append({'case': r['case'], 'split': r['split'], 'source_group': r['group'],
                     'sha256': digest, 'phash': phash(path)})
    train = [r for r in rows if r['split'] == 'train']
    evaluation = [r for r in rows if r['case'] in chosen]
    assert len(evaluation) == len(chosen)
    exact, candidates = [], []
    for x in evaluation:
        for y in train:
            if x['sha256'] == y['sha256']:
                exact.append((x['case'], y['case']))
            distance = (x['phash'] ^ y['phash']).bit_count()
            if distance <= 4:
                candidates.append({'evaluation': x['case'], 'training': y['case'],
                                   'hamming_distance': distance,
                                   'known_source_group_equal': x['source_group'] == y['source_group']})
    result = {'summary': {'training_images': len(train), 'evaluation_images': len(evaluation),
                          'pairs_screened': len(train) * len(evaluation),
                          'exact_image_duplicate_pairs': len(exact),
                          'perceptual_hash_candidates_at_distance_4': len(candidates),
                          'limitation': 'Global perceptual hashes miss overlapping crops, rotations, rescans and slide-level relatedness; source-group IDs are not all verified patients.'},
              'exact_pairs': exact, 'candidate_pairs': candidates}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
