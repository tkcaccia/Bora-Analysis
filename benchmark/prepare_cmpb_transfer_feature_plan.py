"""Freeze one target's predicted coarse mask for calibrated feature extraction."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import tifffile


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry', type=Path, required=True)
    p.add_argument('--tile-root', type=Path, required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    registry = {r['case']: r for r in json.loads(a.registry.read_text())['cases']}
    row = registry[a.target]
    assert row['split'] != 'train'
    frozen = json.loads((a.tile_root/'PREDICTIONS_FROZEN.json').read_text())
    assert a.target in {r['case'] for r in frozen['records']}
    tile = np.load(a.tile_root/a.target/'anonymous_tiles.npy', allow_pickle=False)
    assert digest(a.tile_root/a.target/'anonymous_tiles.npy') == next(r['tile_file_sha256'] for r in frozen['records'] if r['case'] == a.target)
    h, w = row['sample']['working_shape_yx']
    step = int(row['sample']['coarse_tile_px'])
    coarse = np.repeat(np.repeat(tile, step, axis=0), step, axis=1)[:h,:w]
    assert coarse.shape == (h,w) and set(np.unique(coarse)) >= {1,2}
    case_dir = a.output/'cases'/a.target
    case_dir.mkdir(parents=True, exist_ok=False)
    os.symlink(row['original_path'],case_dir/'original.png')
    tifffile.imwrite(case_dir/'coarse.tif',coarse,compression='deflate')
    sample = dict(row['sample'])
    sample['coarse_sha256'] = digest(case_dir/'coarse.tif')
    sample['coarse_reference_derived'] = False
    sample['source'] = {k:v for k,v in sample['source'].items() if k not in {'mask','mask_sha256','raw_labels'}}
    sample['target_reference_read'] = False
    sample['predicted_tile_source_case'] = frozen['source_case']
    plan = {'samples':[sample], 'target_reference_read':False,
            'coarse_origin':'source-trained fastPLS on cached, physically calibrated Virchow2 tile means'}
    (a.output/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'case':a.target,'coarse_sha256':sample['coarse_sha256'],
                      'groups':sorted(int(v) for v in np.unique(coarse) if v),
                      'physical_mpp_xy':sample['working_mpp_xy'],'reference_read':False}))


if __name__ == '__main__':
    main()
