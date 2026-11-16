#!/usr/bin/env python3
"""Cross-check CAM source-only tile counts with released Bora preparation code."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
PACKAGE = DATA/'worktrees/bora-r-package-2026-11-14-v1'
SOURCE = PACKAGE/'src/bora/r_feature_preflight.py'
RUN = ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-14-v14/EXACT_ADMISSION.json'
OUT = Path(__file__).with_name('PACKAGE_TILE_COUNT_AUDIT_VERIFIED.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    sys.path.insert(0,str(PACKAGE/'src'))
    from bora.r_feature_preflight import tissue_tile_counts
    run = json.loads(RUN.read_text())
    records = []
    for row in run['rows']:
        geometry = DATA/f'external-camelyon16-pair-2026-11-13-v54/he-support/{row["slide"]}/tissue-area.geojson'
        assert sha(geometry) == row['accepted_geojson_sha256']
        expected = np.load(row['counts_path'],allow_pickle=False)
        actual = tissue_tile_counts(tuple(row['calibrated_shape_yx']),geometry)
        assert np.array_equal(actual,expected)
        selected = actual.astype(np.int64)*10 > 9*224*224
        assert int(selected.sum()) == row['eligible_tiles']
        records.append({'slide':row['slide'],'tissue_tile_count':int(selected.sum()),
                        'different_tile_counts':0,'accepted_geojson_sha256':sha(geometry),
                        'counts_sha256':sha(row['counts_path'])})
    assert sum(x['tissue_tile_count'] for x in records) == run['joint_eligible_tiles']
    result={'status':'RELEASED_BORA_PREPARATION_MATCHES_EXACT_CAM_TILE_COUNTS',
            'released_package_source_sha256':sha(SOURCE),'exact_run_sha256':sha(RUN),
            'rows':records,'joint_tissue_tiles':run['joint_eligible_tiles'],
            'cam_source_rgb_features_expert_references_opened':False,
            'cam_fit_or_scoring':False,'online_publication':False}
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'joint_tissue_tiles':result['joint_tissue_tiles'],
                      'different_tile_counts':0}))


if __name__=='__main__':
    main()
