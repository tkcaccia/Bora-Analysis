#!/usr/bin/env python3
"""Independent window audit of source-only 0.5-um CAM tile admission."""

import hashlib
import json
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.features import rasterize


ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
RUN = ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-14-v14/EXACT_ADMISSION.json'
PROXY = ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-14-v13/KODAMA_CAPACITY_SCREEN.json'
OUT = Path(__file__).with_name('EXACT_ADMISSION_AUDIT_VERIFIED.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def audit():
    if OUT.exists():
        raise FileExistsError(OUT)
    run = json.loads(RUN.read_text())
    proxy = json.loads(PROXY.read_text())
    assert run['status'] == 'SOURCE_ONLY_EXACT_0P5UM_TISSUE_TILE_CAPACITY'
    assert not run['cam_specific_training_or_scoring_performed']
    assert not run['cam_source_rgb_features_expert_references_opened']
    rng = np.random.default_rng(91519)
    records = []
    for row in run['rows']:
        slide = row['slide']
        path = DATA/f'external-camelyon16-pair-2026-11-13-v54/he-support/{slide}/tissue-area.geojson'
        assert sha(path) == row['accepted_geojson_sha256']
        geometry = json.loads(path.read_text())
        assert geometry['image_shape_yx'] == row['calibrated_shape_yx']
        assert geometry['unpadded_image_shape_yx'] == row['unpadded_shape_yx']
        polygons = [item['geometry'] for item in geometry['features']]
        counts_path = Path(row['counts_path'])
        assert sha(counts_path) == row['counts_sha256']
        counts = np.load(counts_path, allow_pickle=False)
        assert counts.dtype == np.uint16 and list(counts.shape) == row['tile_grid_shape_yx']
        assert int(counts.max()) <= 50176
        eligible = counts.astype(np.int32)*10 > 9*50176
        assert int(eligible.sum()) == row['eligible_tiles']
        assert int((counts == 50176).sum()) == row['fully_supported_tiles']
        assert int((counts > 0).sum()) == row['any_tissue_tiles']
        # Independently rasterize individual tile windows using ALL polygons,
        # including tiles at boundaries and the exact threshold neighborhood.
        boundary = np.flatnonzero((counts > 0) & (counts < 50176))
        close = np.argsort(np.abs(counts.astype(np.int32).ravel()-45159))[:20]
        random = rng.choice(counts.size, size=60, replace=False)
        chosen = np.unique(np.r_[rng.choice(boundary, size=min(60,len(boundary)), replace=False), close, random])
        true_h, true_w = row['unpadded_shape_yx']
        for index in chosen:
            ty, tx = np.unravel_index(int(index), counts.shape)
            y0, x0 = int(ty)*224, int(tx)*224
            tile = rasterize(((polygon, 1) for polygon in polygons),
                             out_shape=(224,224), transform=Affine.translation(x0,y0),
                             fill=0, all_touched=False, dtype='uint8')
            if y0+224 > true_h:
                tile[max(0,true_h-y0):] = 0
            if x0+224 > true_w:
                tile[:,max(0,true_w-x0):] = 0
            if int(tile.sum()) != int(counts[ty,tx]):
                raise AssertionError(f'{slide}: independent tile mismatch at {ty},{tx}')
        prior = next(x for x in proxy['rows'] if x['slide'] == slide)
        records.append({
            'slide': slide, 'counts_sha256': sha(counts_path),
            'exact_eligible_tiles': row['eligible_tiles'],
            'proxy_4um_eligible_tiles': prior['proxy_tiles_gt90_percent_tissue'],
            'exact_minus_proxy': row['eligible_tiles']-prior['proxy_tiles_gt90_percent_tissue'],
            'independent_tile_windows_checked': len(chosen),
            'strict_threshold_integer_rule': '10 * tissue_pixels > 9 * 50176',
        })
    assert sum(x['exact_eligible_tiles'] for x in records) == run['joint_eligible_tiles']
    result = {
        'status': 'CAM_PAIR_EXACT_0P5UM_ADMISSION_INDEPENDENTLY_AUDITED',
        'source_run_sha256': sha(RUN), 'source_proxy_sha256': sha(PROXY),
        'rows': records, 'joint_eligible_tiles': run['joint_eligible_tiles'],
        'default_joint_cap_exceeded': run['joint_eligible_tiles'] > 10000,
        'source_only': True, 'expert_reference_opened': False,
        'cam_specific_fit_or_scoring': False, 'online_publication': False,
    }
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'joint_eligible_tiles':result['joint_eligible_tiles'],
                      'windows_audited':sum(x['independent_tile_windows_checked'] for x in records)}))


if __name__ == '__main__':
    audit()
