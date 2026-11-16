#!/usr/bin/env python3
"""Source-only, bounded 0.5-um KODAMA tile admission for three CAM test slides.

Uses accepted H&E-only polygons, never source RGB, Virchow2, or expert masks.
This is a capacity preflight, not a fitted or semantically named CAM model.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.features import bounds, rasterize


ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
OUT = Path(__file__).parent
ACCEPTED = ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-14-v34/SUPPORT_AUDIT_VERIFIED.json'
VISUAL = ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-14-v34/VISUAL_REVIEW_VERIFIED.json'
GATE = ROOT/'migration/bora-r-camelyon-2026-11-14-v2/LOCAL_CC0_USE_GATE.json'
TILE = 224


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(temp, path)


def main():
    if (OUT/'EXACT_ADMISSION.json').exists():
        raise FileExistsError('Preserve previous admission; use a new version')
    gate = json.loads(GATE.read_text())
    if gate['status'] != 'LOCAL_ANALYSIS_AUTHORIZED_PUBLICATION_HELD' or not gate['local_three_slide_scoring_allowed'] or gate['new_cam_online_publication_allowed']:
        raise RuntimeError('Local-only CAM use gate changed')
    verified = json.loads(ACCEPTED.read_text())
    visual = json.loads(VISUAL.read_text())
    assert verified['status'] == 'PIXEL_VERIFIED_VISUAL_REVIEW_PENDING' and visual['status'] == 'THREE_TEST_SUPPORTS_VISUALLY_ACCEPTED'
    rows = []
    for accepted in verified['rows']:
        slide = accepted['slide_id']
        assert slide in ('normal_019', 'tumor_084', 'tumor_091')
        path = DATA/f'external-camelyon16-five-slide-2026-11-14-v33/he-support/{slide}/tissue-area.geojson'
        assert sha(path) == accepted['geojson_sha256']
        collection = json.loads(path.read_text())
        assert collection['coordinate_system'] == 'calibrated_image_pixel_xy'
        assert collection['source_image_sha256']
        height, width = collection['image_shape_yx']
        true_height, true_width = collection['unpadded_image_shape_yx']
        assert height % TILE == width % TILE == 0
        assert 0 < true_height <= height and 0 < true_width <= width
        geometries = [item['geometry'] for item in collection['features']]
        bounded = [(geometry, bounds(geometry)) for geometry in geometries]
        counts = np.zeros((height//TILE, width//TILE), dtype=np.uint16)
        # One complete tile row at a time: at most 224 * 47,488 mask pixels.
        # Polygon coordinates are in calibrated pixels; rasterio uses pixel
        # centres and Affine.translation(0, y0) for a global-coordinate strip.
        for row in range(counts.shape[0]):
            y0, y1 = row*TILE, (row+1)*TILE
            active = [geometry for geometry, (_, min_y, _, max_y) in bounded
                      if min_y < y1 and max_y > y0]
            if not active:
                continue
            stripe = rasterize(((geometry, 1) for geometry in active),
                               out_shape=(TILE, width),
                               transform=Affine.translation(0, y0),
                               fill=0, all_touched=False, dtype='uint8')
            if y1 > true_height:
                stripe[max(0, true_height-y0):] = 0
            if true_width < width:
                stripe[:, true_width:] = 0
            counts[row] = stripe.reshape(TILE, width//TILE, TILE).sum(axis=(0, 2))
        # Strictly less than 10% background; exact integer comparison.
        eligible = counts.astype(np.int32)*10 > TILE*TILE*9
        output = OUT/f'{slide}-tissue-pixels-per-tile.npy'
        np.save(output, counts, allow_pickle=False)
        rows.append({
            'slide': slide, 'accepted_geojson_sha256': sha(path),
            'calibrated_image_sha256': collection['source_image_sha256'],
            'calibrated_shape_yx': [height, width],
            'unpadded_shape_yx': [true_height, true_width],
            'tile_grid_shape_yx': list(counts.shape),
            'counts_path': str(output), 'counts_sha256': sha(output),
            'eligible_tiles': int(eligible.sum()),
            'fully_supported_tiles': int((counts == TILE*TILE).sum()),
            'any_tissue_tiles': int((counts > 0).sum()),
            'minimum_eligible_tissue_pixels': 45159,
            'sample_tissue_pixels': {f'{r},{c}': int(counts[r,c]) for r,c in
                [(0,0), (counts.shape[0]//2,counts.shape[1]//2),
                 (counts.shape[0]-1,counts.shape[1]-1)]},
        })
        print(slide, rows[-1]['eligible_tiles'], flush=True)
    total = sum(row['eligible_tiles'] for row in rows)
    atomic_json(OUT/'EXACT_ADMISSION.json', {
        'format': 'bora-camelyon16-test-exact-native-tile-admission-v1',
        'status': 'SOURCE_ONLY_EXACT_0P5UM_TISSUE_TILE_CAPACITY',
        'accepted_support_receipt_sha256': sha(ACCEPTED),
        'accepted_visual_review_sha256': sha(VISUAL),
        'permission_gate_sha256': sha(GATE),
        'script_sha256': sha(__file__),
        'tile_edge_px': TILE, 'calibrated_resolution_um_per_pixel': 0.5,
        'background_rule': 'strictly less than 10%; padding counts as background',
        'rows': rows, 'joint_eligible_tiles': total,
        'package_default_max_joint_tiles': 10000,
        'default_joint_cap_exceeded': total > 10000,
        'test_expert_references_opened': False,
        'cam_specific_scoring_performed': False,
        'new_online_publication': False,
    })
    print('joint', total, flush=True)


if __name__ == '__main__':
    main()
