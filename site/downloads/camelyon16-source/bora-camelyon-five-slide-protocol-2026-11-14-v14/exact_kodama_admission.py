#!/usr/bin/env python3
"""Source-only, bounded 0.5-um KODAMA tile admission for the existing CAM pair.

Uses accepted H&E-only polygons, never source RGB, Virchow2, or expert masks.
This is a capacity preflight, not permission to fit a CAM model.
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
ACCEPTED = ROOT/'migration/external-benchmark-wsss-camelyon-2026-11-13-v54/SUPPORT_VERIFIED.json'
GATE = ROOT/'migration/bora-r-camelyon-2026-11-14-v1/PERMISSION_GATE_2026-11-14.json'
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
    if gate['status'] != 'PENDING' or gate['cam_specific_training_allowed_now']:
        raise RuntimeError('This source-only pending-rights preflight is obsolete')
    verified = json.loads(ACCEPTED.read_text())
    assert verified['status'] == 'TWO_CAMELYON16_HE_SUPPORTS_VISUALLY_REVIEWED_AND_ROUNDTRIP_VERIFIED'
    rows = []
    for accepted in verified['rows']:
        slide = accepted['slide']
        assert slide in ('normal_093', 'tumor_093')
        path = DATA/f'external-camelyon16-pair-2026-11-13-v54/he-support/{slide}/tissue-area.geojson'
        assert sha(path) == accepted['geojson_sha256']
        collection = json.loads(path.read_text())
        assert collection['coordinate_system'] == 'calibrated_image_pixel_xy'
        assert collection['source_image_sha256'] == accepted['source_image_sha256']
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
            'calibrated_image_sha256': accepted['source_image_sha256'],
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
        'format': 'bora-camelyon16-exact-native-tile-admission-v1',
        'status': 'SOURCE_ONLY_EXACT_0P5UM_TISSUE_TILE_CAPACITY',
        'accepted_support_receipt_sha256': sha(ACCEPTED),
        'permission_gate_sha256': sha(GATE),
        'script_sha256': sha(__file__),
        'tile_edge_px': TILE, 'calibrated_resolution_um_per_pixel': 0.5,
        'background_rule': 'strictly less than 10%; padding counts as background',
        'rows': rows, 'joint_eligible_tiles': total,
        'package_default_max_joint_tiles': 10000,
        'default_joint_cap_exceeded': total > 10000,
        'cam_source_rgb_features_expert_references_opened': False,
        'cam_specific_training_or_scoring_performed': False,
        'new_online_publication': False,
    })
    print('joint', total, flush=True)


if __name__ == '__main__':
    main()
