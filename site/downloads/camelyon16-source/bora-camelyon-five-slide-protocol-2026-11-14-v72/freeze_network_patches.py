"""Freeze training-only spatially separated CAM pair network patch centres."""
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE = Path(__file__).parent
LABELS = DATA / 'external-camelyon16-five-slide-2026-11-14-v58/training-labels'
BLOCK = 8192
PATCH = 512
MARGIN = PATCH // 2
SEED = 20261114
NORMAL_VALIDATION_BLOCKS = {(6, 2), (8, 3)}
TUMOR_VALIDATION_BLOCKS = {(1, 4)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def token_labels(path):
    with rasterio.open(path) as src:
        ys = np.arange(7, src.height, 14)
        xs = np.arange(7, src.width, 14)
        result = np.empty((len(ys), len(xs)), np.uint8)
        for row, y in enumerate(ys):
            result[row] = src.read(1, window=Window(0, int(y), src.width, 1))[0, xs]
        return result, (src.height, src.width)


def candidates(labels, shape, cls, split, slide):
    indices = np.argwhere(labels == cls)
    xy = indices[:, ::-1] * 14 + 7
    x, y = xy[:, 0], xy[:, 1]
    by, bx = y // BLOCK, x // BLOCK
    inside = ((y >= MARGIN) & (x >= MARGIN) &
              (y + MARGIN <= shape[0]) & (x + MARGIN <= shape[1]) &
              (y % BLOCK >= MARGIN) & (x % BLOCK >= MARGIN) &
              (y % BLOCK <= BLOCK - MARGIN) & (x % BLOCK <= BLOCK - MARGIN))
    held = TUMOR_VALIDATION_BLOCKS if slide == 'tumor_093' else NORMAL_VALIDATION_BLOCKS
    held_bool = np.fromiter(((int(yy), int(xx)) in held for yy, xx in zip(by, bx)),
                            dtype=bool, count=len(by))
    chosen = inside & (held_bool if split == 'inner_validation' else ~held_bool)
    return xy[chosen]


def main():
    out = HERE / 'PATCH_INDEX_FROZEN.json'
    if out.exists():
        raise FileExistsError(out)
    counts = {'inner_fit': 256, 'inner_validation': 64}
    rows = []
    sources = []
    for slide, classes in (('tumor_093', (1, 2)), ('normal_093', (1,))):
        path = LABELS / slide / 'training-classes-in-support.tif'
        audit = ROOT / f'bora-camelyon-five-slide-protocol-2026-11-14-v60/{slide.upper()}_TRAINING_LABELS_AUDIT_VERIFIED.json'
        proof = json.loads(audit.read_text())
        if proof['status'] != 'VERIFIED' or sha(path) != proof['clipped_mask_sha256']:
            raise ValueError('Clipped training mask differs')
        labels, shape = token_labels(path)
        for cls in classes:
            for split, limit in counts.items():
                pool = candidates(labels, shape, cls, split, slide)
                if len(pool) < limit:
                    raise ValueError(f'Insufficient {slide} class {cls} {split} centres: {len(pool)}')
                seed = int.from_bytes(hashlib.sha256(f'{SEED}:{slide}:{cls}:{split}'.encode()).digest()[:8], 'big')
                indices = np.random.default_rng(seed).choice(len(pool), size=limit, replace=False)
                for x, y in pool[indices]:
                    rows.append({'slide': slide, 'split': split, 'official_class_id': cls,
                                 'x_center_0p5': int(x), 'y_center_0p5': int(y),
                                 'patch_size_0p5_px': PATCH})
        sources.append({'slide': slide, 'mask_path': str(path), 'mask_sha256': sha(path),
                        'mask_audit_sha256': sha(audit), 'shape_yx': list(shape)})
    for slide in ('tumor_093', 'normal_093'):
        fit = np.asarray([(x['x_center_0p5'], x['y_center_0p5']) for x in rows if x['slide'] == slide and x['split'] == 'inner_fit'])
        valid = np.asarray([(x['x_center_0p5'], x['y_center_0p5']) for x in rows if x['slide'] == slide and x['split'] == 'inner_validation'])
        if np.any((np.abs(fit[:, None, 0] - valid[None, :, 0]) < PATCH) &
                  (np.abs(fit[:, None, 1] - valid[None, :, 1]) < PATCH)):
            raise ValueError(f'{slide} fit/validation patch rectangles overlap')
    result = {'status': 'TRAINING_ONLY_PATCH_INDEX_FROZEN', 'format': 'bora-cam-pair-network-patches-v1',
              'source_mask_receipts': sources, 'seed': SEED, 'patch_size_0p5_px': PATCH,
              'spatial_block_size_0p5_px': BLOCK, 'minimum_patch_centre_distance_from_block_edge': MARGIN,
              'tumor_validation_blocks_yx': sorted([list(x) for x in TUMOR_VALIDATION_BLOCKS]),
              'normal_validation_blocks_yx': sorted([list(x) for x in NORMAL_VALIDATION_BLOCKS]),
              'per_slide_class_counts_per_split': counts,
              'class_crosswalk': {'0': 'ignore', '1': 'normal network output 0',
                                  '2': 'metastasis network output 1'},
              'patches': rows, 'test_reference_opened': False, 'online_publication': False}
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'patches': len(rows),
                      'by_split': {s: sum(x['split'] == s for x in rows) for s in counts},
                      'sha256': sha(out)}))


if __name__ == '__main__':
    main()
