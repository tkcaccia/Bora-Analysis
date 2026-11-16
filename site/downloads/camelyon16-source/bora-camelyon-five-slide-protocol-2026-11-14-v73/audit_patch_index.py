"""Independently verify every CAM pair network patch against frozen masks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
SOURCE = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v72/PATCH_INDEX_FROZEN.json'
OUT = Path(__file__).parent / 'PATCH_INDEX_AUDIT_VERIFIED.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    spec = json.loads(SOURCE.read_text())
    if spec['status'] != 'TRAINING_ONLY_PATCH_INDEX_FROZEN' or spec['test_reference_opened']:
        raise ValueError('Patch source status differs')
    groups = {}
    validated = []
    for row in spec['patches']:
        groups.setdefault(row['slide'], []).append(row)
    for item in spec['source_mask_receipts']:
        slide = item['slide']
        path = Path(item['mask_path'])
        if sha(path) != item['mask_sha256']:
            raise ValueError('Training mask source hash differs')
        with rasterio.open(path) as mask:
            fit, valid = [], []
            for row in groups[slide]:
                x, y = row['x_center_0p5'], row['y_center_0p5']
                if row['patch_size_0p5_px'] != 512 or x % 14 != 7 or y % 14 != 7:
                    raise ValueError('Patch centre/stride differs')
                if not (256 <= x <= mask.width - 256 and 256 <= y <= mask.height - 256):
                    raise ValueError('Patch exceeds source')
                if min(x % 8192, y % 8192, 8192 - x % 8192, 8192 - y % 8192) < 256:
                    raise ValueError('Patch crosses an 8192px spatial split block')
                got = int(mask.read(1, window=Window(x, y, 1, 1))[0, 0])
                if got != row['official_class_id'] or got not in (1, 2):
                    raise ValueError('Patch centre class differs')
                held = {(1, 4)} if slide == 'tumor_093' else {(6, 2), (8, 3)}
                is_held = (y // 8192, x // 8192) in held
                if is_held != (row['split'] == 'inner_validation'):
                    raise ValueError('Patch split differs')
                (valid if is_held else fit).append((x, y))
            fit, valid = np.asarray(fit), np.asarray(valid)
            if np.any((np.abs(fit[:, None, 0] - valid[None, :, 0]) < 512) &
                      (np.abs(fit[:, None, 1] - valid[None, :, 1]) < 512)):
                raise ValueError('Patch rectangles overlap across split')
            validated.append({'slide': slide, 'fit': len(fit), 'inner_validation': len(valid),
                              'mask_sha256': sha(path)})
    if sum(x['fit'] for x in validated) != 768 or sum(x['inner_validation'] for x in validated) != 192:
        raise ValueError('Frozen patch count differs')
    out = {'status': 'VERIFIED', 'format': 'bora-cam-network-training-patch-audit-v1',
           'patch_index_sha256': sha(SOURCE), 'rows': validated,
           'positive_and_normal_centres_within_frozen_support': True,
           'fit_validation_patch_overlap_count': 0,
           'test_reference_opened': False, 'online_publication': False}
    OUT.write_text(json.dumps(out, indent=2) + '\n')
    print(OUT)


if __name__ == '__main__':
    main()
