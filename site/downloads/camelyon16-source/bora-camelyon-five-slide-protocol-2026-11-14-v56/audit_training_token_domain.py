"""Measure official CAM training labels versus source-only H&E support at token centres."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.windows import Window

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
MASKS = DATA / 'external-camelyon16-five-slide-2026-11-14-v51/training-class-masks'
SUPPORT = DATA / 'external-camelyon16-pair-2026-11-13-v54/he-support'
OUT = Path(__file__).parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main(slide):
    if slide not in ('tumor_093', 'normal_093'):
        raise ValueError('Only the two CAM training slides are admitted')
    out = OUT / f'{slide.upper()}_TOKEN_DOMAIN_AUDIT.json'
    if out.exists():
        raise FileExistsError(out)
    state_path = MASKS / slide / 'STATE.json'
    state = json.loads(state_path.read_text())
    audit_path = ROOT / f'bora-camelyon-five-slide-protocol-2026-11-14-v54/{slide.upper()}_MASK_ALIGNMENT_AUDIT_VERIFIED.json'
    assert json.loads(audit_path.read_text())['status'] == 'VERIFIED'
    assert state['status'] == 'COMPLETE' and not state['test_reference_opened']
    support_path = SUPPORT / slide / 'tissue-area.geojson'
    geo = json.loads(support_path.read_text())
    assert geo['coordinate_system'] == 'calibrated_image_pixel_xy'
    assert geo['image_shape_yx'] == state['padded_shape_yx']
    assert geo['source_image_sha256'] == state['calibrated_image_sha256']
    ph, pw = state['padded_shape_yx']
    ys = np.arange(7, ph, 14)
    xs = np.arange(7, pw, 14)
    shape = len(ys), len(xs)
    support = rasterize(((f['geometry'], 1) for f in geo['features']),
                        out_shape=shape, transform=Affine.scale(14),
                        fill=0, dtype=np.uint8).astype(bool)
    labels = np.empty(shape, np.uint8)
    with rasterio.open(state['class_mask_path']) as source:
        for row, y in enumerate(ys):
            labels[row] = source.read(1, window=Window(0, int(y), pw, 1))[0, xs]
    if labels.max(initial=0) > 2:
        raise ValueError('Unknown official training code')
    within = np.bincount(labels[support].ravel(), minlength=3)[:3]
    outside = np.bincount(labels[~support].ravel(), minlength=3)[:3]
    result = {'format': 'bora-cam-two-slide-training-token-domain-audit-v1',
              'status': 'VERIFIED', 'slide_id': slide,
              'aligned_mask_sha256': state['class_mask_sha256'],
              'alignment_audit_sha256': sha(audit_path),
              'source_only_support_sha256': sha(support_path),
              'token_stride_calibrated_px': 14, 'token_grid_yx': list(shape),
              'class_counts_within_support_0_1_2': within.tolist(),
              'class_counts_outside_support_0_1_2': outside.tolist(),
              'tissue_token_count': int(support.sum()),
              'score_code_contract': {'0': 'unscored', '1': 'automatically detected normal tissue',
                                      '2': 'annotated metastasis'},
              'test_reference_opened': False, 'online_publication': False}
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: audit_training_token_domain.py tumor_093|normal_093')
    main(sys.argv[1])
