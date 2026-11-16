"""Independent pixel and token audit of the two support-clipped CAM training masks."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.windows import Window

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
SRC = DATA / 'external-camelyon16-five-slide-2026-11-14-v51/training-class-masks'
OUT = DATA / 'external-camelyon16-five-slide-2026-11-14-v58/training-labels'
SUP = DATA / 'external-camelyon16-pair-2026-11-13-v54/he-support'
HERE = Path(__file__).parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main(slide):
    if slide not in ('tumor_093', 'normal_093'):
        raise ValueError('Only frozen CAM training slides are admitted')
    result = HERE / f'{slide.upper()}_TRAINING_LABELS_AUDIT_VERIFIED.json'
    if result.exists():
        raise FileExistsError(result)
    state = json.loads((OUT / slide / 'STATE.json').read_text())
    source = json.loads((SRC / slide / 'STATE.json').read_text())
    if state['status'] != 'COMPLETE' or source['status'] != 'COMPLETE':
        raise ValueError('Incomplete input')
    if sha(state['training_mask_path']) != state['training_mask_sha256']:
        raise ValueError('Clipped TIFF hash differs')
    if sha(source['class_mask_path']) != source['class_mask_sha256']:
        raise ValueError('Official aligned TIFF hash differs')
    geo_path = SUP / slide / 'tissue-area.geojson'
    geo = json.loads(geo_path.read_text())
    if sha(geo_path) != json.loads((OUT / slide / 'INPUTS_FROZEN.json').read_text())['source_only_support_sha256']:
        raise ValueError('Support hash differs')
    geom = [(f['geometry'], 1) for f in geo['features']]
    ph, pw = source['padded_shape_yx']
    counts = np.zeros(3, np.int64)
    outside = np.zeros(3, np.int64)
    matched = 0
    with rasterio.open(source['class_mask_path']) as native, rasterio.open(state['training_mask_path']) as clipped:
        if (native.height, native.width, clipped.height, clipped.width) != (ph, pw, ph, pw):
            raise ValueError('Mask registration differs')
        for y0 in range(0, ph, 896):
            height = min(896, ph - y0)
            window = Window(0, y0, pw, height)
            original = native.read(1, window=window)
            actual = clipped.read(1, window=window)
            support = rasterize(geom, out_shape=(height, pw),
                                transform=Affine.translation(0, y0), fill=0,
                                dtype=np.uint8).astype(bool)
            expected = np.where(support, original, 0).astype(np.uint8)
            if not np.array_equal(actual, expected):
                raise ValueError(f'Pixel mismatch in band {y0}')
            counts += np.bincount(actual.ravel(), minlength=3)[:3]
            outside += np.bincount(original[~support].ravel(), minlength=3)[:3]
            matched += actual.size
        ys = np.arange(7, ph, 14)
        xs = np.arange(7, pw, 14)
        tokens = np.empty((len(ys), len(xs)), np.uint8)
        for row, y in enumerate(ys):
            tokens[row] = clipped.read(1, window=Window(0, int(y), pw, 1))[0, xs]
    # All outside-support labels become zero; only positive-class counts match
    # the writer's within-support counts exactly.
    if counts[1:].tolist() != state['class_counts_inside_support_0_1_2'][1:]:
        raise ValueError('Class counts differ')
    if outside.tolist() != state['official_class_counts_outside_support_0_1_2']:
        raise ValueError('Outside-support class counts differ')
    token_audit = json.loads((ROOT / f'bora-camelyon-five-slide-protocol-2026-11-14-v56/{slide.upper()}_TOKEN_DOMAIN_AUDIT.json').read_text())
    token_counts = np.bincount(tokens.ravel(), minlength=3)[:3]
    if token_counts[1:].tolist() != token_audit['class_counts_within_support_0_1_2'][1:]:
        raise ValueError('Token-level class counts differ from independent reference/support audit')
    service = 'bora-cam16-training-labels-' + ('tumor' if slide == 'tumor_093' else 'normal') + '-v58.service'
    props = subprocess.run(['systemctl', '--user', 'show', service, '-p', 'ActiveState', '-p', 'Result',
                            '-p', 'MainPID', '-p', 'ControlGroup', '-p', 'ExecMainStatus'],
                           check=True, capture_output=True, text=True).stdout
    if 'ActiveState=inactive' not in props or 'Result=success' not in props or 'MainPID=0' not in props:
        raise ValueError('Owned service is not terminally clean')
    out = {'format': 'bora-cam-training-labels-independent-audit-v1', 'status': 'VERIFIED',
           'slide': slide, 'full_pixels_exact': matched,
           'clipped_mask_sha256': state['training_mask_sha256'],
           'official_aligned_mask_sha256': source['class_mask_sha256'],
           'source_only_support_sha256': sha(geo_path),
           'clipped_class_counts_0_1_2': counts.tolist(),
           'official_class_counts_outside_support_0_1_2': outside.tolist(),
           'token_class_counts_0_1_2': token_counts.tolist(),
           'service': props.strip().splitlines(), 'test_reference_opened': False,
           'online_publication': False}
    result.write_text(json.dumps(out, indent=2) + '\n')
    print(result)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: audit_training_labels.py tumor_093|normal_093')
    main(sys.argv[1])
