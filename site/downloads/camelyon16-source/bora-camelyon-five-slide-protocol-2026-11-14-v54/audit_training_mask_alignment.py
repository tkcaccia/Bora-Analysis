#!/usr/bin/env python3
"""Independently compare every aligned training-mask pixel to official native TIFF."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
V19 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v19/TEST_IDS_FROZEN.json'
V31 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v31/TRAINING_REFS_STATE.json'
OUTPUT = DATA / 'external-camelyon16-five-slide-2026-11-14-v51/training-class-masks'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('slide', choices=('tumor_093', 'normal_093'))
    args = parser.parse_args()
    out = Path(__file__).parent / f'{args.slide.upper()}_MASK_ALIGNMENT_AUDIT_VERIFIED.json'
    if out.exists():
        raise FileExistsError(out)
    ids = json.loads(V19.read_text())
    refs = json.loads(V31.read_text())
    assert args.slide in ids['training_ids'] and args.slide not in ids['test_ids']
    state_path = OUTPUT / args.slide / 'STATE.json'
    state = json.loads(state_path.read_text())
    assert state['status'] == 'COMPLETE' and not state['test_reference_opened']
    source_entry = next(r for r in refs['completed'] if r['slide_id'] == args.slide)
    source_path = Path(source_entry['mask_path'])
    mask_path = Path(state['class_mask_path'])
    assert sha(source_path) == source_entry['sha256'] == state['official_mask_sha256']
    assert sha(mask_path) == state['class_mask_sha256']
    cal_state = json.loads(Path(state['calibrated_state_path']).read_text())
    assert sha(state['calibrated_state_path']) == state['calibrated_state_sha256']
    assert cal_state['target_sha256'] == state['calibrated_image_sha256']
    nh, nw = state['native_shape_yx']
    h, w = state['unpadded_shape_yx']
    ph, pw = state['padded_shape_yx']
    service_name = f'bora-cam16-align-{args.slide.replace("_", "")}-v51.service'
    fields = dict(line.split('=', 1) for line in subprocess.check_output(
        ['systemctl', '--user', 'show', service_name,
         '-p', 'ActiveState', '-p', 'SubState', '-p', 'Result',
         '-p', 'ExecMainStatus', '-p', 'MainPID', '-p', 'ControlGroup'], text=True).splitlines())
    assert fields['ActiveState'] == 'inactive' and fields['SubState'] == 'dead'
    assert fields['Result'] == 'success' and fields['ExecMainStatus'] == '0'
    assert fields['MainPID'] == '0' and fields['ControlGroup'] == ''
    counts = np.zeros(3, np.int64)
    pixels = 0
    with rasterio.open(source_path) as native, rasterio.open(mask_path) as aligned:
        assert (native.height, native.width) == (nh, nw)
        assert (aligned.height, aligned.width, aligned.count, aligned.dtypes[0]) == (ph, pw, 1, 'uint8')
        for _, win in aligned.block_windows(1):
            y0, x0 = int(win.row_off), int(win.col_off)
            bh, bw = int(win.height), int(win.width)
            actual = aligned.read(1, window=win)
            expected = np.zeros((bh, bw), np.uint8)
            y1, x1 = min(y0 + bh, h), min(x0 + bw, w)
            if y0 < h and x0 < w:
                yi = np.minimum(((np.arange(y0, y1, dtype=np.float64) + .5) * nh / h).astype(np.int64), nh-1)
                xi = np.minimum(((np.arange(x0, x1, dtype=np.float64) + .5) * nw / w).astype(np.int64), nw-1)
                ymin, ymax = int(yi.min()), int(yi.max())
                xmin, xmax = int(xi.min()), int(xi.max())
                raw = native.read(1, window=Window(xmin, ymin, xmax-xmin+1, ymax-ymin+1))
                expected[:y1-y0, :x1-x0] = raw[(yi-ymin)[:, None], (xi-xmin)[None, :]]
            if not np.array_equal(actual, expected):
                raise ValueError(f'Alignment mismatch at output block {y0},{x0}')
            if actual.max(initial=0) > 2:
                raise ValueError('Unexpected semantic code')
            counts += np.bincount(actual.ravel(), minlength=3)[:3]
            pixels += actual.size
    assert pixels == ph * pw and counts.tolist() == state['class_counts_0_1_2']
    gpu = subprocess.check_output(['timeout', '8', 'nvidia-smi',
                                   '--query-compute-apps=pid,gpu_uuid',
                                   '--format=csv,noheader'], text=True)
    assert not gpu.strip()
    result = {'format': 'bora-cam-training-mask-alignment-pixel-audit-v1',
              'status': 'VERIFIED', 'slide_id': args.slide,
              'source_mask_sha256': sha(source_path),
              'aligned_mask_sha256': sha(mask_path),
              'state_sha256': sha(state_path),
              'class_counts_0_1_2': counts.tolist(),
              'all_aligned_pixels_equal_independent_native_resampling': True,
              'terminal_service': fields, 'gpu_contexts_after': [],
              'test_reference_opened': False, 'online_publication': False}
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'slide': args.slide, 'status': 'VERIFIED',
                      'counts': counts.tolist()}))

if __name__ == '__main__':
    main()
