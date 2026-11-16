#!/usr/bin/env python3
"""Align only the two official CAM training masks to audited 0.5-µm H&E grids.

Nearest-neighbor conversion preserves official codes: 0 unscored, 1 detected
normal tissue, 2 annotated metastasis. No test mask or threshold is read.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import pyvips

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
V19 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v19/TEST_IDS_FROZEN.json'
V31 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v31/TRAINING_REFS_STATE.json'
GATE = ROOT / 'bora-r-camelyon-2026-11-14-v2/LOCAL_CC0_USE_GATE.json'
OUT = DATA / 'external-camelyon16-five-slide-2026-11-14-v42/training-class-masks'
CAL = {
    'normal_093': DATA / 'external-camelyon16-pair-2026-11-13-v48/calibrated/normal_093/STATE.json',
    'tumor_093': DATA / 'external-camelyon16-pair-2026-11-13-v49/calibrated/tumor_093/STATE.json',
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()

def atomic(path, value):
    temp = path.with_name(path.name + '.partial')
    with temp.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)

def run(slide):
    if slide not in CAL:
        raise ValueError('Only the two frozen CAM training slides are admitted')
    gate = json.loads(GATE.read_text())
    frozen = json.loads(V19.read_text())
    refs = json.loads(V31.read_text())
    if (gate['status'] != 'LOCAL_ANALYSIS_AUTHORIZED_PUBLICATION_HELD' or
        not gate['local_cam_specific_training_allowed'] or
        gate['new_cam_online_publication_allowed'] or
        frozen['training_ids'] != ['tumor_093', 'normal_093'] or
        frozen['test_ids'] != ['tumor_084', 'tumor_091', 'normal_019'] or
        refs['status'] != 'COMPLETE_SOURCE_BYTES_VERIFIED_NOT_OPENED'):
        raise PermissionError('Local-only five-slide source gate differs')
    ref = next(r for r in refs['completed'] if r['slide_id'] == slide)
    mask_path = Path(ref['mask_path'])
    if sha(mask_path) != ref['sha256']:
        raise ValueError('Training mask differs from official verified source')
    state = json.loads(CAL[slide].read_text())
    if state['status'] != 'COMPLETE' or sha(state['target_image']) != state['target_sha256']:
        raise ValueError('Audited calibrated H&E image differs')
    directory = OUT / slide
    if directory.exists():
        raise FileExistsError('Preserve prior training-mask attempt; use new version')
    directory.mkdir(parents=True)
    receipt = {'format': 'bora-cam-training-mask-0p5-v1', 'status': 'RUNNING',
               'slide_id': slide, 'official_mask_path': str(mask_path),
               'official_mask_sha256': ref['sha256'], 'official_mask_md5': ref['official_md5'],
               'calibrated_state_path': str(CAL[slide]),
               'calibrated_state_sha256': sha(CAL[slide]),
               'calibrated_image_sha256': state['target_sha256'],
               'started_unix': time.time(), 'pid': os.getpid(),
               'test_reference_opened': False, 'online_publication': False}
    atomic(directory / 'STATE.json', receipt)
    try:
        mask = pyvips.Image.tiffload(str(mask_path), page=0, access='sequential')
        native_h, native_w = state['native_shape_yx']
        if (mask.height, mask.width, mask.bands, mask.format) != (native_h, native_w, 1, 'uchar'):
            raise ValueError('Official mask native grid or pixel type differs')
        h, w = state['unpadded_shape_yx']
        ph, pw, _ = state['target_shape_yxc']
        mapped = mask.resize(w / native_w, vscale=h / native_h, kernel='nearest')
        if (mapped.height, mapped.width) != (h, w):
            raise ValueError('Nearest-neighbor mask output dimensions differ')
        padded = mapped.embed(0, 0, pw, ph, extend='black')
        temp = directory / 'official-class-0p5.partial.tif'
        final = directory / 'official-class-0p5.tif'
        padded.tiffsave(str(temp), tile=True, tile_width=256, tile_height=256,
                        compression='zstd', predictor='none', bigtiff=True,
                        pyramid=False, xres=2000, yres=2000)
        os.replace(temp, final)
        stored = pyvips.Image.new_from_file(str(final), access='random')
        if (stored.height, stored.width, stored.bands, stored.format) != (ph, pw, 1, 'uchar'):
            raise ValueError('Persisted class mask geometry differs')
        preview = stored.crop(0, 0, w, h).thumbnail_image(1600)
        preview_path = directory / 'class-mask-overview.png'
        preview.write_to_file(str(preview_path))
        receipt.update(status='COMPLETE', completed_unix=time.time(),
                       native_shape_yx=[native_h, native_w],
                       unpadded_shape_yx=[h, w], padded_shape_yx=[ph, pw],
                       class_mask_path=str(final), class_mask_sha256=sha(final),
                       class_mask_bytes=final.stat().st_size,
                       preview_sha256=sha(preview_path),
                       transform='native level-0 mask to 0.5-µm H&E grid, nearest neighbor; right/bottom code-0 padding',
                       peak_host_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic(directory / 'STATE.json', receipt)
        print(json.dumps({'slide_id': slide, 'status': 'COMPLETE',
                          'mask_bytes': receipt['class_mask_bytes']}))
    except BaseException as error:
        receipt.update(status='FAILED_PRESERVE_PARTIAL', failure_type=type(error).__name__,
                       failure=str(error), failed_unix=time.time())
        atomic(directory / 'STATE.json', receipt)
        raise

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: align_training_masks.py tumor_093|normal_093')
    run(sys.argv[1])
