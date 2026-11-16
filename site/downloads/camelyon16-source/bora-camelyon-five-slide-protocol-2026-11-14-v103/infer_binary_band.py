#!/usr/bin/env python3
"""One bounded source-only CAM binary network band at the exact 4-um grid."""
import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch import nn

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE = Path(__file__).resolve().parent
MODEL_CODE = ROOT / 'migration/benchmark1-eight-class-comparator-refit-2026-11-12-v2/models.py'
GRID_CODE = ROOT / 'migration/bora-camelyon-five-slide-protocol-2026-11-14-v103/binary_grid.py'
OUT = DATA / 'external-camelyon16-five-slide-2026-11-14-v103/binary-network-bands'
sys.path.insert(0, str(MODEL_CODE.parent))
sys.path.insert(0, str(GRID_CODE.parent))
from models import make_model
from binary_grid import accumulate, sampled_axis, starts


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, obj):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(temporary, path)


def save_array(path, array):
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide', choices=('tumor_084', 'tumor_091', 'normal_019'), required=True)
    parser.add_argument('--method', choices=('unet', 'deeplabv3plus'), required=True)
    parser.add_argument('--band', type=int, required=True)
    args = parser.parse_args()
    frozen_path = HERE / 'INPUTS_FROZEN.json'
    frozen = json.loads(frozen_path.read_text())
    if sha(__file__) != frozen['source_sha256']['infer_band'] or sha(GRID_CODE) != frozen['source_sha256']['grid_accumulation']:
        raise RuntimeError('Frozen band source differs')
    if sha(MODEL_CODE) != frozen['source_sha256']['model_architecture']:
        raise RuntimeError('Frozen network architecture differs')
    if sha(ROOT / 'migration/bora-camelyon-five-slide-protocol-2026-11-14-v44/TEST_FEATURES_AUDIT_VERIFIED.json') != frozen['source_sha256']['test_feature_audit']:
        raise RuntimeError('Frozen source-only test feature audit differs')
    slide = frozen['slides'][args.slide]
    model = frozen['models'][args.method]
    checkpoint = Path(model['checkpoint_path'])
    if sha(checkpoint) != model['checkpoint_sha256']:
        raise RuntimeError('Checkpoint hash differs')
    state = Path(slide['calibrated_state_path'])
    if sha(state) != slide['calibrated_state_sha256']:
        raise RuntimeError('Calibrated image receipt differs')
    source_receipt = json.loads(state.read_text())
    if source_receipt['status'] != 'COMPLETE':
        raise RuntimeError('Calibrated source receipt is incomplete')
    source_entry = next(x for x in source_receipt['completed']
                        if x['slide_id'] == args.slide)
    if (source_entry['calibrated_sha256'] != slide['calibrated_image_sha256'] or
            source_entry['calibrated_path'] != slide['calibrated_image_path']):
        raise RuntimeError('Source image identity differs')
    height, width = slide['calibrated_shape_yx']
    if source_entry['padded_shape_yx'] != [height, width]:
        raise RuntimeError('Calibrated shape differs')
    sy, sx = sampled_axis(height), sampled_axis(width)
    tops, lefts = starts(height), starts(width)
    group_size = frozen['patch_rows_per_band']
    groups = (len(tops) + group_size - 1) // group_size
    if args.band < 0 or args.band >= groups:
        raise ValueError('Band outside frozen slide geometry')
    selected_tops = tops[args.band * group_size:(args.band + 1) * group_size]
    y0 = int(np.searchsorted(sy, selected_tops[0], side='left'))
    y1 = int(np.searchsorted(sy, selected_tops[-1] + 512, side='left'))
    slab_bytes = 2 * (y1 - y0) * len(sx) * 4 + (y1 - y0) * len(sx)
    if slab_bytes > frozen['max_band_array_bytes']:
        raise RuntimeError('Band array exceeds frozen peak bound')
    folder = OUT / args.method / args.slide / f'band-{args.band:04d}'
    if folder.exists():
        raise FileExistsError('No in-place band retry; use isolated recovery')
    folder.mkdir(parents=True)
    start = time.monotonic()
    try:
        torch.set_num_threads(4)
        network = make_model(args.method)
        if args.method == 'unet':
            previous = network.head
            network.head = nn.Conv2d(previous.in_channels, 2, 1)
        else:
            previous = network.decoder[-1]
            network.decoder[-1] = nn.Conv2d(previous.in_channels, 2, 1)
        network.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        network = network.cuda().eval()
        scores = np.zeros((2, y1 - y0, len(sx)), dtype=np.float32)
        counts = np.zeros((y1 - y0, len(sx)), dtype=np.uint8)
        with rasterio.open(slide['calibrated_image_path']) as image:
            if (image.height, image.width, image.count) != (height, width, 3):
                raise RuntimeError('Tiled RGB source geometry differs')
            with torch.no_grad():
                for top in selected_tops:
                    for left in lefts:
                        patch = np.moveaxis(image.read(window=Window(left, top, 512, 512)), 0, -1)
                        if patch.shape != (512, 512, 3):
                            raise RuntimeError('Incomplete RGB patch')
                        tensor = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1))).float()[None].cuda() / 255
                        with torch.autocast('cuda', dtype=torch.float16):
                            logits = network(tensor)[0].float().cpu().numpy()
                        accumulate(scores, counts, logits, top, left, sy[y0:y1], sx)
        if not np.all(counts):
            raise RuntimeError('Band left some comparison pixels uncovered')
        score_path = folder / 'score-sum.npy'
        count_path = folder / 'patch-count.npy'
        save_array(score_path, scores)
        save_array(count_path, counts)
        record = {'status': 'CAM_BINARY_NETWORK_SOURCE_ONLY_BAND_COMPLETE',
                  'method': args.method, 'slide': args.slide, 'band': args.band,
                  'bands_total': groups, 'comparison_row_start': y0,
                  'comparison_row_stop': y1, 'comparison_width': len(sx),
                  'selected_patch_tops': selected_tops,
                  'patches': len(selected_tops) * len(lefts),
                  'min_patch_coverage_in_band': int(counts.min()),
                  'max_patch_coverage_in_band': int(counts.max()),
                  'checkpoint_sha256': model['checkpoint_sha256'],
                  'calibrated_image_sha256': slide['calibrated_image_sha256'],
                  'source_sha256': sha(__file__),
                  'grid_source_sha256': sha(GRID_CODE),
                  'input_freeze_sha256': sha(frozen_path),
                  'score_path': str(score_path), 'score_sha256': sha(score_path),
                  'count_path': str(count_path), 'count_sha256': sha(count_path),
                  'slab_array_bytes': slab_bytes,
                  'seconds': time.monotonic() - start,
                  'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'peak_cuda_bytes': int(torch.cuda.max_memory_allocated()),
                  'expert_reference_opened': False}
        atomic(folder / 'BAND_COMPLETE.json', record)
        print(json.dumps({'band': args.band, 'slide': args.slide, 'method': args.method,
                          'seconds': record['seconds'], 'score_sha256': record['score_sha256']}), flush=True)
    except BaseException as error:
        atomic(folder / 'BAND_FAILED.json', {'status': 'FAILED_LATCHED_PRESERVE_PARTIAL',
                                            'error': repr(error), 'slide': args.slide,
                                            'method': args.method, 'band': args.band,
                                            'expert_reference_opened': False})
        raise


if __name__ == '__main__':
    main()
