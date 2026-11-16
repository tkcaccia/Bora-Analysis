#!/usr/bin/env python3
"""Assemble audited CAM VISTA tile probabilities on the exact 4-um grid.

This is source-only processing. Class probabilities are blended with the
released VISTA Gaussian window, thresholded per class, and merged in descending
area order. Native output and the separately H&E-support-clipped output are
retained. No CAM expert reference is read here.
"""
import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import tifffile
from skimage.filters import threshold_otsu

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'migration/bora-camelyon-five-slide-protocol-2026-11-15-v132/VISTA_SOURCE_FROZEN.json'
AUDIT_DIR = ROOT / 'migration'
AUDIT_RUNS = {
    'normal_019': 'bora-camelyon-five-slide-protocol-2026-11-15-v137',
    'tumor_091': 'bora-camelyon-five-slide-protocol-2026-11-15-v139',
    'tumor_084': 'bora-camelyon-five-slide-protocol-2026-11-15-v140',
}
TILES = DATA / 'external-camelyon16-five-slide-2026-11-15-v136/vista-ranges'
OUTPUT = DATA / 'external-camelyon16-five-slide-2026-11-15-v141/vista-masks'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def gaussian_weight(height=128, width=128, sigma_scale=0.125):
    """The released 1024-pixel VISTA window sampled at 4-um pixel centres."""
    if (height, width) != (128, 128):
        raise ValueError('Frozen comparison tile geometry differs')
    y = np.linspace(-1, 1, 1024)
    x = np.linspace(-1, 1, 1024)
    xv, yv = np.meshgrid(x, y)
    sigma_x = sigma_scale * 2
    sigma_y = sigma_scale * 2
    weight = np.exp(-((xv**2) / (2 * sigma_x**2) +
                      (yv**2) / (2 * sigma_y**2)))
    weight /= weight.max()
    return weight.astype(np.float32)[4::8, 4::8]


def run(slide):
    freeze_path = HERE / 'ASSEMBLY_FROZEN.json'
    freeze = json.loads(freeze_path.read_text())
    audit_path = AUDIT_DIR / AUDIT_RUNS[slide] / 'VISTA_RANGES_AUDIT_VERIFIED.json'
    if sha(__file__) != freeze['assembly_source_sha256'] or sha(SOURCE) != freeze['vista_source_sha256']:
        raise RuntimeError('Assembly source or VISTA source freeze changed')
    if sha(audit_path) != freeze['independent_tile_audit_sha256'][slide]:
        raise RuntimeError('Independent VISTA tile audit changed')
    source = json.loads(SOURCE.read_text())
    audit = json.loads(audit_path.read_text())
    row = source['slides'][slide]
    if audit['status'] != 'VERIFIED' or audit['slide'] != slide or audit['selected_tiles'] != len(row['selected_tiles']) or audit['test_reference_opened']:
        raise RuntimeError('Slide outside independently audited source-only scope')
    support_receipt = json.loads(Path(row['support_receipt_path']).read_text())
    support_path = Path(support_receipt['support']['support_path'])
    if sha(support_path) != row['support_mask_sha256']:
        raise RuntimeError('Common H&E support changed')
    support_shape = tuple(support_receipt['support']['shape_yx'])
    support = np.memmap(support_path, mode='r', dtype=np.uint8, shape=support_shape)
    height, width = support.shape
    if [height, width] != freeze['comparison_shapes_yx'][slide] or support.dtype != np.uint8:
        raise RuntimeError('Comparison grid or H&E support differs')
    folder = OUTPUT / slide
    if folder.exists():
        raise FileExistsError('No in-place VISTA assembly retry')
    folder.mkdir(parents=True)
    start_time = time.monotonic()
    try:
        weight = gaussian_weight()
        components = []
        count = len(row['selected_tiles'])
        chunk = int(freeze['tiles_per_child'])
        if count != audit['selected_tiles'] or chunk != 100:
            raise RuntimeError('Tile count or bounded range differs')
        for class_index in range(13):
            totals = np.memmap(folder / 'probability-sum.partial.mmap', mode='w+',
                               dtype=np.float32, shape=(height, width))
            counts = np.memmap(folder / 'weight-sum.partial.mmap', mode='w+',
                               dtype=np.float32, shape=(height, width))
            totals[:] = 0
            counts[:] = 0
            for index, tile in enumerate(row['selected_tiles']):
                low = (index // chunk) * chunk
                high = min(low + chunk, count)
                path = TILES / slide / f'range-{low:04d}-{high:04d}' / f'tile-{index:04d}' / 'probabilities-13x128x128.npy'
                probs = np.load(path, mmap_mode='r', allow_pickle=False)
                if probs.shape != (13, 128, 128) or probs.dtype != np.float32:
                    raise RuntimeError(f'Tile probability geometry differs at {index}')
                y, x = tile['origin_yx_native']
                if y % 8 or x % 8:
                    raise RuntimeError('Tile origin is off comparison grid')
                y //= 8
                x //= 8
                if y + 128 > height or x + 128 > width:
                    raise RuntimeError('Tile exceeds comparison grid')
                totals[y:y+128, x:x+128] += probs[class_index] * weight
                counts[y:y+128, x:x+128] += weight
            totals.flush()
            counts.flush()
            for y in range(0, height, 256):
                end = min(y + 256, height)
                valid = counts[y:end] > 0
                np.divide(totals[y:end], counts[y:end], out=totals[y:end], where=valid)
                totals[y:end][~valid] = 0
            totals.flush()
            positive = int(np.count_nonzero(totals))
            if positive:
                try:
                    threshold = float(threshold_otsu(totals[::2, ::2]))
                except ValueError:
                    threshold = 0.5
            else:
                threshold = 1.0
            candidate = np.memmap(folder / f'class-{class_index+1:02d}.partial.mmap',
                                  mode='w+', dtype=np.uint8, shape=(height, width))
            area = 0
            for y in range(0, height, 256):
                end = min(y + 256, height)
                p = totals[y:end]
                binary = (p > 0) & (p >= threshold)
                candidate[y:end] = binary
                area += int(np.count_nonzero(binary))
            candidate.flush()
            components.append({'class': class_index + 1, 'threshold': threshold,
                               'positive_probability_pixels': positive, 'candidate_pixels': area,
                               'path': str(folder / f'class-{class_index+1:02d}.partial.mmap')})
            del candidate, counts, totals
        native = np.memmap(folder / 'native.partial.mmap', mode='w+', dtype=np.uint8,
                           shape=(height, width))
        native[:] = 0
        for component in sorted(components, key=lambda c: -c['candidate_pixels']):
            binary = np.memmap(component['path'], mode='r', dtype=np.uint8,
                               shape=(height, width))
            for y in range(0, height, 256):
                end = min(y + 256, height)
                view = native[y:end]
                view[binary[y:end] != 0] = component['class']
            del binary
        native.flush()
        shared = np.memmap(folder / 'shared-support.partial.mmap', mode='w+',
                           dtype=np.uint8, shape=(height, width))
        for y in range(0, height, 256):
            end = min(y + 256, height)
            shared[y:end] = np.where(support[y:end] > 0, native[y:end], 0)
        shared.flush()
        native_path = folder / 'native-classes-4um.tif'
        shared_path = folder / 'shared-support-classes-4um.tif'
        tifffile.imwrite(native_path, native, photometric='minisblack',
                         tile=(256, 256), compression='zlib')
        tifffile.imwrite(shared_path, shared, photometric='minisblack',
                         tile=(256, 256), compression='zlib')
        receipt = {'status': 'CAM_VISTA_4UM_MASKS_ASSEMBLED_SOURCE_ONLY',
                   'slide': slide, 'comparison_shape_yx': [height, width],
                   'comparison_spacing_um_xy': [4.0, 4.0], 'selected_tiles': count,
                   'ordered_class_names': source['ordered_class_names'],
                   'class_thresholds_and_areas': components,
                   'native_path': str(native_path), 'native_sha256': sha(native_path),
                   'shared_path': str(shared_path), 'shared_sha256': sha(shared_path),
                   'common_support_sha256': row['support_mask_sha256'],
                   'independent_tile_audit_sha256': sha(audit_path),
                   'assembly_source_sha256': sha(__file__),
                   'assembly_freeze_sha256': sha(freeze_path),
                   'seconds': time.monotonic() - start_time,
                   'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   'test_reference_opened': False, 'online_publication': False,
                   'partial_scratch_pending_independent_audit': True}
        atomic(folder / 'ASSEMBLY_COMPLETE.json', receipt)
        print(json.dumps({'slide': slide, 'native_sha256': receipt['native_sha256'],
                          'shared_sha256': receipt['shared_sha256']}), flush=True)
    except BaseException as error:
        atomic(folder / 'ASSEMBLY_FAILED.json',
               {'status': 'FAILED_LATCHED_PRESERVE_PARTIAL', 'slide': slide,
                'error': repr(error), 'test_reference_opened': False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide', choices=tuple(AUDIT_RUNS), required=True)
    args = parser.parse_args()
    run(args.slide)


if __name__ == '__main__':
    main()
