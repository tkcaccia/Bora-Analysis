"""Independently rebuild every 4-um VISTA pixel from audited source-only tiles."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import tifffile
from skimage.filters import threshold_otsu

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
SOURCE = ROOT / 'migration/bora-camelyon-five-slide-protocol-2026-11-15-v132/VISTA_SOURCE_FROZEN.json'
TILES = DATA / 'external-camelyon16-five-slide-2026-11-15-v136/vista-ranges'
MASKS = DATA / 'external-camelyon16-five-slide-2026-11-15-v141/vista-masks'
RUNS = {'normal_019': 'v137', 'tumor_091': 'v139', 'tumor_084': 'v140'}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def audit(slide):
    freeze_path = ROOT / 'migration/bora-camelyon-five-slide-protocol-2026-11-15-v141/ASSEMBLY_FROZEN.json'
    assembly_path = freeze_path.with_name('assemble_vista.py')
    freeze = json.loads(freeze_path.read_text())
    assert digest(assembly_path) == freeze['assembly_source_sha256']
    assert digest(SOURCE) == freeze['vista_source_sha256']
    source = json.loads(SOURCE.read_text())['slides'][slide]
    run = ROOT / f'migration/bora-camelyon-five-slide-protocol-2026-11-15-{RUNS[slide]}'
    tile_audit_path = run / 'VISTA_RANGES_AUDIT_VERIFIED.json'
    assert digest(tile_audit_path) == freeze['independent_tile_audit_sha256'][slide]
    assert json.loads(tile_audit_path.read_text())['status'] == 'VERIFIED'
    folder = MASKS / slide
    receipt_path = folder / 'ASSEMBLY_COMPLETE.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'CAM_VISTA_4UM_MASKS_ASSEMBLED_SOURCE_ONLY'
    assert receipt['slide'] == slide and not receipt['test_reference_opened']
    assert receipt['assembly_freeze_sha256'] == digest(freeze_path)
    support_receipt_path = Path(source['support_receipt_path'])
    assert digest(support_receipt_path) == source['support_receipt_sha256']
    support_info = json.loads(support_receipt_path.read_text())['support']
    support_path = Path(support_info['support_path'])
    assert digest(support_path) == source['support_mask_sha256']
    shape = tuple(support_info['shape_yx'])
    assert list(shape) == freeze['comparison_shapes_yx'][slide]
    support = np.memmap(support_path, dtype=np.uint8, mode='r', shape=shape)
    output = np.zeros(shape, dtype=np.uint8)
    tiles = source['selected_tiles']
    # The Gaussian is independently sampled from the released 1024-pixel window.
    grid = np.linspace(-1, 1, 1024, dtype=np.float64)
    xx, yy = np.meshgrid(grid, grid)
    window = np.exp(-((xx**2) / (2 * .25**2) + (yy**2) / (2 * .25**2)))
    window /= window.max()
    weight = window.astype(np.float32)[4::8, 4::8]
    assert weight.shape == (128, 128)
    class_areas = {}
    for class_id in range(1, 14):
        numerator = np.zeros(shape, dtype=np.float32)
        denominator = np.zeros(shape, dtype=np.float32)
        for i, tile in enumerate(tiles):
            low = i // 100 * 100
            high = min(low + 100, len(tiles))
            path = TILES / slide / f'range-{low:04d}-{high:04d}' / f'tile-{i:04d}/probabilities-13x128x128.npy'
            values = np.load(path, mmap_mode='r', allow_pickle=False)
            assert values.shape == (13, 128, 128) and values.dtype == np.float32
            y, x = tile['origin_yx_native']
            assert y % 8 == x % 8 == 0
            y //= 8
            x //= 8
            numerator[y:y+128, x:x+128] += values[class_id-1] * weight
            denominator[y:y+128, x:x+128] += weight
        np.divide(numerator, denominator, out=numerator, where=denominator > 0)
        numerator[denominator == 0] = 0
        positive = int(np.count_nonzero(numerator))
        if positive:
            try:
                threshold = float(threshold_otsu(numerator[::2, ::2]))
            except ValueError:
                threshold = .5
        else:
            threshold = 1.0
        candidate = (numerator > 0) & (numerator >= threshold)
        component = receipt['class_thresholds_and_areas'][class_id-1]
        assert component['class'] == class_id
        assert abs(threshold - component['threshold']) < 1e-7
        assert positive == component['positive_probability_pixels']
        assert int(candidate.sum()) == component['candidate_pixels']
        saved = np.memmap(component['path'], dtype=np.uint8, mode='r', shape=shape)
        assert np.array_equal(candidate, saved)
        class_areas[class_id] = int(candidate.sum())
        del saved, candidate, numerator, denominator
    for component in sorted(receipt['class_thresholds_and_areas'], key=lambda v: -v['candidate_pixels']):
        saved = np.memmap(component['path'], dtype=np.uint8, mode='r', shape=shape)
        output[saved != 0] = component['class']
        del saved
    native_path = Path(receipt['native_path'])
    shared_path = Path(receipt['shared_path'])
    assert digest(native_path) == receipt['native_sha256']
    assert digest(shared_path) == receipt['shared_sha256']
    native = tifffile.imread(native_path)
    shared = tifffile.imread(shared_path)
    assert native.dtype == shared.dtype == np.uint8 and native.shape == shared.shape == shape
    assert np.array_equal(native, output)
    assert np.array_equal(shared, np.where(support > 0, output, 0))
    assert not subprocess.check_output(['systemctl', '--user', 'list-units', 'bora-*', '--state=running', '--no-legend', '--no-pager'], text=True).strip()
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'], text=True).strip()
    result = {'status': 'VERIFIED', 'slide': slide, 'shape_yx': list(shape),
              'selected_tiles': len(tiles), 'class_areas': class_areas,
              'native_sha256': digest(native_path), 'shared_sha256': digest(shared_path),
              'all_pixels_reconstructed': True, 'terminal_gpu_contexts': [],
              'source_sha256': digest(SOURCE), 'assembly_receipt_sha256': digest(receipt_path),
              'test_reference_opened': False, 'online_publication': False}
    out = folder / 'MASKS_AUDIT_VERIFIED.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'VERIFIED', 'slide': slide, 'audit_sha256': digest(out)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide', choices=tuple(RUNS), required=True)
    audit(parser.parse_args().slide)
