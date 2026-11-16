"""Independent exact-pixel audit of two source-only CAM Bora test masks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import rasterio


ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE = Path(__file__).resolve().parent
OUTPUT = DATA / 'external-camelyon16-five-slide-2026-11-14-v102/bora-test-masks'
ADMISSION = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v102/ADMISSION_FROZEN.json'
SOURCE = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v97/project_bora_mask.py'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def audit(slide, admission):
    directory = OUTPUT / slide
    receipt_path = directory / 'MASK_COMPLETE.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt['status'] != 'SOURCE_ONLY_BORA_TEST_MASK_COMPLETE' or receipt['slide_id'] != slide:
        raise ValueError('Bora test completion identity differs')
    if receipt['test_reference_opened'] or receipt['online_publication']:
        raise ValueError('Held-out reference/publication gate violated')
    shape = tuple(receipt['output_shape_yx'])
    expected = next(x for x in admission['cases'] if x['slide_id'] == slide)
    if list(shape) != expected['output_shape_yx']:
        raise ValueError('Frozen four-micrometre output grid differs')
    support_path = Path(receipt['support']['support_path'])
    native_path = Path(receipt['native_projection']['mask_path'])
    clipped_path = Path(receipt['support_clipped_projection']['mask_path'])
    for path, record in ((support_path, receipt['support']),
                         (native_path, receipt['native_projection']),
                         (clipped_path, receipt['support_clipped_projection'])):
        if sha(path) != record.get('support_sha256', record.get('mask_sha256')):
            raise ValueError('Projected source/support byte hash differs')
    support = np.memmap(support_path, mode='r', dtype=np.uint8, shape=shape)
    native = np.memmap(native_path, mode='r', dtype=np.uint8, shape=shape)
    clipped = np.memmap(clipped_path, mode='r', dtype=np.uint8, shape=shape)
    if not set(np.unique(native)) <= {1, 2} or \
       not np.array_equal(clipped, np.where(support > 0, native, 0)):
        raise ValueError('Native/shared-support source projection differs')
    final_receipt_path = directory / 'cleanup/FINALIZATION_COMPLETE.json'
    final_receipt = json.loads(final_receipt_path.read_text())
    if final_receipt['status'] != 'COMPLETE' or \
       sha(final_receipt_path) != receipt['finalization_receipt_sha256']:
        raise ValueError('Frozen cleanup/vector finalization receipt differs')
    final_path = Path(final_receipt['final_mask_path'])
    final = np.memmap(final_path, mode='r', dtype=np.uint8, shape=shape)
    if sha(final_path) != final_receipt['final_mask_sha256'] or \
       np.any((final > 0) & (support == 0)) or \
       not set(np.unique(final)) <= {0, 1, 2}:
        raise ValueError('Final tissue classes or support differ')
    exact_vector_raster = directory / 'cleanup/vectors-output-grid/exact/mask.uint8.dat'
    if sha(exact_vector_raster) != sha(final_path):
        raise ValueError('Exact GeoJSON raster roundtrip differs')
    for name, array in (('native_mask', native), ('final_mask', final)):
        item = receipt[name]
        if sha(Path(item['path'])) != item['sha256']:
            raise ValueError('Published source-only TIFF SHA256 differs')
        with rasterio.open(item['path']) as dataset:
            if dataset.shape != shape or not np.array_equal(dataset.read(1), array):
                raise ValueError('Published source-only TIFF pixel differs')
    for class_id in (1, 2):
        item = receipt['binary_images'][str(class_id)]
        if sha(Path(item['path'])) != item['sha256']:
            raise ValueError('Binary class TIFF SHA256 differs')
        with rasterio.open(item['path']) as dataset:
            expected_bw = np.where(final == class_id, 255, 0).astype(np.uint8)
            if not np.array_equal(dataset.read(1), expected_bw):
                raise ValueError('Binary class TIFF pixel differs')
    if sha(directory / 'class-mask.geojson') != receipt['source_vector']['source_geojson_sha256'] or \
       sha(Path(receipt['preview']['path'])) != receipt['preview']['sha256']:
        raise ValueError('Native-coordinate vector or preview SHA256 differs')
    vector_header = json.loads((directory / 'class-mask.geojson').read_text())
    if vector_header['coordinate_space'] != 'calibrated_source_image_pixel_edges':
        raise ValueError('GeoJSON coordinate space differs')
    return {'slide_id': slide, 'mask_receipt_sha256': sha(receipt_path),
            'output_shape_yx': list(shape), 'support_pixels': int(np.count_nonzero(support)),
            'final_class_pixels': {str(k): int(np.count_nonzero(final == k))
                                   for k in (0, 1, 2)},
            'native_classes': np.unique(native).tolist(),
            'final_classes': np.unique(final).tolist(),
            'exact_tiff_bw_support_vector_pixels': True,
            'preview_sha256': receipt['preview']['sha256']}


def main():
    if sha(SOURCE) != json.loads(ADMISSION.read_text())['source_sha256']:
        raise ValueError('Frozen source-only mask adapter changed')
    complete = json.loads((OUTPUT / 'CONTROLLER_COMPLETE.json').read_text())
    if complete['status'] != 'COMPLETE' or complete['completed'] != ['tumor_084', 'normal_019']:
        raise ValueError('Serial source-only controller incomplete')
    service = subprocess.check_output(
        ['systemctl', '--user', 'show', 'bora-cam16-bora-masks-remaining-v102.service',
         '-p', 'ActiveState', '-p', 'MainPID', '-p', 'Result',
         '-p', 'ExecMainStatus', '-p', 'ControlGroup'], text=True)
    state = dict(row.split('=', 1) for row in service.strip().splitlines())
    if state != {'MainPID': '0', 'Result': 'success', 'ExecMainStatus': '0',
                 'ControlGroup': '', 'ActiveState': 'inactive'}:
        raise RuntimeError('Owned process group did not terminate cleanly')
    output = {'status': 'VERIFIED', 'format': 'bora-cam-remaining-source-only-mask-audit-v1',
              'admission_sha256': sha(ADMISSION),
              'controller_sha256': sha(OUTPUT / 'CONTROLLER_COMPLETE.json'),
              'cases': [audit(slide, json.loads(ADMISSION.read_text()))
                        for slide in ('tumor_084', 'normal_019')],
              'terminal_service': state, 'test_reference_opened': False,
              'online_publication': False}
    target = HERE / 'REMAINING_BORA_MASKS_AUDIT_VERIFIED.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(output, indent=2) + '\n')
    print(target)


if __name__ == '__main__':
    main()
