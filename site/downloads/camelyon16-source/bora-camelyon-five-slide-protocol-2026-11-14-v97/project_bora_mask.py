"""Source-only CAM test mask/GeoJSON from audited sparse Bora score grids.

Only released Bora projection/cleanup functions are used after sparse scoring.
The CAM expert test reference paths do not occur in this source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

from bora.r_prediction import (load_model, output_shape, output_support,
    project_mask, finalize_projected_mask, source_coordinate_geojson,
    _write_mask_tiff, _preview)


ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
FEATURES = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v41/TEST_FEATURES_FROZEN.json'
MODEL = DATA / 'external-camelyon16-five-slide-2026-11-14-v69/bora-pair-binary-head'
SCORE_PATHS = {
    'normal_019': DATA / 'external-camelyon16-five-slide-2026-11-14-v93/bora-sparse-scores/normal_019',
    'tumor_084': DATA / 'external-camelyon16-five-slide-2026-11-14-v95/bora-sparse-scores/tumor_084',
    'tumor_091': DATA / 'external-camelyon16-five-slide-2026-11-14-v95/bora-sparse-scores/tumor_091',
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run(slide: str, output: Path) -> dict:
    begun = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(FEATURES.read_text())
    case = next(x for x in manifest['cases'] if x['slide_id'] == slide)
    scale = next(x for x in case['scales'] if x['factor'] == 1)
    spec = json.loads(Path(scale['source_spec_path']).read_text())
    score_path = SCORE_PATHS[slide] / 'SCORES_COMPLETE.json'
    score = json.loads(score_path.read_text())
    model = load_model(MODEL)
    if (score['status'] != 'SPARSE_BORA_SCORES_COMPLETE' or
            score['slide_id'] != slide or score['image_sha256'] != case['image_sha256'] or
            score['support_sha256'] != case['support_sha256'] or
            score['model_sha256'] != model['model_sha256'] or
            score['classes'] != [1, 2] or
            sha(Path(score['smoothed_scores_path'])) != score['smoothed_scores_sha256']):
        raise ValueError('Audited source/model scores differ')
    image = Path(spec['image'])
    tissue = Path(spec['tissue_support'])
    if (sha(image) != case['image_sha256'] or sha(tissue) != case['support_sha256'] or
            tuple(spec['image_shape_yx']) != tuple(spec['padded_shape_yx'])):
        raise ValueError('Calibrated H&E or frozen source-only tissue support differs')
    shape_yx = output_shape(tuple(spec['image_shape_yx']), (.5, .5), (4., 4.))
    pixels = shape_yx[0] * shape_yx[1]
    if pixels > 150_000_000:
        raise MemoryError('Four-micrometre output grid above bounded source-only adapter')
    output.mkdir(parents=True)
    try:
        source = {'image_shape_yx': spec['image_shape_yx'],
                  'tissue_support': str(tissue),
                  'tissue_support_sha256': case['support_sha256']}
        support = output_support(source, shape_yx, (8., 8.),
                                 output / 'output-support.uint8.dat')
        mask_support = np.memmap(support['support_path'], mode='r',
                                 dtype=np.uint8, shape=shape_yx)
        smooth = np.load(score['smoothed_scores_path'], mmap_mode='r', allow_pickle=False)
        if tuple(smooth.shape) != tuple(score['score_shape_class_yx']):
            raise ValueError('Registered source-only score grid differs')
        native = project_mask(smooth, model['classes'], shape_yx, (8., 8.),
                              output / 'native-mask.uint8.dat')
        clipped = project_mask(smooth, model['classes'], shape_yx, (8., 8.),
                               output / 'support-clipped-mask.uint8.dat',
                               support=mask_support)
        final = finalize_projected_mask(
            Path(clipped['mask_path']), Path(support['support_path']),
            shape_yx, (4., 4.), [1, 2], case['image_sha256'],
            model['model_sha256'], output / 'cleanup',
            minimum_area_um2=2000., contour_sigma_um=6.,
            class_names={'1': 'non-metastatic tissue', '2': 'metastasis'})
        final_mask = np.memmap(final['final_mask_path'], mode='r',
                               dtype=np.uint8, shape=shape_yx)
        source_vector = source_coordinate_geojson(
            output / 'cleanup/vectors-output-grid/exact/mask.geojson',
            output / 'class-mask.geojson', tuple(spec['image_shape_yx']),
            (.5, .5), (4., 4.))
        native_tiff = _write_mask_tiff(
            np.memmap(native['mask_path'], mode='r', dtype=np.uint8, shape=shape_yx),
            output / 'native-mask.tif')
        final_tiff = _write_mask_tiff(final_mask, output / 'class-mask.tif')
        bw = {str(k): _write_mask_tiff(final_mask,
                output / f'bw-class-{k:03d}.tif', equals=k) for k in (1, 2)}
        preview = _preview(image, Path(final_tiff['path']), [1, 2],
                           output / 'preview.png', backend='vips-lanczos3')
        result = {'status': 'SOURCE_ONLY_BORA_TEST_MASK_COMPLETE', 'slide_id': slide,
                  'image_sha256': case['image_sha256'],
                  'source_support_sha256': case['support_sha256'],
                  'score_receipt_sha256': sha(score_path),
                  'model_sha256': model['model_sha256'],
                  'output_mpp_xy': [4., 4.], 'output_shape_yx': list(shape_yx),
                  'classes': [1, 2], 'support': support,
                  'native_projection': native, 'support_clipped_projection': clipped,
                  'finalization_receipt_sha256': sha(output / 'cleanup/FINALIZATION_COMPLETE.json'),
                  'source_vector': source_vector,
                  'native_mask': native_tiff, 'final_mask': final_tiff,
                  'binary_images': bw, 'preview': preview,
                  'wall_seconds': time.monotonic() - begun,
                  'peak_host_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'test_reference_opened': False, 'online_publication': False}
        atomic(output / 'MASK_COMPLETE.json', result)
        return result
    except BaseException as error:
        atomic(output / 'FAILED.json', {'status': 'FAILED_LATCHED',
               'slide_id': slide, 'error': repr(error),
               'partial_bytes_preserved': True,
               'test_reference_opened': False, 'online_publication': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide', choices=sorted(SCORE_PATHS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.slide, args.output)), flush=True)
