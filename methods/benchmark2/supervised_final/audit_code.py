#!/usr/bin/env python3
"""Audit selection, physical geometry edge cases and production receipts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import py_compile
import tempfile

import numpy as np
from shapely.geometry import box

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE = Path(__file__).resolve().parent
P30 = ROOT / 'migration/benchmark2-supervised-final-2026-11-11-v30/supervised_wsi_pipeline.py'
P32 = ROOT / 'migration/benchmark2-supervised-final-2026-11-11-v32/repair_unoccupied_gaps.py'
P34 = ROOT / 'migration/benchmark2-supervised-final-2026-11-11-v34/finalize_public_wsi.py'
SEL = ROOT / 'migration/benchmark2-supervised-final-2026-11-11-v30/CLASS_PRESENCE_FROZEN.json'
SEARCH = ROOT / 'migration/benchmark2-supervised-final-2026-11-11-v30/TRAINING_PRESENCE_SEARCH.json'
spec = importlib.util.spec_from_file_location('supervised_pipeline', P30)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def synthetic_checks():
    with tempfile.TemporaryDirectory(prefix='bora-supervised-audit-') as name:
        folder = Path(name)
        mask = np.memmap(folder / 'raw.dat', mode='w+', dtype=np.uint8, shape=(384, 384))
        mask[:] = 0
        mask[10:374, 10:374] = 1
        mask[40:60, 40:60] = 2  # exactly 400 um2 at 1 um/pixel: retained
        mask[80:99, 40:60] = 3  # 380 um2: merged
        mask[150:154, 150:154] = 0  # enclosed hole: filled
        mask.flush()
        support = np.memmap(folder / 'support.dat', mode='w+', dtype=np.uint8, shape=mask.shape)
        support[:] = 0
        support[10:374, 10:374] = 1
        support.flush()
        cleaned, report = pipeline.area_repair(mask, support, folder / 'clean.dat', (1, 1), 400)
        assert cleaned[50, 50] == 2 and cleaned[90, 50] == 1
        assert cleaned[151, 151] == 1 and cleaned[0, 0] == 0
        assert report['positive_parts'] == 1 and report['enclosed_holes'] == 1
        contour = pipeline.smooth_global(cleaned, support, (1, 1),
                                         folder / 'contour.npy', sigma_um=6, block_size=128)
        smoothed = np.load(folder / 'contour.npy', mmap_mode='r')
        assert np.array_equal(np.asarray(smoothed) > 0, np.asarray(cleaned) > 0)
        assert all(row['different_pixels'] == 0 for row in contour['exact_block_seam_checks'])
        scores = np.zeros((8, 10, 10), np.float32)
        scores[0] = 1
        scores[1, 0, 0:2] = 2
        chosen, _ = pipeline.select_classes(scores, [box(0, 0, 140, 140)], .02)
        assert chosen == [1, 2]
        scores[1, 0, 1] = 0
        chosen, _ = pipeline.select_classes(scores, [box(0, 0, 140, 140)], .02)
        assert chosen == [1]
        return {'strict_400um2_boundary': True, 'enclosed_hole_filled': True,
                'exterior_background_preserved': True,
                'six_um_exact_seams': contour['exact_block_seam_checks'],
                'two_percent_presence_boundary': True}


def production(slug, version):
    folder = DATA / f'benchmark2-supervised-final-2026-11-11-{version}' / slug
    receipt_path = folder / 'FULL_SLIDE_VERIFIED.json'
    row = json.loads(receipt_path.read_text())
    assert row['status'] == 'SUPERVISED_BORA_FULL_SLIDE_VECTOR_VERIFIED'
    assert row['independent_geometry_audit']['native_geojson_roundtrip_different_pixels'] == 0
    assert row['independent_geometry_audit']['outside_reviewed_support_pixels'] == 0
    assert all(v['below_2000um2'] == 0 for v in row['independent_geometry_audit']['positive_and_hole_components'])
    assert row['native_vector']['minimum_polygon_um2'] >= 2000
    assert sha(folder / 'supervised-bora.native.geojson') == row['output_sha256']['native_geojson']
    assert sha(folder / 'supervised-bora.mask-0p5um.tif') == row['output_sha256']['mask_tiff']
    return {'slug': slug, 'version': version, 'receipt_sha256': sha(receipt_path),
            'class_ids': row['class_ids'],
            'minimum_polygon_um2': row['native_vector']['minimum_polygon_um2']}


def main():
    output = HERE / 'CODE_AUDIT.json'
    if output.exists():
        raise FileExistsError(output)
    selection = json.loads(SEL.read_text())
    search = json.loads(SEARCH.read_text())
    assert selection['training_search_sha256'] == sha(SEARCH)
    assert selection['minimum_predicted_tissue_fraction'] == .02
    assert selection['development_validation_masks_opened'] == 0
    assert selection['whole_slide_expert_masks_opened'] == 0
    assert len(search['cases']) == 149
    for path in (P30, P32, P34, HERE / 'stage_public_page.py'):
        py_compile.compile(str(path), doraise=True)
    result = {'status': 'SUPERVISED_BORA_CODE_AND_OUTPUT_AUDIT_PASSED',
              'selection_sha256': sha(SEL), 'training_heldout_cases': len(search['cases']),
              'selection_threshold': .02,
              'training_macro_iou_without_gate': search['summary'][0]['mean_case_macro_iou'],
              'training_macro_iou_with_gate': search['summary'][3]['mean_case_macro_iou'],
              'synthetic': synthetic_checks(),
              'production': [production('visium_hd_human_ovarian_cancer_ff', 'v32'),
                             production('visium_hd_human_prostate_cancer_ffpe', 'v34')],
              'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (P30, P32, P34)},
              'pm016_processed': False, 'gpu_used': False}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('CODE_AUDIT_PASSED', len(result['production']), 'production slides')


if __name__ == '__main__':
    main()
