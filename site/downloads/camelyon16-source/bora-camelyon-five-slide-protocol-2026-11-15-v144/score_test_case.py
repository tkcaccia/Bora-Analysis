"""Local-only, locked binary CAM test scorer on the official known-pixel domain."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import tifffile

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
GATE = ROOT / 'migration/bora-r-camelyon-2026-11-14-v2/LOCAL_CC0_USE_GATE.json'
ENGINE = ROOT / 'migration/bora-camelyon-five-slide-protocol-2026-11-14-v7/binary_score.py'
HERE = Path(__file__).resolve().parent
OUT = DATA / 'external-camelyon16-five-slide-2026-11-15-v144/test-scores'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_array(spec, shape):
    path = Path(spec['path'])
    if digest(path) != spec['sha256']:
        raise ValueError('Frozen mask hash differs: ' + str(path))
    if spec['format'] == 'uint8_raw':
        array = np.memmap(path, mode='r', dtype=np.uint8, shape=shape)
    elif spec['format'] == 'tiff':
        array = tifffile.imread(path)
    else:
        raise ValueError('Unknown frozen raster format')
    if array.shape != shape or array.dtype != np.uint8:
        raise ValueError('Frozen mask shape/dtype differs')
    return array


def main(slide, method):
    frozen_path = HERE / 'SCORING_FROZEN.json'
    frozen = json.loads(frozen_path.read_text())
    gate = json.loads(GATE.read_text())
    assert gate['status'] == 'LOCAL_ANALYSIS_AUTHORIZED_PUBLICATION_HELD'
    assert gate['local_three_slide_scoring_allowed'] and not gate['new_cam_online_publication_allowed']
    assert digest(__file__) == frozen['scorer_source_sha256']
    assert digest(ENGINE) == frozen['metric_engine_sha256']
    assert frozen['test_ids'] == ['tumor_084', 'tumor_091', 'normal_019']
    assert frozen['training_ids'] == ['tumor_093', 'normal_093']
    assert not frozen['test_references_opened_at_freeze'] and not frozen['online_publication']
    row = frozen['slides'][slide]
    item = row['methods'][method]
    shape = tuple(row['shape_yx'])
    support = read_array(row['support'], shape)
    native = read_array(item['native'], shape)
    clipped = read_array(item['shared_support'], shape)
    if np.any(clipped[support == 0] != 0):
        raise ValueError('Shared-support mask predicts outside reviewed support')
    if np.any(native > item['maximum_class_code']) or np.any(clipped > item['maximum_class_code']):
        raise ValueError('Unrecognized method class code')
    binary_native = (native == item['positive_class_code']).astype(np.uint8)
    binary_shared = np.where(support > 0, clipped == item['positive_class_code'], 255).astype(np.uint8)
    ref_receipt_path = Path(row['reference_receipt_path'])
    ref_receipt = json.loads(ref_receipt_path.read_text())
    if ref_receipt['status'] != 'COMPLETE' or ref_receipt['slide'] != slide:
        raise ValueError('Official reference alignment incomplete')
    code_path = Path(ref_receipt['aligned_path'])
    if digest(code_path) != ref_receipt['aligned_sha256']:
        raise ValueError('Official reference aligned hash differs')
    official = np.memmap(code_path, mode='r', dtype=np.uint8, shape=shape)
    if np.any(official > 2):
        raise ValueError('Official reference has unexpected class code')
    reference = (official == 2).astype(np.uint8)
    annotated = (official > 0).astype(np.uint8)
    excluded = np.zeros(shape, dtype=np.uint8)
    spec = importlib.util.spec_from_file_location('frozen_binary_engine', ENGINE)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    case = OUT / method / slide
    if case.exists():
        raise FileExistsError('No in-place CAM scoring retry')
    case.mkdir(parents=True)
    started = time.monotonic()
    try:
        metrics = engine.score_pair(reference, binary_native, binary_shared,
                                    support, annotated, excluded,
                                    spacing_xy_um=(4.0, 4.0),
                                    tolerance_um=5.0, match_iou=.1)
        result = {'status': 'COMPLETE_LOCAL_CC0_NO_PUBLICATION',
                  'slide': slide, 'method': method, 'method_role': item['role'],
                  'metrics': metrics, 'reference_receipt_sha256': digest(ref_receipt_path),
                  'native_mask_sha256': item['native']['sha256'],
                  'shared_support_mask_sha256': item['shared_support']['sha256'],
                  'support_sha256': row['support']['sha256'],
                  'scoring_freeze_sha256': digest(frozen_path),
                  'score_seconds': time.monotonic()-started,
                  'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'online_publication': False}
        tmp = case / 'SCORE_COMPLETE.json.partial'
        tmp.write_text(json.dumps(result,indent=2)+'\n')
        os.replace(tmp, case/'SCORE_COMPLETE.json')
        print(json.dumps({'status':result['status'],'slide':slide,'method':method,
                          'iou':metrics['shared_support']['iou']}),flush=True)
    except BaseException as error:
        (case/'FAILED.json').write_text(json.dumps({'status':'FAILED_LATCHED_PRESERVE_PARTIAL',
            'slide':slide,'method':method,'error':repr(error),'online_publication':False},indent=2)+'\n')
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--slide', choices=('tumor_084','tumor_091','normal_019'), required=True)
    p.add_argument('--method', required=True)
    args = p.parse_args()
    main(args.slide,args.method)
