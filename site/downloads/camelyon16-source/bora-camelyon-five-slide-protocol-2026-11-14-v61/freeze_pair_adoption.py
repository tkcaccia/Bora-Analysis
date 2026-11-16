"""Bind independently audited full-channel CAM pair grids to Bora R cache adoption.

No feature array is copied, no encoder is run, and no test reference is read.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE = Path(__file__).parent
PAIR = ROOT / 'external-benchmark-wsss-camelyon-2026-11-13-v85/PAIR_FEATURES_FROZEN.json'
MODEL = DATA / 'models/Virchow2/3158645804b69e3f3bc4439d4116edddf0840a72'
CAL = {'tumor_093': DATA / 'external-camelyon16-pair-2026-11-13-v49/calibrated/tumor_093',
       'normal_093': DATA / 'external-camelyon16-pair-2026-11-13-v48/calibrated/normal_093'}
SUP = DATA / 'external-camelyon16-pair-2026-11-13-v54/he-support'
AUDIT = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v60'
OUT = HERE / 'source-proofs-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    pair = json.loads(PAIR.read_text())
    if pair['status'] != 'PAIR_FULL1280_THREE_SCALE_FEATURES_FROZEN':
        raise ValueError('Pair features differ')
    model_hash = sha(MODEL / 'model.safetensors')
    config_hash = sha(MODEL / 'config.json')
    if model_hash != '8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b':
        raise ValueError('Virchow2 model differs')
    records = {(r['slide'], int(r['factor'])): r for r in pair['records']}
    images = []
    proof_values = []
    for slide in ('tumor_093', 'normal_093'):
        cal_state_path = CAL[slide] / 'STATE.json'
        cal = json.loads(cal_state_path.read_text())
        image = CAL[slide] / 'calibrated-0p5-padded.tif'
        support = SUP / slide / 'tissue-area.geojson'
        audited = AUDIT / f'{slide.upper()}_TRAINING_LABELS_AUDIT_VERIFIED.json'
        if json.loads(audited.read_text())['status'] != 'VERIFIED':
            raise ValueError('Training label/support audit differs')
        if sha(image) != cal['target_sha256']:
            raise ValueError('Calibrated RGB differs')
        if json.loads(support.read_text())['source_image_sha256'] != cal['target_sha256']:
            raise ValueError('Reviewed source-only support differs')
        scales = {}
        for factor in (1, 2, 4):
            item = records[(slide, factor)]
            if item['source_image_sha256'] != cal['target_sha256'] or item['model_sha256'] != model_hash:
                raise ValueError('Pair grid source/model differs')
            path = Path(item['canonical_path'])
            audit_path = Path(item['pixel_audit_path'])
            if sha(audit_path) != item['pixel_audit_sha256'] or not path.is_file():
                raise ValueError('Independent feature audit differs')
            registration = {'axis_order': 'yxc', 'origin_source_px_yx': [0, 0],
                            'token_stride_source_px_yx': [14 * factor, 14 * factor],
                            'padding': 'right-bottom-to-multiple-896'}
            name = f'{slide}-factor-{factor}-SOURCE_VERIFIED.json'
            proof_values.append((name, {'format': 'bora-precomputed-scale-receipt-v1',
                'status': 'VERIFIED', 'image_sha256': cal['target_sha256'],
                'model_checkpoint_sha256': model_hash, 'model_config_sha256': config_hash,
                'factor': factor, 'features_sha256': item['canonical_sha256'],
                'shape_yxc': item['shape_yxc'], 'registration': registration,
                'encoder_selection': 'full-field', 'encoder_tissue_sha256': None,
                'upstream_pair_manifest_sha256': sha(PAIR),
                'upstream_pixel_audit_sha256': item['pixel_audit_sha256'],
                'source_only_support_sha256': sha(support),
                'training_label_audit_sha256': sha(audited)}))
            scales[str(factor)] = {'path': str(path), 'sha256': item['canonical_sha256'],
                                   'source_receipt_path': str(OUT / name),
                                   'registration': registration}
        images.append({'slide_id': slide, 'image_path': str(image),
                       'image_sha256': cal['target_sha256'],
                       'encoder_selection': 'full-field', 'scales': scales,
                       'reviewed_support_path': str(support),
                       'reviewed_support_sha256': sha(support),
                       'training_mask_audit_sha256': sha(audited)})
    OUT.mkdir()
    for name, value in proof_values:
        write(OUT / name, value)
    for image in images:
        for factor in (1, 2, 4):
            item = image['scales'][str(factor)]
            item['source_receipt_sha256'] = sha(item['source_receipt_path'])
    manifest = OUT / 'PRECOMPUTED_FEATURES_FROZEN.json'
    write(manifest, {'format': 'bora-precomputed-features-v1',
                     'model_checkpoint_sha256': model_hash,
                     'model_config_sha256': config_hash,
                     'images': images,
                     'pair_feature_manifest_sha256': sha(PAIR),
                     'test_reference_opened': False,
                     'online_publication': False})
    print(json.dumps({'manifest': str(manifest), 'sha256': sha(manifest),
                      'six_source_proofs': len(proof_values)}))


if __name__ == '__main__':
    main()
