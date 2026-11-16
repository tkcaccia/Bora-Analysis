"""Independent source/shape/receipt audit of CAM pair Bora cache adoption."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
OUT = DATA / 'external-camelyon16-five-slide-2026-11-14-v62/bora-pair-preparation'
SPEC = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v61/source-proofs-v1/PRECOMPUTED_FEATURES_FROZEN.json'
PAIR = ROOT / 'external-benchmark-wsss-camelyon-2026-11-13-v85/PAIR_FEATURES_FROZEN.json'
HERE = Path(__file__).parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    target = HERE / 'PAIR_ADOPTION_AUDIT_VERIFIED.json'
    if target.exists():
        raise FileExistsError(target)
    spec = json.loads(SPEC.read_text())
    pair = json.loads(PAIR.read_text())
    prep_path = OUT / 'PREPARATION_COMPLETE.json'
    prep = json.loads(prep_path.read_text())
    input_plan = json.loads((OUT / 'PREPARATION_INPUTS.json').read_text())
    admission = json.loads((OUT / 'RESOURCE_ADMISSION.json').read_text())
    if prep['status'] != 'COMPLETE' or input_plan['status'] != 'COMPLETE' or not prep['cache_adoption_not_new_extraction']:
        raise ValueError('Adoption did not finish without extraction')
    if input_plan['precomputed_manifest_sha256'] != sha(SPEC) or admission['new_feature_bytes'] != 0 or admission['gpu_workers_launched'] != 0:
        raise ValueError('Adoption source or resource claim differs')
    if len(prep['cases']) != 2 or len(prep['scales']) != 6:
        raise ValueError('Pair or scale count differs')
    source_records = {(x['slide'], x['factor']): x for x in pair['records']}
    rows = []
    for i, slide in enumerate(('tumor_093', 'normal_093')):
        image_spec = spec['images'][i]
        case = prep['cases'][i]
        source = case['source']
        clipped = json.loads((DATA / f'external-camelyon16-five-slide-2026-11-14-v58/training-labels/{slide}/STATE.json').read_text())
        support_pixels = sum(clipped['class_counts_inside_support_0_1_2'])
        if (source['image_sha256'] != image_spec['image_sha256'] or
                source['tissue_support_sha256'] != image_spec['reviewed_support_sha256'] or
                source['selected_source_pixels'] != support_pixels or
                source['model_checkpoint_sha256'] != spec['model_checkpoint_sha256'] or
                source['model_config_sha256'] != spec['model_config_sha256'] or
                source['historical_encoder_selection'] != 'full-field' or
                source['new_encoder_job']):
            raise ValueError(f'{slide} image/support/model identity differs')
        for factor in (1, 2, 4):
            item = next(x for x in prep['scales'] if x['case_dir'] == case['case_dir'] and x['factor'] == factor)
            receipt_path = Path(item['receipt_path'])
            if sha(receipt_path) != item['receipt_sha256']:
                raise ValueError('Adopted scale receipt hash differs')
            receipt = json.loads(receipt_path.read_text())
            old = source_records[(slide, factor)]
            array = np.load(receipt['features_path'], mmap_mode='r', allow_pickle=False)
            if (receipt['features_path'] != old['canonical_path'] or
                    receipt['features_sha256'] != old['canonical_sha256'] or
                    list(array.shape) != old['shape_yxc'] or array.dtype != np.float16 or
                    receipt['encoder_selection'] != 'full-field'):
                raise ValueError(f'{slide} factor {factor} array identity differs')
        rows.append({'slide': slide, 'image_sha256': source['image_sha256'],
                     'support_sha256': source['tissue_support_sha256'],
                     'support_pixels': support_pixels,
                     'eligible_tiles': source['eligible_tiles']})
    service = subprocess.run(['systemctl', '--user', 'show', 'bora-cam16-pair-adoption-v62.service',
                              '-p', 'ActiveState', '-p', 'MainPID', '-p', 'ControlGroup',
                              '-p', 'Result', '-p', 'ExecMainStatus'],
                             check=True, capture_output=True, text=True).stdout
    if 'ActiveState=inactive' not in service or 'MainPID=0' not in service or 'ControlGroup=\n' not in service or 'Result=success' not in service:
        raise ValueError('Adoption owned process not terminally clean')
    result = {'status': 'VERIFIED', 'format': 'bora-cam-pair-adoption-audit-v1',
              'preparation_path': str(prep_path), 'preparation_sha256': sha(prep_path),
              'source_manifest_sha256': sha(SPEC), 'cases': rows,
              'scales': 6, 'copied_feature_bytes': 0, 'gpu_launched': False,
              'service': service.strip().splitlines(),
              'test_reference_opened': False, 'online_publication': False}
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(target)


if __name__ == '__main__':
    main()
