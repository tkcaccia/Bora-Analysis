"""Adopt frozen CAM training-pair full-1280 Virchow2 caches into Bora 0.3.4.

This calls the released package backend and does not run GPU extraction.
"""
import json
from pathlib import Path

from bora.r_api import prepare

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
MANIFEST = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v61/source-proofs-v1/PRECOMPUTED_FEATURES_FROZEN.json'
MODEL = DATA / 'models/Virchow2/3158645804b69e3f3bc4439d4116edddf0840a72'
OUT = DATA / 'external-camelyon16-five-slide-2026-11-14-v62/bora-pair-preparation'


def main():
    spec = json.loads(MANIFEST.read_text())
    if spec['format'] != 'bora-precomputed-features-v1' or [x['slide_id'] for x in spec['images']] != ['tumor_093', 'normal_093']:
        raise ValueError('Frozen source-only training-pair manifest differs')
    images = [x['image_path'] for x in spec['images']]
    masks = [x['reviewed_support_path'] for x in spec['images']]
    result = prepare(images, masks, str(OUT), str(MODEL), [1, 2, 4],
                     precomputed_manifest=str(MANIFEST),
                     minimum_free_gib=100)
    print(json.dumps({'status': result['status'], 'manifest': str(OUT / 'PREPARATION_COMPLETE.json'),
                      'cases': len(result['cases']), 'scales': len(result['scales']),
                      'gpu_launched': False}), flush=True)


if __name__ == '__main__':
    main()
