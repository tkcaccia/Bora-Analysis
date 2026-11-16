#!/usr/bin/env python3
"""Independently check the source-only CAM five-ID freeze, without image reads."""
import hashlib
import itertools
import json
import re
from pathlib import Path

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
V18 = ROOT/'bora-camelyon-five-slide-protocol-2026-11-14-v18'
V19 = ROOT/'bora-camelyon-five-slide-protocol-2026-11-14-v19'
HERE = Path(__file__).parent
SALT = 'bora-camelyon16-five-slide-metadata-v3-2026-11-14'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def listed(path, suffix):
    result = {}
    expression = re.compile(r'^\d{4}-\d\d-\d\d\s+\S+\s+(\d+)\s+CAMELYON16/\S+/((?:normal|tumor)_\d{3})' + suffix + r'$')
    for line in path.read_text().splitlines():
        match = expression.fullmatch(line)
        if match:
            assert match[2] not in result
            result[match[2]] = int(match[1])
    return result


def main():
    frozen_path = V19/'TEST_IDS_FROZEN.json'
    frozen = json.loads(frozen_path.read_text())
    gate_path = ROOT/'bora-r-camelyon-2026-11-14-v2/LOCAL_CC0_USE_GATE.json'
    gate = json.loads(gate_path.read_text())
    assert gate['local_cam_specific_training_allowed'] and not gate['new_cam_online_publication_allowed']
    assert frozen['local_gate_sha256'] == digest(gate_path)
    assert frozen['training_ids'] == ['tumor_093', 'normal_093']
    assert frozen['test_ids'] == ['tumor_084', 'tumor_091', 'normal_019']
    assert len(set(frozen['training_ids'] + frozen['test_ids'])) == 5
    assert not frozen['candidate_image_pixels_opened'] and not frozen['expert_reference_pixels_opened']
    assert not frozen['cam_specific_fit_started'] and not frozen['online_publication_allowed']
    source = V18/'source-metadata'
    for key, name in [('official_license_sha256','license.txt'),
                      ('official_readme_sha256','README.md'),
                      ('official_checksums_sha256','checksums.md5')]:
        assert frozen[key] == digest(source/name)
    assert 'available under CC0' in (source/'license.txt').read_text()
    readme = (source/'README.md').read_text()
    excluded = set(re.findall(r'^\* (tumor_\d{3})(?::|\s*$)', readme, re.M))
    assert len(excluded) == 20
    checksums = dict((path, md5) for md5, path in
                     re.findall(r'^([0-9a-f]{32}) \*(\S+)$',
                                (source/'checksums.md5').read_text(), re.M))
    images = listed(V18/'image_listing.txt', r'\.tif')
    masks = listed(V18/'mask_listing.txt', r'_mask\.tif')
    annotations = set(re.findall(r'CAMELYON16/annotations/(tumor_\d{3})\.xml',
                                 (V18/'annotation_listing.txt').read_text()))
    assert len(images) == len(masks) == 270
    candidates = sorted(slide for slide, size in images.items()
                        if (slide in frozen['training_ids'] or size <= 1_000_000_000)
                        and slide in masks and slide not in excluded
                        and (slide.startswith('normal_') or slide in annotations)
                        and f'images/{slide}.tif' in checksums
                        and f'masks/{slide}_mask.tif' in checksums)
    assert candidates == frozen['candidate_ids']
    assert len(candidates) == frozen['candidate_count'] == 11
    rank = {slide: int(hashlib.sha256(f'{SALT}:{slide}'.encode()).hexdigest(), 16)
            for slide in candidates}
    triples = [(left, right, normal)
               for left, right in itertools.combinations(
                   (x for x in candidates if x.startswith('tumor_') and x != 'tumor_093'), 2)
               for normal in candidates if normal.startswith('normal_') and normal != 'normal_093']
    expected = min(triples, key=lambda row: (sum(rank[x] for x in row), row))
    assert list(expected) == frozen['test_ids']
    for item in frozen['selected_official_metadata']:
        slide = item['slide_id']
        assert item['source_object_bytes'] == images[slide]
        assert item['mask_object_bytes'] == masks[slide]
        assert item['source_object_md5'] == checksums[f'images/{slide}.tif']
        assert item['mask_object_md5'] == checksums[f'masks/{slide}_mask.tif']
        assert item['patient_id'] is None and item['center'] is None
        assert item['source_sha256'] is None
    assert frozen['patient_id_status'].startswith('UNAVAILABLE_')
    assert frozen['center_status'].endswith('NOT_VERIFIED')
    assert sum(images[x] for x in expected) == frozen['selected_test_source_object_bytes']
    result = {
        'format': 'bora-camelyon16-source-only-id-freeze-audit-v1',
        'status': 'VERIFIED',
        'frozen_manifest_sha256': digest(frozen_path),
        'training_ids': frozen['training_ids'],
        'test_ids': frozen['test_ids'],
        'candidate_count': len(candidates),
        'selected_source_bytes': frozen['selected_test_source_object_bytes'],
        'selection_recomputed_from_official_listing': True,
        'official_md5_and_object_sizes_matched': True,
        'patient_and_center_relationships': 'UNVERIFIED',
        'pixels_or_reference_masks_opened': False,
        'new_online_publication_allowed': False,
    }
    out = HERE/'SELECTION_AUDIT_VERIFIED.json'
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
