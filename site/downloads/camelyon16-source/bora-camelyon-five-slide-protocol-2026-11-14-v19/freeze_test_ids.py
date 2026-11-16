#!/usr/bin/env python3
"""Freeze CAM test IDs from official S3 metadata; never open pixels or masks."""

import hashlib
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
BASE = ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-14-v18'
OUT = Path(__file__).parent
GATE = ROOT/'migration/bora-r-camelyon-2026-11-14-v2/LOCAL_CC0_USE_GATE.json'
SELECTOR = ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-14-v9/select_metadata_only.py'
MAX_TEST_SOURCE_BYTES = 1_000_000_000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def listing(path, extension):
    items = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r'\d{4}-\d\d-\d\d\s+\S+\s+(\d+)\s+CAMELYON16/\S+/((?:normal|tumor)_\d{3})'+extension,line)
        if match:
            slide = match[2]
            if slide in items:
                raise ValueError(f'Duplicate listing {slide}')
            items[slide] = int(match[1])
    return items


def main():
    target = OUT/'TEST_IDS_FROZEN.json'
    if target.exists():
        raise FileExistsError(target)
    gate = json.loads(GATE.read_text())
    if gate['status'] != 'LOCAL_ANALYSIS_AUTHORIZED_PUBLICATION_HELD' or \
       not gate['local_cam_specific_training_allowed'] or \
       gate['new_cam_online_publication_allowed']:
        raise PermissionError('Exact local-only CC0 gate required')
    source_dir = BASE/'source-metadata'
    readme = source_dir/'README.md'
    license_file = source_dir/'license.txt'
    checksums = source_dir/'checksums.md5'
    text = readme.read_text()
    assert 'CAMELYON16 data set is available under CC0.' in license_file.read_text()
    assert 'value 1 for normal tissue, and 2 for tumor' in text
    excluded = set(re.findall(r'^\* (tumor_\d{3})(?::|\s*$)',text,re.M))
    assert len(excluded) == 20
    md5 = {}
    for line in checksums.read_text().splitlines():
        match = re.fullmatch(r'([0-9a-f]{32}) \*(\S+)',line)
        if match:
            md5[match[2]] = match[1]
    assert hashlib.md5(readme.read_bytes()).hexdigest() == md5['README.md']
    assert hashlib.md5(license_file.read_bytes()).hexdigest() == md5['license.txt']
    images = listing(BASE/'image_listing.txt',r'\.tif')
    masks = listing(BASE/'mask_listing.txt',r'_mask\.tif')
    annotation_text = (BASE/'annotation_listing.txt').read_text()
    annotations = set(re.findall(r'CAMELYON16/annotations/(tumor_\d{3})\.xml',annotation_text))
    assert len(images) == len(masks) == 270
    candidates = []
    for slide, size in sorted(images.items()):
        if slide not in ('normal_093','tumor_093') and size > MAX_TEST_SOURCE_BYTES:
            continue
        if slide not in masks or slide in excluded or \
           (slide.startswith('tumor_') and slide not in annotations):
            continue
        image_key = f'images/{slide}.tif'
        mask_key = f'masks/{slide}_mask.tif'
        if image_key not in md5 or mask_key not in md5:
            continue
        candidates.append({
            'slide_id':slide,
            'reference_kind':'lesion_mask' if slide.startswith('tumor_') else 'negative_slide_label',
            'partition':'public_development',
            'official_listing_verified':True,
            'patient_id':None,'center':None,'source_sha256':None,
            'source_object_bytes':size,
            'source_object_md5':md5[image_key],
            'mask_object_bytes':masks[slide],
            'mask_object_md5':md5[mask_key],
            'source_url':f'https://camelyon-dataset.s3.amazonaws.com/CAMELYON16/{image_key}',
            'mask_url':f'https://camelyon-dataset.s3.amazonaws.com/CAMELYON16/{mask_key}',
            'positive_annotation_object_listed':slide in annotations,
            'official_exhaustive_annotation_exception':slide in excluded,
        })
    spec = importlib.util.spec_from_file_location('frozen_selector',SELECTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    input_meta = {'format':'bora-camelyon16-official-metadata-v1','slides':candidates}
    proposal = module.propose(input_meta)
    chosen = proposal['test_slide_ids']
    # This assertion makes any upstream metadata or selector change fail closed.
    assert chosen == ['tumor_084','tumor_091','normal_019']
    selected = [next(item for item in candidates if item['slide_id']==slide)
                for slide in ('tumor_093','normal_093',*chosen)]
    assert sum(item['source_object_bytes'] for item in selected[2:]) <= 3*MAX_TEST_SOURCE_BYTES
    result = {
        'format':'bora-camelyon16-local-five-slide-id-freeze-v1',
        'status':'TEST_IDS_FROZEN_SOURCE_SHA_AND_PATIENT_CHECK_PENDING',
        'training_ids':['tumor_093','normal_093'],
        'test_ids':chosen,
        'selection_rule':'public development slides with listed official mask, exhaustive positive annotation, source object <=1,000,000,000 bytes; deterministic v9 SHA256 rank; no image or reference content',
        'size_limit_test_source_bytes':MAX_TEST_SOURCE_BYTES,
        'candidate_count':len(candidates),
        'candidate_ids':[item['slide_id'] for item in candidates],
        'selected_official_metadata':selected,
        'selected_test_source_object_bytes':sum(item['source_object_bytes'] for item in selected[2:]),
        'patient_id_status':'UNAVAILABLE_IN_OFFICIAL_S3_LISTING; DO_NOT_CLAIM_PATIENT_DISJOINT',
        'center_status':'TWO_CENTERS_KNOWN; SLIDE_LEVEL_MAPPING_NOT_VERIFIED',
        'official_binary_label_contract':'mask 0 unscored, 1 normal tissue, 2 tumor; XML Tumor and Exclusion groups',
        'expert_reference_pixels_opened':False,
        'candidate_image_pixels_opened':False,
        'cam_specific_fit_started':False,
        'online_publication_allowed':False,
        'local_gate_sha256':sha(GATE),
        'selector_sha256':sha(SELECTOR),
        'official_license_sha256':sha(license_file),
        'official_readme_sha256':sha(readme),
        'official_checksums_sha256':sha(checksums),
        'image_listing_sha256':sha(BASE/'image_listing.txt'),
        'mask_listing_sha256':sha(BASE/'mask_listing.txt'),
        'annotation_listing_sha256':sha(BASE/'annotation_listing.txt'),
    }
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'training':result['training_ids'],'test':chosen,
                      'source_bytes':result['selected_test_source_object_bytes']}))


if __name__=='__main__':
    main()
