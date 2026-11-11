#!/usr/bin/env python3
"""Independent receipt and file-hash audit of the public supervised WSI release."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from bs4 import BeautifulSoup

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE = Path(__file__).resolve().parent
CAMPAIGN = ROOT / 'migration/benchmark2-unified-wsi-panels-2026-11-11-v17/CAMPAIGN_VERIFIED.json'
V33 = ROOT / 'migration/benchmark2-supervised-final-2026-11-11-v33'
V34 = ROOT / 'migration/benchmark2-supervised-final-2026-11-11-v34'
PRESENCE = ROOT / 'migration/benchmark2-supervised-final-2026-11-11-v30/CLASS_PRESENCE_FROZEN.json'
SITE = Path('/tmp/bora-analysis-bw106-v2/site/visium-hd')


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def audit_slide(slug: str, page_name: str) -> dict:
    ovarian = slug == 'visium_hd_human_ovarian_cancer_ff'
    version = 'v32' if ovarian else 'v34'
    final_dir = DATA / f'benchmark2-supervised-final-2026-11-11-{version}' / slug
    final_path = final_dir / 'FULL_SLIDE_VERIFIED.json'
    final = read(final_path)
    render_path = (DATA / 'benchmark2-supervised-final-2026-11-11-v33/rendered' / slug /
                   'RENDER_VERIFIED.json') if ovarian else (
                   DATA / 'benchmark2-supervised-final-2026-11-11-v34/rendered' / slug /
                   'RENDER_VERIFIED.json')
    render = read(render_path)
    public_path = (V33 / 'OVARIAN_PUBLICATION_VERIFIED.json') if ovarian else (
                   HERE / f'PUBLICATION_{slug}.json')
    public = read(public_path)
    assert final['status'] == 'SUPERVISED_BORA_FULL_SLIDE_VECTOR_VERIFIED'
    assert render['status'] == 'TEN_VIEWS_FROM_ONE_FINAL_GLOBAL_SUPERVISED_MASK_VERIFIED'
    assert render['full_slide_receipt_sha256'] == sha(final_path)
    assert len(render['views']) == 10 and len(public['assets']) == 12
    assert public['status'].endswith('PUBLICATION_VERIFIED')
    assert not final['gpu_used'] and not final['pm016_included']
    assert final['whole_slide_expert_accuracy'] is None
    assert final['training_presence_sha256'] == sha(PRESENCE)
    if ovarian:
        source_contour = (DATA / 'benchmark2-supervised-final-2026-11-11-v30' /
                          slug / 'contour-6um.npy')
        assert final['contour_smoothing_source_sha256'] == sha(source_contour)
    else:
        assert final['contour_smoothing']['sigma_um'] == 6
    assert final['independent_geometry_audit']['physical_minimum_um2'] == 2000
    assert final['independent_geometry_audit']['outside_reviewed_support_pixels'] == 0
    assert final['independent_geometry_audit']['native_geojson_roundtrip_different_pixels'] == 0
    assert all(row['below_2000um2'] == 0 for row in
               final['independent_geometry_audit']['positive_and_hole_components'])
    assert final['native_vector']['minimum_polygon_um2'] >= 2000
    assert set(final['class_ids']) == set(final['class_selection']['retained_class_ids'])
    assert final['class_selection']['minimum_fraction'] == .02
    assert sha(final_dir / 'supervised-bora.mask-0p5um.tif') == final['output_sha256']['mask_tiff']
    assert sha(final_dir / 'supervised-bora.native.geojson') == final['output_sha256']['native_geojson']
    asset_hashes = {row['name']: row['sha256'] for row in public['assets']}
    for view in render['views']:
        name = view['name'] + '-supervised.webp'
        assert asset_hashes[name] == view['asset_sha256']
        assert sha(render_path.parent / name) == view['asset_sha256']
    page_path = SITE / (page_name + '.html')
    assert sha(page_path) == public['page_sha256']
    html = BeautifulSoup(page_path.read_text(), 'html.parser')
    sections = [html.find('section', id='overview')] + html.select('section.unified-zoom')
    assert len(sections) == 10 and all(section is not None for section in sections)
    expected_methods = {'he', 'bora', 'supervised', 'unet', 'deeplabv3plus', 'medsam', 'vista_path'}
    view_rows = {row['name']: row for row in render['views']}
    for section in sections:
        figures = section.select('figure[data-method]')
        assert len(figures) == 7
        assert {figure['data-method'] for figure in figures} == expected_methods
        supervised = section.find('figure', attrs={'data-method': 'supervised'})
        image = supervised.find('img')
        assert image and image['src'].endswith(section['id'] + '-supervised.webp')
        assert '/benchmark2-supervised-final-2026-11-11-v1/' in image['src']
        assert len(supervised.select('.class-swatches li')) == len(view_rows[section['id']]['class_ids_on_support'])
    return {'slug': slug, 'class_ids': final['class_ids'],
            'final_receipt_sha256': sha(final_path), 'render_receipt_sha256': sha(render_path),
            'publication_receipt_sha256': sha(public_path), 'page_url': public['page_url'],
            'minimum_polygon_um2': final['native_vector']['minimum_polygon_um2']}


def main() -> None:
    pages = read(CAMPAIGN)['pages']
    slugs = [row['slug'] for row in pages]
    assert len(slugs) == 13 and len(set(slugs)) == 13
    assert all('pm016' not in slug.lower() and 'beetle' not in slug.lower() for slug in slugs)
    controller = read(V34 / 'CAMPAIGN_LOCAL_VERIFIED.json')
    assert controller['status'] == 'TWELVE_PUBLIC_SUPERVISED_GLOBAL_MASKS_AND_VECTORS_VERIFIED'
    assert controller['count'] == 12 and controller['views'] == 120
    assert {r['slug'] for r in controller['slides']} == set(slugs) - {'visium_hd_human_ovarian_cancer_ff'}
    rows = [audit_slide(row['slug'], row['page']) for row in pages]
    result = {'status': 'THIRTEEN_PUBLIC_SUPERVISED_FINAL_OUTPUTS_AND_PAGES_AUDITED',
              'slides': rows, 'slide_count': 13, 'render_count': 130, 'published_assets': 156,
              'training_presence_sha256': sha(PRESENCE), 'controller_receipt_sha256': sha(V34 / 'CAMPAIGN_LOCAL_VERIFIED.json'),
              'pm016_processed': False, 'beetle_processed': False, 'gpu_used': False,
              'whole_slide_expert_accuracy': None}
    output = HERE / 'CAMPAIGN_PUBLICATION_AUDITED.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'], result['slide_count'], result['render_count'], result['published_assets'])


if __name__ == '__main__':
    main()
