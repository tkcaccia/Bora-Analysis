"""Attach inspectors to every published whole-slide and expression tissue page.

Only browser controls and a checksum-bound index are generated. Saved polygons,
H&E previews, scientific provenance and classification identities are preserved.
Run after rebuilding either benchmark section.
"""
import argparse
import colorsys
import gzip
import hashlib
import html
import json
from pathlib import Path
import re


TISSUES = {
    'visium-hd': {'colon': 'Colorectal cancer', 'pancreas': 'Pancreas',
                  'lymph': 'Lymph node', 'tonsil': 'Tonsil'},
    'visium': {'breast': 'Breast cancer · Giotto', 'colorectal': 'Colorectal cancer · Seurat',
               'tonsil': 'Tonsil · SpatialExperiment'},
}


def include_additional_wsi(site):
    manifest = site / 'visium-hd/ADDITIONAL_WSI_PROVENANCE.json'
    if manifest.exists():
        additions = json.loads(manifest.read_text())
        for slug, record in additions.items():
            if slug in TISSUES['visium-hd']:
                raise ValueError('Additional WSI duplicates a core tissue: ' + slug)
            TISSUES['visium-hd'][slug] = record['title']


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def color(value):
    rgb = colorsys.hsv_to_rgb(((value - 1) * .61803398875) % 1, .68, .9)
    return '#' + ''.join(f'{round(255 * c):02x}' for c in rgb)


def attach(site):
    include_additional_wsi(site)
    records = []
    for section, tissues in TISSUES.items():
        for tissue, title in tissues.items():
            page = site / section / (tissue + '.html')
            relative = (f'{tissue}/BORA_FULL_SLIDE.wsi.geojson' if section == 'visium-hd' and tissue in ('colon','pancreas','lymph','tonsil') else
                        f'{tissue}/{tissue}-Bora-final.geojson.gz' if section == 'visium-hd' else
                        f'fullsection/{tissue}/{tissue}-Visium-Bora-final.geojson.gz')
            geometry = site / section / relative
            stored = geometry.read_bytes()
            payload = gzip.decompress(stored) if geometry.suffix == '.gz' else stored
            vector = json.loads(payload)
            height, width = vector['native_shape_yx']
            bounds = vector['full_slide_input_identity']['analysis_bounds_native_xyxy']
            x0, y0, x1, y1 = bounds
            assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
            label_source = ('Anonymous image-derived KODAMA group' if section == 'visium-hd' else
                            'Anonymous gene-expression group')
            description = (
                'These polygons retain Bora’s numbered image-derived groups. '
                'A fastPLS tissue-name assignment is not available for this saved result.'
                if section == 'visium-hd' else
                'These polygons retain the gene-expression group identifiers refined by Bora. '
                'A fastPLS tissue-name assignment is not available for this saved result. '
                'Output beyond the measured spot footprint is image-only extrapolation.'
            )
            groups = []
            for feature in vector['features']:
                props = feature['properties']
                value = props['value']
                # The existing renders use this exact deterministic group palette.
                assert props['classification'] == f'group_{value}'
                groups.append(dict(value=value, name=f'Group {value:02d}', color=color(value),
                                   area_um2=props.get('area_um2')))
            assert len({g['value'] for g in groups}) == len(groups)
            image = geometry.parent / 'he-overview.jpg'
            record = dict(
                id=f'{section}-{tissue}', title=title, section=section,
                page=page.relative_to(site).as_posix() + '#inspect-geojson',
                geometry=geometry.relative_to(site).as_posix(),
                image=image.relative_to(site).as_posix(), image_sha256=sha256(image.read_bytes()),
                image_bounds_xywh=[x0, y0, x1-x0, y1-y0], native_shape_yx=[height, width],
                native_mpp_xy=vector['native_mpp_xy'],
                source_feature_count=len(groups), geojson_bytes=len(stored),
                stored_geojson_sha256=sha256(stored), decoded_geojson_sha256=sha256(payload),
                label_source=label_source, label_description=description,
                fastpls_tissue_names_available=False, groups=groups,
            )
            records.append(record)
            content = page.read_text()
            # Idempotent so a section rebuild can safely reattach all inspectors.
            content = re.sub(r'<!-- BEGIN GEOJSON INSPECTOR -->.*?<!-- END GEOJSON INSPECTOR -->', '', content, flags=re.S)
            block = ('<!-- BEGIN GEOJSON INSPECTOR -->'
                     f'<section class="geojson-inspector" id="inspect-geojson" data-record="{record["id"]}" '
                     'data-manifest="../geojson-inspectors.json">'
                     '<h2>Inspect the simplified GeoJSON</h2>'
                     f'<p>{html.escape(description)}</p>'
                     '<p>Interactive group inspection is loading. '
                     f'<a href="{relative}" download>Download the saved GeoJSON</a>.</p>'
                     '<noscript>Enable JavaScript for zooming and group inspection.</noscript></section>'
                     '<!-- END GEOJSON INSPECTOR -->')
            anchor = '<h2>Full-slide overview</h2>' if '<h2>Full-slide overview</h2>' in content else '<div class="overview">'
            assert anchor in content, page
            content = content.replace(anchor, block + anchor, 1)
            if '../geojson-inspector.css' not in content:
                content = content.replace('</head>', '<link rel="stylesheet" href="../geojson-inspector.css">'
                                          '<script src="../geojson-inspector.js" defer></script></head>', 1)
            page.write_text(content)
    manifest = dict(format='bora-geojson-inspectors-v1',
                    coordinate_policy='Native image pixel edges; preview uses the exact analysis bounds.',
                    geometry_policy='Saved polygons are displayed without changing coordinates or topology.',
                    public_dataset_images=116, records=records)
    (site / 'geojson-inspectors.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for section in TISSUES:
        page = site / section / 'index.html'
        content = re.sub(r'<!-- BEGIN INSPECTOR LINKS -->.*?<!-- END INSPECTOR LINKS -->', '', page.read_text(), flags=re.S)
        links = ' · '.join(f'<a href="{Path(row["page"]).name}">{html.escape(row["title"])}</a>'
                           for row in records if row['section'] == section)
        block = '<!-- BEGIN INSPECTOR LINKS --><p class="note">Inspect the assigned Bora groups in the interactive GeoJSON viewer on every tissue page: ' + links + '.</p><!-- END INSPECTOR LINKS -->'
        page.write_text(content.replace('<main>', '<main>' + block, 1))
    page = site / 'index.html'
    content = re.sub(r'<!-- BEGIN INSPECTOR LINKS -->.*?<!-- END INSPECTOR LINKS -->', '', page.read_text(), flags=re.S)
    content = content.replace('<h2>Inspect the simplified GeoJSON</h2>', '<h2 id="inspect-geojson">Inspect the simplified GeoJSON</h2>')
    block = '<!-- BEGIN INSPECTOR LINKS --><details><summary>Inspect tissues in the other benchmarks</summary>'
    for section, label in [('visium-hd', 'Visium HD whole slides'), ('visium', 'Visium gene clusters')]:
        links = ' · '.join(f'<a href="{row["page"]}">{html.escape(row["title"])}</a>'
                           for row in records if row['section'] == section)
        block += f'<p><strong>{label}</strong><br>{links}</p>'
    block += '</details><!-- END INSPECTOR LINKS -->'
    anchor = '<h2 id="inspect-geojson">Inspect the simplified GeoJSON</h2>'
    assert anchor in content
    page.write_text(content.replace(anchor, anchor + block, 1))
    print(json.dumps(dict(inspectors=len(records), public_dataset_images=116, saved_geometry_unchanged=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=Path(__file__).resolve().parents[1] / 'site')
    attach(parser.parse_args().site)
