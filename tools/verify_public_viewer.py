"""Validate the complete static viewer without scientific or GPU dependencies."""
import argparse
from collections import Counter
import colorsys
import csv
import gzip
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


def checked_path(base, relative, root):
    if urlsplit(relative).scheme or relative.startswith('/'):
        raise ValueError('Expected a relative asset URL: ' + relative)
    p = (base / unquote(relative)).resolve()
    if not p.is_relative_to(root) or not p.is_file():
        raise ValueError('Missing or escaping asset: ' + relative)
    return p


def ring_area(ring):
    if len(ring) < 4 or ring[0] != ring[-1]:
        raise ValueError('Invalid polygon ring')
    return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring, ring[1:]))) / 2


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.nodes = []

    def handle_starttag(self, tag, attrs):
        self.nodes.append((tag, dict(attrs)))
        for key, value in attrs:
            if key in ('src', 'href', 'data-src') and value:
                self.links.append(value)


def verify_visium(site, sections):
    base = site / 'visium'
    manifest = checked_path(base, sections['visium_fullsection_provenance'], site)
    records = json.loads(manifest.read_text())
    assert set(records) == {'breast_giotto', 'colorectal_seurat', 'tonsil_spatialexperiment'}
    for sample, record in records.items():
        assert record['crop_count'] == 9
        assert record['minimum_polygon_and_hole_um2'] == 400
        for rel, digest in record['assets_sha256'].items():
            asset = checked_path(base, rel, site)
            assert hashlib.sha256(asset.read_bytes()).hexdigest() == digest, asset
        archive = checked_path(base, record['download'], site)
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == record['compressed_sha256']
        with gzip.open(archive, 'rb') as stream:
            payload = stream.read()
        assert len(payload) == record['geojson_bytes']
        assert hashlib.sha256(payload).hexdigest() == record['global_geojson_sha256']
        vector = json.loads(payload)
        del payload
        assert vector['type'] == 'FeatureCollection'
        assert vector['minimum_component_and_hole_um2'] == 400
        assert len(vector['features']) == record['group_count']
        area_scale = math.prod(record['native_mpp_xy'])
        for feature in vector['features']:
            geometry = feature['geometry']
            assert geometry['type'] in ('Polygon', 'MultiPolygon')
            polys = [geometry['coordinates']] if geometry['type'] == 'Polygon' else geometry['coordinates']
            for polygon in polys:
                areas = [ring_area(ring) * area_scale for ring in polygon]
                assert areas[0] - sum(areas[1:]) >= 400 - 1e-4, sample
                assert all(a >= 400 - 1e-4 for a in areas[1:]), sample
    return len(records)


def verify_additional_wsi(site):
    base = site / 'visium-hd'
    manifest = base / 'ADDITIONAL_WSI_PROVENANCE.json'
    if not manifest.exists():
        assert 'ADDITIONAL_WSI_2026_10_29' not in (base / 'index.html').read_text()
        return 0
    records = json.loads(manifest.read_text())
    assert records and all(isinstance(slug, str) and slug for slug in records)
    for sample, record in records.items():
        assert record['confirmed_visium_hd'] is False
        assert record['expert_masks_available'] is False
        assert record['development_demonstration'] is True
        minimum_area = record['minimum_polygon_and_hole_um2']
        assert minimum_area in (400, 1000, 2000)
        page = (base / (sample + '.html')).read_text()
        assert 'Visium HD acquisition has not been established' in page
        for kind in ('wide', 'large', 'field'):
            for index in range(1, 4):
                assert f'{kind}-{index:02d}/output-mask-with-geojson.jpg' in record['assets_sha256']
        for rel, digest in record['assets_sha256'].items():
            asset = checked_path(base / sample, rel, site)
            assert hashlib.sha256(asset.read_bytes()).hexdigest() == digest, asset
        archive = checked_path(base / sample, sample + '-Bora-final.geojson.gz', site)
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == record['compressed_sha256']
        with gzip.open(archive, 'rb') as stream:
            payload = stream.read()
        assert len(payload) == record['decoded_bytes']
        assert hashlib.sha256(payload).hexdigest() == record['global_geojson_sha256']
        vector = json.loads(payload)
        assert vector['type'] == 'FeatureCollection'
        assert vector['minimum_component_and_hole_um2'] == minimum_area
        assert 0 < len(vector['features']) <= record['groups']
        area_scale = math.prod(record['source_mpp_xy'])
        for feature in vector['features']:
            geometry = feature['geometry']
            assert geometry['type'] in ('Polygon', 'MultiPolygon')
            polygons = [geometry['coordinates']] if geometry['type'] == 'Polygon' else geometry['coordinates']
            for polygon in polygons:
                areas = [ring_area(ring) * area_scale for ring in polygon]
                assert areas[0] - sum(areas[1:]) >= minimum_area - 1e-3, sample
                assert all(area >= minimum_area - 1e-3 for area in areas[1:]), sample
    return len(records)


def verify_spatial_expression(site):
    expected = {'xenium-breast-2fov': ('xenium-03', 7275, 7273, 280, 6),
                'xenium-ovary-multimodal-tiny': ('xenium-08', 623, 623, 403, 4),
                'xenium-ovary-nuclear-expansion-tiny': ('xenium-09', 632, 632, 403, 4)}
    for slug, (dataset_id, native, admitted, genes, groups) in expected.items():
        base = site / 'spatial-expression' / slug
        record = json.loads((base / 'PROVENANCE.json').read_text())
        assert record['dataset_id'] == dataset_id
        assert record['platform'] == 'Xenium'
        assert record['native_cells'] == native
        assert record['analyzed_cells'] == admitted
        assert record['genes'] == genes
        assert record['groups'] == groups
        assert sum(record['group_sizes']) == admitted
        assert record['morphology_or_HE_used'] is False
        assert record['spatial_coordinates_used_for_clustering'] is False
        for rel, digest in record['assets_sha256'].items():
            asset = checked_path(base, rel, site)
            assert hashlib.sha256(asset.read_bytes()).hexdigest() == digest, asset
        with (base / 'cell-groups.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == admitted
        assert len({row['cell_id'] for row in rows}) == len(rows)
        assert Counter(int(row['group']) for row in rows) == dict(enumerate(record['group_sizes'], 1))
        assert all(math.isfinite(float(row['x_centroid_um'])) and math.isfinite(float(row['y_centroid_um'])) for row in rows)
    base = site / 'spatial-expression/visium-breast-if'
    record = json.loads((base / 'PROVENANCE.json').read_text())
    assert record['dataset_id'] == 'cytassist_ffpe_protein_expression_human_breast_cancer'
    assert record['stain'] == 'immunofluorescence'
    assert record['native_filtered_spots'] == record['analyzed_spots'] == 4169
    assert record['gene_expression_features'] == 18085
    assert record['antibody_capture_features_preserved_in_source'] == 35
    assert record['groups'] == 13 and sum(record['group_sizes']) == 4169
    assert record['Bora_image_refinement_complete'] is False
    assert record['HE_Virchow2_used'] is False
    for rel, digest in record['assets_sha256'].items():
        asset = checked_path(base, rel, site)
        assert hashlib.sha256(asset.read_bytes()).hexdigest() == digest, asset
    with (base / 'spot-groups.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4169 and len({row['barcode'] for row in rows}) == 4169
    assert Counter(int(row['group']) for row in rows) == dict(enumerate(record['group_sizes'], 1))
    assert sum(float(row['x_fullres_px']) < 0 for row in rows) == 4
    return len(expected) + 1


def verify_inspectors(site, sections):
    """All published tissues must link to a registered, correctly aligned inspector."""
    manifest = json.loads((site / 'geojson-inspectors.json').read_text())
    assert manifest['format'] == 'bora-geojson-inspectors-v1'
    additional = site / 'visium-hd/ADDITIONAL_WSI_PROVENANCE.json'
    extras = json.loads(additional.read_text()) if additional.exists() else {}
    expected = {'visium-hd-' + tissue for tissue in [*sections['whole_slide_tissues'], *extras]}
    expected |= {'visium-breast', 'visium-colorectal', 'visium-tonsil'}
    assert {row['id'] for row in manifest['records']} == expected
    assert len(manifest['records']) == len(expected)
    assert manifest['public_dataset_images'] == 116
    root = (site / 'index.html').read_text()
    assert 'id="inspect-geojson"' in root
    for record in manifest['records']:
        page = checked_path(site, record['page'].split('#')[0], site)
        content = page.read_text()
        assert content.count('class="geojson-inspector"') == 1
        assert 'id="inspect-geojson"' in content
        assert f'data-record="{record["id"]}"' in content
        assert 'data-manifest="../geojson-inspectors.json"' in content
        assert '../geojson-inspector.js' in content and '../geojson-inspector.css' in content
        assert record['page'] in root
        assert record['fastpls_tissue_names_available'] is False
        assert 'fastPLS tissue-name assignment is not available' in record['label_description']
        image = checked_path(site, record['image'], site)
        assert hashlib.sha256(image.read_bytes()).hexdigest() == record['image_sha256']
        geometry = checked_path(site, record['geometry'], site)
        stored = geometry.read_bytes()
        assert len(stored) == record['geojson_bytes']
        assert hashlib.sha256(stored).hexdigest() == record['stored_geojson_sha256']
        decoded = gzip.decompress(stored) if geometry.suffix == '.gz' else stored
        assert hashlib.sha256(decoded).hexdigest() == record['decoded_geojson_sha256']
        vector = json.loads(decoded)
        assert record['native_shape_yx'] == vector['native_shape_yx']
        assert record['native_mpp_xy'] == vector['native_mpp_xy']
        x0, y0, x1, y1 = vector['full_slide_input_identity']['analysis_bounds_native_xyxy']
        assert record['image_bounds_xywh'] == [x0, y0, x1-x0, y1-y0]
        assert len(vector['features']) == record['source_feature_count'] == len(record['groups'])
        for feature, group in zip(vector['features'], record['groups']):
            props = feature['properties']
            assert group['value'] == props['value']
            assert props['classification'] == f'group_{group["value"]}'
            assert group['area_um2'] == props.get('area_um2')
            rgb = colorsys.hsv_to_rgb(((group['value']-1)*.61803398875) % 1, .68, .9)
            assert group['color'] == '#' + ''.join(f'{round(255*c):02x}' for c in rgb)
    return len(expected)


def verify_template(site):
    pages = [('index.html', 'public', None), ('visium-hd/index.html', 'visium-hd', None),
             ('visium/index.html', 'visium', None),
             ('spatial-expression/index.html', 'spatial-expression', None),
             ('spatial-expression/xenium-breast-2fov.html', 'spatial-expression', 'xenium-breast-2fov'),
             ('spatial-expression/xenium-ovary-multimodal-tiny.html', 'spatial-expression', 'xenium-ovary-multimodal-tiny'),
             ('spatial-expression/xenium-ovary-nuclear-expansion-tiny.html', 'spatial-expression', 'xenium-ovary-nuclear-expansion-tiny'),
             ('spatial-expression/visium-breast-if.html', 'spatial-expression', 'visium-breast-if')]
    extras_path = site / 'visium-hd/ADDITIONAL_WSI_PROVENANCE.json'
    extras = json.loads(extras_path.read_text()) if extras_path.exists() else {}
    pages += [(f'visium-hd/{name}.html', 'visium-hd', name) for name in (*('colon', 'pancreas', 'lymph', 'tonsil'), *extras)]
    pages += [(f'visium/{name}.html', 'visium', name) for name in ('breast', 'colorectal', 'tonsil')]
    titles = ['Public datasets and masks', 'Whole Slide Images', 'Visium gene clusters', 'Spatial expression']
    for relative, collection, tissue in pages:
        content = (site / relative).read_text()
        parsed = Links()
        parsed.feed(content)
        tags = lambda name: [attrs for tag, attrs in parsed.nodes if tag == name]
        body, = tags('body')
        assert 'benchmark-page' in body['class'].split() and body['data-collection'] == collection
        main, = tags('main')
        assert main['id'] == 'main-content'
        header, = tags('header')
        assert header['class'] == 'site-header'
        footer, = tags('footer')
        assert footer['class'] == 'site-footer'
        assert not tags('style'), 'Page-local styles must not override the shared template'
        assert len(tags('h1')) == 1
        prefix = '' if collection == 'public' else '../'
        styles = [node['href'] for node in tags('link') if node.get('rel') == 'stylesheet']
        assert styles[0] == prefix + 'viewer.css'
        assert styles[-1] == prefix + 'benchmark-template.css'
        assert any(node.get('class') == 'skip-link' and node.get('href') == '#main-content' for node in tags('a'))
        nav = re.search(r'<nav class="benchmark-sections"[^>]*>(.*?)</nav>', content, re.S).group(1)
        assert all(f'{i} · {title}' in nav for i, title in enumerate(titles, 1))
        navigation = Links()
        navigation.feed(nav)
        current = [attrs for tag, attrs in navigation.nodes if tag == 'a' and attrs.get('aria-current') == 'page']
        assert len(current) == 1
        assert current[0]['href'] == prefix + {'public': 'index.html', 'visium-hd': 'visium-hd/', 'visium': 'visium/',
                                                'spatial-expression': 'spatial-expression/'}[collection]
        if collection != 'public':
            local = re.search(r'<nav class="tissue-nav"[^>]*>(.*?)</nav>', content, re.S).group(1)
            local_parser = Links()
            local_parser.feed(local)
            active = [attrs for tag, attrs in local_parser.nodes if tag == 'a' and attrs.get('aria-current') == 'page']
            assert len(active) == 1 and active[0]['href'] == (tissue + '.html' if tissue else 'index.html')
        if collection == 'visium':
            assert sum(node.get('src') == 'geojson-download.js' for node in tags('script')) == 1
        if relative == 'visium/index.html':
            assert sum(node.get('class') == 'tissue-card' for node in tags('article')) == 3
            assert 'Coverage limitation:' in re.sub(r'<details\b.*?</details>', '', content, flags=re.S)
        if relative == 'visium-hd/index.html':
            assert sum(node.get('class') == 'tissue-card' for node in tags('article')) == 4 + len(extras)
    return len(pages)


def verify(site):
    site = site.resolve()
    provenance = json.loads((site / 'BUILD_PROVENANCE.json').read_text())
    sections = json.loads((site / 'BENCHMARK_SECTIONS_PROVENANCE.json').read_text())
    expected_tissues = {'colon', 'pancreas', 'lymph', 'tonsil'}
    assert set(sections['whole_slide_tissues']) == expected_tissues
    assert sections['native_geojson_unchanged'] is True
    for rel, record in sections['whole_slide_assets'].items():
        asset = checked_path(site / 'visium-hd', rel, site)
        if hashlib.sha256(asset.read_bytes()).hexdigest() != record['display_sha256']:
            raise ValueError('Whole-slide display asset differs: ' + rel)
        if rel.endswith('.geojson') and record['source_sha256'] != record['display_sha256']:
            raise ValueError('Native whole-slide GeoJSON was altered: ' + rel)
    visium_samples = verify_visium(site, sections)
    additional_samples = verify_additional_wsi(site)
    spatial_expression_samples = verify_spatial_expression(site)
    inspectors = verify_inspectors(site, sections)
    template_pages = verify_template(site)
    root_html = (site / 'index.html').read_text()
    for link in ('visium-hd/', 'visium/', 'id="public-datasets"'):
        assert link in root_html
    for tissue in expected_tissues:
        html = (site / 'visium-hd' / (tissue + '.html')).read_text()
        assert html.count('class="field"') == 9
        assert f'{tissue}/BORA_FULL_SLIDE.wsi.geojson' in html
        assert 'brain.html' not in html
    for rel, digest in provenance['source_asset_sha256'].items():
        p = checked_path(site, rel, site)
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Copied scientific asset differs: ' + rel)
    vector_paths = set()
    for directory, expected in [(site, 1)]:
        data = json.loads((directory / 'data.json').read_text())
        count = data['cohort_cases']
        assert len(data['cases']) == count == provenance['case_count']
        assert len(data['models']) == expected
        assert data['models'][0]['id'] == 'neighbor-context'
        assert Counter(c['dataset'] for c in data['cases']) == data['dataset_counts']
        assert Counter(c['split'] for c in data['cases']) == data['split_counts']
        assert data['evaluation_policy']['development_evaluation'] is True
        assert 'puma' not in data['dataset_counts']
        assert data['threshold_native_pixels'] == 20
        assert len({c['case'] for c in data['cases']}) == count
        for c in data['cases']:
            assert len(c['models']) == expected
            assert c['models'][0]['id'] == 'neighbor-context'
            assert len(c['native_mpp_xy']) == 2 and all(math.isfinite(m) and m > 0 for m in c['native_mpp_xy'])
            assert len(c['native_shape_yx']) == 2 and min(c['native_shape_yx']) >= 224
            assert c['expert_legend']
            for panel in ('original', 'expert', 'tiles', 'converted', 'bora'):
                p = checked_path(directory, c['asset'] + '/' + panel + '.jpg', site)
                assert p.read_bytes().startswith(b'\xff\xd8\xff')
            for m in c['models']:
                checked_path(directory, m['preview'], site)
                vec = checked_path(directory, m['vector'], site)
                vector_paths.add(vec)
                assert 0 <= m['iou'] <= 1 and 0 <= m['group_top1'] <= 1
                assert m['seconds'] >= 0
                assert m['cleanup']['positive_polygon_parts_below_threshold'] == 0
                assert m['cleanup']['polygon_holes_below_threshold'] == 0
        for m in data['models']:
            values = [next(q for q in c['models'] if q['id'] == m['id'])['iou'] for c in data['cases']]
            assert math.isclose(sum(values) / count, m['mean_final_geojson_iou'], abs_tol=1e-12)
        with (directory / 'scores.csv').open() as f:
            assert len(list(csv.DictReader(f))) == count * expected
        payload = (directory / 'data.json').read_text()
        assert not any(p in payload for p in ('/Users/', '/home/', '/mnt/', '/media/', '127.0.0.1', 'localhost:'))
    parts = holes = 0
    for path in vector_paths:
        g = json.loads(path.read_text())
        assert g['type'] == 'FeatureCollection'
        assert g['coordinates_are_roi_local'] is True
        assert g['final_native_pixel_region_cleanup']['positive_polygon_parts_below_threshold'] == 0
        for feature in g['features']:
            geometry = feature['geometry']
            assert geometry['type'] in ('Polygon', 'MultiPolygon')
            polys = [geometry['coordinates']] if geometry['type'] == 'Polygon' else geometry['coordinates']
            for poly in polys:
                areas = [ring_area(r) for r in poly]
                assert areas[0] - sum(areas[1:]) >= 20 - 1e-7, path
                assert all(a >= 20 - 1e-7 for a in areas[1:]), path
                parts += 1
                holes += len(areas) - 1
    for path in site.rglob('*.html'):
        parser = Links()
        parser.feed(path.read_text())
        for link in parser.links:
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            rel = parsed.path + ('index.html' if parsed.path.endswith('/') else '')
            checked_path(path.parent, rel, site)
    for pattern in ['*.html', '*.js', '*.json', '*.csv', '*.md']:
        for path in site.rglob(pattern):
            if re.search(r'\bcycle(?:s)?\b|\bevolution\b|\bP\d{2}\b|\bC\d{2}-\d{2}\b|parent.lineage', path.read_text(), re.I):
                raise ValueError('Internal development history or algorithm code in public site: ' + str(path))
    print(json.dumps(dict(ok=True, cases=provenance['case_count'], copied_assets=len(provenance['source_asset_sha256']),
                         unique_vectors=len(vector_paths), polygon_parts=parts, holes=holes,
                         native_area_threshold=20, visium_whole_section_downloads=visium_samples,
                         visium_minimum_area_um2=400, additional_whole_slides=additional_samples,
                         spatial_expression_samples=spatial_expression_samples,
                         whole_slide_and_expression_inspectors=inspectors,
                         shared_template_pages=template_pages,
                         pages=sum(1 for _ in site.rglob('*.html')))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=Path(__file__).resolve().parents[1] / 'site')
    verify(parser.parse_args().site)
