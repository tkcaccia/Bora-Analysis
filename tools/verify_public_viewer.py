"""Validate the complete static viewer without scientific or GPU dependencies."""
import argparse
from collections import Counter
import csv
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

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('src', 'href') and value:
                self.links.append(value)


def verify(site):
    site = site.resolve()
    provenance = json.loads((site / 'BUILD_PROVENANCE.json').read_text())
    for rel, digest in provenance['source_asset_sha256'].items():
        p = checked_path(site, rel, site)
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Copied scientific asset differs: ' + rel)
    vector_paths = set()
    for directory, expected in [(site, 1), (site / 'comparison', 5)]:
        data = json.loads((directory / 'data.json').read_text())
        count = data['cohort_cases']
        assert len(data['cases']) == count == provenance['case_count']
        assert len(data['models']) == expected
        assert Counter(c['dataset'] for c in data['cases']) == data['dataset_counts']
        assert Counter(c['split'] for c in data['cases']) == data['split_counts']
        assert data['evaluation_policy']['development_evaluation'] is True
        assert 'puma' not in data['dataset_counts']
        assert data['threshold_native_pixels'] == 20
        assert len({c['case'] for c in data['cases']}) == count
        for c in data['cases']:
            assert len(c['models']) == expected
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
                         native_area_threshold=20, pages=sum(1 for _ in site.rglob('*.html')))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=Path(__file__).resolve().parents[1] / 'site')
    verify(parser.parse_args().site)
