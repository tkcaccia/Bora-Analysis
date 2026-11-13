"""Verify the current public Bora Pages tree and its internal links."""

import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

FORBIDDEN = ('brendon', 'pm016', 'pm049', 'f0065', 'f0085',
             'scanned-image-2', '/media/user/', 'tcga-acc',
             'a5jp', 'a5jy', 'a5k8', 'a5k9')

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in ('href', 'src') and value)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main(site):
    site = site.resolve()
    receipt = json.loads((site / 'PUBLIC_RELEASE_VERIFIED.json').read_text())
    assert receipt['format'] == 'bora-public-site-v2'
    assert receipt['public_wsi_count'] == 12
    assert receipt['comparator_wsi_count'] == 7
    assert receipt['grayscale_group_map_count'] == 129
    files = {str(path.relative_to(site)): path for path in site.rglob('*')
             if path.is_file() and path.name != 'PUBLIC_RELEASE_VERIFIED.json'}
    assert set(files) == set(receipt['file_sha256'])
    assert len(files) == receipt['file_count']
    assert sum(path.stat().st_size for path in files.values()) == receipt['size_bytes']
    assert receipt['size_bytes'] < 950_000_000
    for name, path in files.items():
        assert path.stat().st_size < 100_000_000, name
        assert not any(term in name.lower() for term in FORBIDDEN), name
        assert sha(path) == receipt['file_sha256'][name], name
        if path.suffix.lower() not in ('.html', '.js', '.css', '.json', '.md', '.csv'):
            continue
        content = path.read_text()
        assert not any(term in content.lower() for term in FORBIDDEN), name
        if path.suffix.lower() != '.html':
            continue
        parser = Links(); parser.feed(content)
        for link in parser.links:
            if link.startswith(('https:', 'http:', 'mailto:', '#', 'data:')):
                continue
            local = unquote(urlsplit(link).path)
            if local:
                target = (path.parent / local).resolve()
                assert target.is_relative_to(site) and target.exists(), (name, link)
    assert len(list((site / 'visium-hd').glob('*.html'))) == 13
    assert len(list((site / 'visium-hd/comparators').glob('*.html'))) == 8
    for page in (site / 'visium-hd').glob('*.html'):
        if page.name != 'index.html':
            assert '<section id="groups">' in page.read_text(), page
    assert all((site / section / 'index.html').is_file() for section in ('visium', 'visium-hd'))
    print(f"Verified public Pages tree: {len(files)} files, {receipt['size_bytes']} bytes")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: verify_public_release.py SITE')
    main(Path(sys.argv[1]))
