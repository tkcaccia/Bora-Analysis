"""Verify the exact curated Pages tree in a sparse GitHub Actions checkout.

Archived source media are independently audited before staging; this job
checks only files uploaded to Pages, their links, and the public allowlist.
"""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

FORBIDDEN = ('brendon', 'pm016', 'pm049', 'f0065', 'f0085', 'scanned-image-2', '/media/user/')


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in ('href', 'src') and value)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main(site):
    site = site.resolve()
    receipt = json.loads((site / 'PUBLIC_RELEASE_VERIFIED.json').read_text())
    assert receipt['format'] == 'bora-public-site-v1' and receipt['public_wsi_count'] == 13
    files = {str(p.relative_to(site)): p for p in site.rglob('*')
             if p.is_file() and p.name != 'PUBLIC_RELEASE_VERIFIED.json'}
    assert set(files) == set(receipt['file_sha256'])
    assert len(files) == receipt['file_count']
    assert sum(p.stat().st_size for p in files.values()) == receipt['size_bytes']
    assert receipt['size_bytes'] < 950_000_000
    for rel, path in files.items():
        assert path.stat().st_size < 100_000_000 and not any(x in rel.lower() for x in FORBIDDEN)
        assert sha(path) == receipt['file_sha256'][rel], rel
        if path.suffix.lower() not in ('.html', '.js', '.css', '.json', '.md', '.csv'):
            continue
        content = path.read_text()
        assert not any(x in content.lower() for x in FORBIDDEN), rel
        if path.suffix.lower() != '.html':
            continue
        parser = Links()
        parser.feed(content)
        for link in parser.links:
            if link.startswith(('https:', 'http:', 'mailto:', '#', 'data:')):
                continue
            local = unquote(urlsplit(link).path)
            if local:
                target = (path.parent / local).resolve()
                assert target.is_relative_to(site) and target.exists(), (rel, link)
    assert len(list((site / 'visium-hd').glob('*.html'))) == 14
    assert (site / 'benchmark1-comparators/index.html').is_file()
    print(f"Verified Pages tree: {len(files)} files, {receipt['size_bytes']} bytes")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: verify_pages_site_only.py SITE')
    main(Path(sys.argv[1]))
