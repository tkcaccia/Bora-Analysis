"""Fail closed before uploading the curated Bora Pages artifact."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit


FORBIDDEN = ("brendon", "pm016", "pm049", "f0065", "f0085", "scanned-image-2", "/media/user/")


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in ("href", "src") and value)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


site = Path(sys.argv[1]).resolve()
receipt = json.loads((site / "PUBLIC_RELEASE_VERIFIED.json").read_text())
assert receipt["format"] == "bora-public-site-v1" and receipt["public_wsi_count"] == 13
files = {str(p.relative_to(site)): p for p in site.rglob("*") if p.is_file() and p.name != "PUBLIC_RELEASE_VERIFIED.json"}
assert set(files) == set(receipt["file_sha256"]), "Asset set changed after verification"
assert sum(p.stat().st_size for p in files.values()) == receipt["size_bytes"]
assert receipt["size_bytes"] < 950_000_000
for rel, path in files.items():
    assert path.stat().st_size < 100_000_000, rel
    assert not any(term in rel.lower() for term in FORBIDDEN), rel
    assert sha(path) == receipt["file_sha256"][rel], rel
    if path.suffix.lower() in (".html", ".js", ".css", ".json", ".md", ".csv"):
        content = path.read_text()
        assert not any(term in content.lower() for term in FORBIDDEN), rel
        if path.suffix.lower() == ".html":
            parser = Links(); parser.feed(content)
            for link in parser.links:
                if link.startswith(("https:", "http:", "mailto:", "#", "data:")):
                    continue
                local = unquote(urlsplit(link).path)
                if local:
                    target = (path.parent / local).resolve()
                    assert target.is_relative_to(site) and target.exists(), (rel, link)
assert len(list((site / "visium-hd").glob("*.html"))) == 14
assert receipt["grayscale_group_map_count"] == 143
assert len(list((site / "visium-hd").glob("*/group-*.webp"))) == 143
for page in (site / "visium-hd").glob("*.html"):
    if page.name != "index.html":
        assert '<section id="groups">' in page.read_text(), page
assert all((site / section / "index.html").is_file() for section in ("visium", "visium-hd"))
print(f"Verified public-only Pages artifact: {len(files)} files, {receipt['size_bytes']/1e6:.1f} MB")
