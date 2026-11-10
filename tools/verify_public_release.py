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
if receipt.get("public_media_archived_file_count"):
    archive = site.parent / "archive/public-media-2026-11-10-v3"
    archived = json.loads((archive / "ARCHIVE_VERIFIED.json").read_text())
    assert archived["status"] == "PINNED_PUBLIC_MEDIA_ARCHIVED_OUTSIDE_PAGES"
    assert archived["file_count"] == receipt["public_media_archived_file_count"] == 4086
    assert archived["bytes"] == receipt["public_media_archived_bytes"]
    assert archived["raw_prefix"] == (
        "https://raw.githubusercontent.com/tkcaccia/Bora-Analysis/"
        "b0f5ec6741a47acbbb8559ed80e7b89c3ba15973/site/"
    )
    assert sum(e["bytes"] for e in archived["files"].values()) == archived["bytes"]
    for rel, entry in archived["files"].items():
        path = archive / rel
        assert not (site.parent / rel).exists(), rel
        assert path.stat().st_size == entry["bytes"], rel
        assert sha(path) == entry["sha256"], rel
    assert len([r for r in archived["files"] if r.startswith("site/visium-hd/") and
                "/group-" in r and r.endswith(".webp")]) == 143
    assert archived["case_paths_rewritten"] > 400
    assert archived["training_assets_rewritten"] > 80
    assert len(archived["wsi_pages_rewritten"]) >= 14
    assert archived["raw_prefix"] in (site / "data.json").read_text()
    assert archived["raw_prefix"] in (site / "benchmark1-comparators/cases.json").read_text()
    assert archived["raw_prefix"] in (site / "benchmark1-comparators/gallery.js").read_text()
else:
    assert len(list((site / "visium-hd").glob("*/group-*.webp"))) == 143
if receipt.get("visium_pinned_raw_asset_count"):
    archive = site.parent / "archive/visium-fullsection-2026-11-10-v1"
    archived = json.loads((archive / "ARCHIVE_VERIFIED.json").read_text())
    assert archived["format"] == "bora-pinned-visium-assets-v1"
    assert archived["file_count"] == receipt["visium_pinned_raw_asset_count"] == 192
    assert not (site / "visium/fullsection").exists()
    assert len(archived["sha256"]) == 192
    assert archived["pinned_raw_prefix"].startswith(
        "https://raw.githubusercontent.com/tkcaccia/Bora-Analysis/"
    )
    assert sum((archive / rel).stat().st_size for rel in archived["sha256"]) == archived["bytes"]
    for rel, digest in archived["sha256"].items():
        assert sha(archive / rel) == digest, rel
    for page in (site / "visium").glob("*.html"):
        content = page.read_text()
        assert archived["pinned_raw_prefix"] in content
        assert '="fullsection/' not in content
    assert archived["pinned_raw_prefix"] in (site / "geojson-inspectors.json").read_text()
for page in (site / "visium-hd").glob("*.html"):
    if page.name != "index.html":
        assert '<section id="groups">' in page.read_text(), page
assert all((site / section / "index.html").is_file() for section in ("visium", "visium-hd"))
print(f"Verified public-only Pages artifact: {len(files)} files, {receipt['size_bytes']/1e6:.1f} MB")
