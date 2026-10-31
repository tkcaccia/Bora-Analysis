"""Publish verified benchmark display assets without changing scientific results.

Example:
  python tools/build_benchmark_sections.py --wsi-viewer /path/to/verified/viewer \
      --visium-render-root /path/to/verified/whole-section/renders \
      --visium-study-root /path/to/verified/whole-section/outputs

The WSI HTML and images are display copies of a completed whole-slide viewer.
Source files are never changed. Large raster views are resized for web display;
native full-slide GeoJSON files are copied without modification.
"""

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil

from PIL import Image
from build_visium_fullsection import main as build_visium_fullsection
from add_geojson_inspectors import attach as attach_geojson_inspectors
from apply_benchmark_template import apply as apply_benchmark_template


TISSUES = ("colon", "pancreas", "lymph", "tonsil")


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = set()

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("src", "href") and value:
                self.paths.add(value)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def display_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in (".jpg", ".jpeg", ".png") and source.name != "kodama-groups.png":
        with Image.open(source) as im:
            if max(im.size) > 1800:
                im.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
            im.convert("RGB").save(destination, "JPEG", quality=84, optimize=True)
    else:
        shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsi-viewer", type=Path, required=True)
    parser.add_argument("--visium-render-root", type=Path, required=True)
    parser.add_argument("--visium-study-root", type=Path, required=True)
    parser.add_argument("--site", type=Path, default=Path(__file__).resolve().parents[1] / "site")
    args = parser.parse_args()
    wsi = args.wsi_viewer.resolve()
    site = args.site.resolve()
    receipt = json.loads((wsi / "VIEWER_COMPLETE.json").read_text())
    if receipt.get("status") != "COMPLETE" or not receipt.get("five_full_slide_pages"):
        raise ValueError("A completed whole-slide viewer is required")
    if not receipt.get("all_new_zooms_are_exact_full_slide_crops"):
        raise ValueError("The whole-slide crop audit is missing")
    source_hashes = receipt["files_sha256"]
    for name in ("index.html", "colon.html", "pancreas.html", "lymph.html", "tonsil.html"):
        if sha(wsi / name) != source_hashes[name]:
            raise ValueError(f"Viewer receipt mismatch: {name}")

    output = site / "visium-hd"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir()
    copied = {}
    for tissue in TISSUES:
        source_html = (wsi / f"{tissue}.html").read_text()
        refs = References()
        refs.feed(source_html)
        for ref in sorted(refs.paths):
            if not ref.startswith(tissue + "/"):
                continue
            original = (wsi / ref).resolve()
            if not original.is_relative_to(wsi) or not original.is_file():
                raise ValueError(f"Missing whole-slide display asset: {ref}")
            target_ref = re.sub(r"/he(?:-corrected)?\.png$", lambda m: m.group().replace(".png", ".jpg"), ref)
            target = output / target_ref
            display_copy(original, target)
            source_digest = sha(original)
            if ref in source_hashes and source_digest != source_hashes[ref]:
                raise ValueError(f"Viewer asset receipt mismatch: {ref}")
            copied[target_ref] = {"source_sha256": source_digest, "display_sha256": sha(target),
                                  "reencoded_for_display": target.suffix == ".jpg"}
            if target_ref != ref:
                source_html = source_html.replace(ref, target_ref)
        source_html = re.sub(r'<a href="brain\.html"[^>]*>.*?</a>', '', source_html)
        source_html = source_html.replace('href="index.html"', 'href="index.html"')
        source_html = re.sub(r'<a href="http://127\.0\.0\.1:[^\"]+"[^>]*>.*?</a>', '', source_html)
        source_html = source_html.replace("<nav>", '<nav><a href="../index.html">Public datasets</a><a href="../visium/index.html">Visium gene clusters</a>')
        source_html = source_html.replace("</main>", '<p class="note">Web images are resized for viewing. The downloadable GeoJSON retains full-slide coordinates.</p></main>')
        (output / f"{tissue}.html").write_text(source_html)

    index = (wsi / "index.html").read_text()
    index = re.sub(r'<a href="brain\.html"[^>]*>.*?</a>', '', index)
    index = re.sub(r'<a class="tissue" href="brain\.html">.*?</a>', '', index)
    index = index.replace("<nav>", '<nav><a href="../index.html">Public datasets</a><a href="../visium/index.html">Visium gene clusters</a>')
    index = index.replace("Select a tissue", "Whole Slide Images benchmark")
    index = index.replace("Bora · select a tissue", "Bora · Whole Slide Images")
    (output / "index.html").write_text(index)
    shutil.copy2(wsi / "physical-scale.css", output / "physical-scale.css")

    build_visium_fullsection(args.visium_render_root.resolve(), args.visium_study_root.resolve(), site)
    provenance = {
        "whole_slide_receipt_sha256": sha(wsi / "VIEWER_COMPLETE.json"),
        "whole_slide_tissues": list(TISSUES),
        "whole_slide_assets": copied,
        "visium_fullsection_provenance": "FULLSECTION_PROVENANCE.json",
        "display_max_dimension_px": 1800,
        "native_geojson_unchanged": True,
    }
    (site / "BENCHMARK_SECTIONS_PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    attach_geojson_inspectors(site)
    apply_benchmark_template(site)
    print(json.dumps({"tissues": len(TISSUES), "wsi_assets": len(copied),
                      "visium_whole_section_samples": 3}))


if __name__ == "__main__":
    main()
