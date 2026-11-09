"""Build a fail-closed, public-data-only subset of the local Bora viewer.

Run --prepare locally to make resized public WSI previews. CI runs --assemble,
which only consumes committed public-wsi files and the existing public data.
"""
import argparse
import hashlib
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
PREPARED = ROOT / "public-wsi"
OUTPUT = ROOT / "public-site"
CASES = {
    "public-10x-colon-cancer": ("10x colorectal cancer", "colon16-geometry50-v1"),
    "public-10x-lymph-node-3prime": ("10x lymph node, 3′", "lymph-geometry50-v1"),
    "public-10x-tonsil-3prime": ("10x tonsil, 3′", "tonsil-geometry50-v1"),
    "public-10x-brain-cancer-ffpe": ("10x brain cancer FFPE", "public-10x-brain-cancer-ffpe-v1"),
    "public-10x-breast-ff": ("10x breast cancer", "public-10x-breast-ff-v2"),
    "public-10x-kidney-ffpe": ("10x kidney FFPE", "public-10x-kidney-ffpe-v1"),
    "public-10x-lung-cancer-fixed-frozen": ("10x lung cancer", "public-10x-lung-cancer-fixed-frozen-v1"),
    "public-10x-lymph-node-ffpe": ("10x lymph node FFPE", "public-10x-lymph-node-ffpe-v1"),
    "public-10x-ovarian-3prime": ("10x ovarian cancer, 3′", "public-10x-ovarian-3prime-v2"),
    "public-10x-ovarian-ff": ("10x ovarian cancer, fresh frozen", "public-10x-ovarian-ff-v2"),
    "public-10x-pancreatic-cancer-fresh-frozen": ("10x pancreatic cancer", "public-10x-pancreatic-cancer-fresh-frozen-v1"),
    "public-10x-prostate-ffpe": ("10x prostate cancer FFPE", "public-10x-prostate-ffpe-v1"),
    "tcga-acc-or-a5jp-01z-00-dx1": ("TCGA-ACC OR-A5JP", "tcga-acc-or-a5jp-01z-00-dx1-reduced256-v1"),
}
FORBIDDEN = ("brendon", "pm016", "pm049", "f0065", "f0085", "scanned-image-2", "/media/user/")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resized(source, target, limit=1500):
    from PIL import Image
    with Image.open(source) as image:
        image.thumbnail((limit, limit), getattr(Image, "Resampling", Image).LANCZOS)
        if image.mode == "RGBA":
            rgb = Image.new("RGB", image.size, "white")
            rgb.paste(image, mask=image.getchannel("A"))
            image = rgb
        else:
            image = image.convert("RGB")
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, "JPEG", quality=84, optimize=True, progressive=True)


def resized_grayscale(source, target):
    from PIL import Image
    with Image.open(source) as image:
        assert image.mode == "L", source
        image.thumbnail((1200, 1200), getattr(Image, "Resampling", Image).LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, "WEBP", quality=82, method=6)


def prepare():
    PREPARED.mkdir(exist_ok=True)
    for slug, (title, source_dir) in CASES.items():
        source = SITE / "visium-hd" / source_dir
        assert source.is_dir(), source
        target = PREPARED / slug
        if target.exists():
            assert (target / "SOURCE.json").is_file(), target
            continue  # Preserve previously reviewed public previews.
        target.mkdir()
        pairs = [("he-overview.jpg", "he-overview.jpg", 1800),
                 ("full-output-mask-with-geojson.jpg", "mask-overview.jpg", 1800)]
        for kind in ("wide", "large", "field"):
            for number in (1, 2, 3):
                folder = f"{kind}-{number:02d}"
                pairs += [(f"{folder}/he.jpg", f"{folder}-he.jpg", 1400),
                          (f"{folder}/output-mask-with-geojson.jpg", f"{folder}-mask.jpg", 1400)]
        for input_name, output_name, limit in pairs:
            src = source / input_name
            assert src.is_file(), src
            resized(src, target / output_name, limit)
        kodama = source / "kodama-groups.png"
        assert kodama.is_file(), kodama
        shutil.copy2(kodama, target / "kodama-groups.png")
        archives = sorted(source.glob("*wsi-pls.geojson.gz")) or sorted(source.glob("*wsi.geojson.gz"))
        if archives:
            assert archives[0].stat().st_size < 60_000_000
            shutil.copy2(archives[0], target / "Bora-GeoJSON.gz")
        (target / "SOURCE.json").write_text(json.dumps({"title": title, "source_directory": source_dir,
            "source_credits": "10x Genomics public example H&E" if slug.startswith("public-10x") else "NCI GDC open-access TCGA-ACC H&E",
            "visuals_are_downsampled": True, "result_status": "exploratory"}, indent=2) + "\n")
    for slug, (_, source_dir) in CASES.items():
        source = SITE / "visium-hd" / source_dir
        target = PREPARED / slug
        groups = sorted(source.glob("group-*.jpg"))
        assert groups, source
        assert [p.name for p in groups] == [f"group-{i:02d}.jpg" for i in range(1, len(groups) + 1)]
        for group in groups:
            output = target / (group.stem + ".webp")
            if not output.exists():
                resized_grayscale(group, output)
    print(f"Prepared {len(CASES)} public WSI previews")


def page(slug, title):
    case = PREPARED / slug
    geo = f'<p><a href="{slug}/Bora-GeoJSON.gz" download>Download compressed Bora GeoJSON</a></p>' if (case / "Bora-GeoJSON.gz").exists() else ""
    group_files = sorted(case.glob("group-*.webp"))
    assert group_files and [p.name for p in group_files] == [f"group-{i:02d}.webp" for i in range(1, len(group_files) + 1)]
    groups = ('<section id="groups"><h2>Grayscale group responses across the full slide</h2>'
              '<p class="note">These black-and-white maps show each anonymous group relative to its '
              'strongest competing group. They are diagnostic displays, not expert tissue labels.</p>'
              '<div class="group-grid">' + ''.join(
                  f'<figure><img loading="lazy" src="{slug}/{p.name}" '
                  f'alt="Grayscale response for anonymous Bora group {i:02d}">'
                  f'<figcaption>Group {i:02d} versus strongest competing group</figcaption></figure>'
                  for i, p in enumerate(group_files, 1)) + '</div></section>')
    zooms = "".join(f'<section class="zoom-pair"><h3>{kind.title()} {n:02d}</h3><div class="overview">'
        f'<figure><img loading="lazy" src="{slug}/{kind}-{n:02d}-he.jpg" alt="H&amp;E field"><figcaption>H&amp;E</figcaption></figure>'
        f'<figure><img loading="lazy" src="{slug}/{kind}-{n:02d}-mask.jpg" alt="Bora mask on the same field"><figcaption>Bora mask</figcaption></figure>'
        '</div></section>' for kind in ("wide", "large", "field") for n in (1, 2, 3))
    credit = "10x Genomics public example data" if slug.startswith("public-10x") else "NCI GDC open-access slide image"
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)} · Bora</title><link rel="stylesheet" href="../viewer.css"><link rel="stylesheet" href="../benchmark-template.css"></head><body class="benchmark-page" data-collection="visium-hd"><a class="skip-link" href="#main-content">Skip to result</a><header class="site-header"><p class="eyebrow">BORA · BENCHMARK VIEWER</p><h1>{escape(title)}</h1><p class="site-intro">Exploratory H&amp;E whole-slide result. Source: {escape(credit)}.</p><nav class="benchmark-sections"><a href="../index.html">1 · Public datasets and masks</a><a href="index.html">2 · Whole Slide Images</a><a href="../visium/">3 · Visium gene clusters</a></nav></header><main id="main-content"><p><a href="index.html">← All public whole-slide examples</a></p><p class="note">Images here are downsampled for web review. Anonymous groups and model-predicted tissue names are exploratory, not expert-validated diagnoses.</p><section id="overview"><h2>Full-slide overview</h2><div class="overview"><figure><img src="{slug}/he-overview.jpg" alt="Original H&amp;E overview"><figcaption>H&amp;E overview</figcaption></figure><figure><img src="{slug}/mask-overview.jpg" alt="Bora result on H&amp;E"><figcaption>Bora output and contours</figcaption></figure></div></section><section id="kodama"><h2>KODAMA embedding</h2><figure><img src="{slug}/kodama-groups.png" alt="KODAMA plot with Bora groups"><figcaption>KODAMA groups</figcaption></figure></section>{groups}{geo}<section id="zooms"><h2>Nine H&amp;E and Bora mask fields</h2>{zooms}</section></main><footer class="site-footer">Research preview · {escape(credit)} · <a href="../attribution.html">Attribution and reuse terms</a></footer></body></html>'''


def assemble():
    assert PREPARED.is_dir()
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir()
    for directory in ("assets", "previews", "vectors", "visium", "visium-hd-colon"):
        shutil.copytree(SITE / directory, OUTPUT / directory)
    shutil.copytree(SITE / "benchmark1-failures", OUTPUT / "benchmark1-failures")
    shutil.copytree(SITE / "benchmark1-comparators", OUTPUT / "benchmark1-comparators")
    for name in ("viewer.js", "viewer.css", "benchmark-template.css", "geojson-inspector.js",
                 "geojson-inspector.css", "data.json", "scores.csv", "BUILD_PROVENANCE.json",
                 "attribution.html", "DATA_LICENSES.md"):
        shutil.copy2(SITE / name, OUTPUT / name)
    inspectors = json.loads((SITE / "geojson-inspectors.json").read_text())
    inspectors["records"] = [row for row in inspectors["records"] if row["id"] in {"visium-breast", "visium-colorectal", "visium-tonsil"}]
    (OUTPUT / "geojson-inspectors.json").write_text(json.dumps(inspectors, indent=2) + "\n")
    root = (SITE / "index.html").read_text()
    root = re.sub(r'<details><summary>Inspect tissues in the other benchmarks</summary>.*?</details>', "", root, flags=re.S)
    root = re.sub(r'<!-- BENCHMARK 1 RERUN STATUS[^>]*--><aside.*?</aside>',
        '<aside role="status" class="notice"><strong>Development evaluation.</strong> The 503-image full-channel rerun is complete. <a href="benchmark1-comparators/">Compare Bora, expert annotations, U-Net, VISTA-PATH and MedSAM on all 106 development evaluation images</a>. The images were inspected during development, and model scores use different matched class sets. <a href="benchmark1-failures/">View the 58 images without a scored result in the earlier image-only Bora run</a>. The interactive gallery below is an older controlled display.</aside>', root, flags=re.S)
    (OUTPUT / "index.html").write_text(root)
    shutil.copytree(PREPARED, OUTPUT / "visium-hd")
    cards = "".join(f'<article class="tissue-card"><a class="card-preview" href="{slug}.html"><img loading="lazy" src="{slug}/he-overview.jpg" alt="{escape(title)} H&amp;E overview"></a><div class="card-body"><h2>{escape(title)}</h2><p>Exploratory H&amp;E result · KODAMA plot · nine paired fields</p><a href="{slug}.html">View result →</a></div></article>' for slug, (title, _) in CASES.items())
    (OUTPUT / "visium-hd/index.html").write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Whole Slide Images · Bora</title><link rel="stylesheet" href="../viewer.css"><link rel="stylesheet" href="../benchmark-template.css"></head><body class="benchmark-page" data-collection="visium-hd"><a class="skip-link" href="#main-content">Skip to results</a><header class="site-header"><p class="eyebrow">BORA · BENCHMARK VIEWER</p><h1>Whole Slide Images</h1><p class="site-intro">Public 10x and open-access TCGA H&amp;E examples. Exploratory results.</p><nav class="benchmark-sections"><a href="../index.html">1 · Public datasets and masks</a><a href="index.html">2 · Whole Slide Images</a><a href="../visium/">3 · Visium gene clusters</a></nav></header><main id="main-content"><h2>Browse results</h2><div class="tissue-cards">{cards}</div></main><footer class="site-footer">Data providers retain their rights · <a href="../attribution.html">Attribution and reuse terms</a></footer></body></html>')
    for slug, (title, _) in CASES.items():
        (OUTPUT / "visium-hd" / f"{slug}.html").write_text(page(slug, title))
    verify()


class LocalLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        self.links += [value for key, value in attrs if key in ("src", "href") and value]


def verify():
    files = [p for p in OUTPUT.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    assert total < 950_000_000, total
    assert max(p.stat().st_size for p in files) < 100_000_000
    assert len([p for p in (OUTPUT / "visium-hd").glob("*.html")]) == len(CASES) + 1
    comparator = OUTPUT / "benchmark1-comparators"
    manifest = json.loads((comparator / "cases.json").read_text())
    assert manifest["status"] == "AUDITED_PUBLIC_BENCHMARK1_COMPARATOR_VISUALS"
    assert manifest["case_count"] == 106 == len(manifest["cases"])
    assert not manifest["untouched_test"]
    assert len(list(comparator.glob("case-*.webp"))) == 106
    for row in manifest["cases"]:
        assert re.fullmatch(r"case-\d{3}\.webp", row["image"])
        path = comparator / row["image"]
        assert sha(path) == row["image_sha256"] and path.stat().st_size == row["preview_bytes"]
    for p in files:
        rel = p.relative_to(OUTPUT).as_posix().lower()
        assert not any(word in rel for word in FORBIDDEN), rel
        if p.suffix.lower() in (".html", ".js", ".css", ".json", ".md", ".csv"):
            data = p.read_text()
            assert not any(word in data.lower() for word in FORBIDDEN), rel
        if p.suffix == ".html":
            links = LocalLinks(); links.feed(p.read_text())
            for link in links.links:
                if link.startswith(("https:", "http:", "mailto:", "#", "data:")):
                    continue
                path = unquote(urlsplit(link).path)
                if path:
                    target = (p.parent / path).resolve()
                    assert target.is_relative_to(OUTPUT) and target.exists(), (rel, link)
    receipt = {"format": "bora-public-site-v1", "size_bytes": total, "file_count": len(files),
               "public_wsi_count": len(CASES), "file_sha256": {str(p.relative_to(OUTPUT)): sha(p) for p in files}}
    (OUTPUT / "PUBLIC_RELEASE_VERIFIED.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Public release verified: {len(files)} files, {total/1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("prepare", "assemble", "verify"))
    args = parser.parse_args()
    {"prepare": prepare, "assemble": assemble, "verify": verify}[args.command]()
