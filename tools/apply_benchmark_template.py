"""Apply one static page template to the three benchmark collections.

Run after section builders and add_geojson_inspectors.py. Scientific image,
vector, metadata and score files are never modified. Existing physical image
frames, scale bars, coverage disclosures and downloads are retained.
"""
import argparse
import html
import json
from pathlib import Path
import re


COLLECTIONS = {
    'public': ('Public datasets and masks', 'BCSS, TIGER, GCSS, IGNITE · expert tissue masks',
               'Bora refinement and tissue-group identification on public histology images with expert annotations.'),
    'visium-hd': ('Visium HD whole slides', 'Colorectal cancer, pancreas, lymph node, tonsil',
                  'Explore complete whole-slide Bora results. All nine zooms per tissue are crops of one saved global output.'),
    'visium': ('Visium gene clusters', 'Giotto, Seurat, SpatialExperiment · expression spot groups',
               'Explore whole-section Bora refinement of gene-expression spot groups from three analysis workflows.'),
}
TISSUES = {
    'visium-hd': {'colon': 'Colorectal cancer', 'pancreas': 'Pancreas', 'lymph': 'Lymph node', 'tonsil': 'Tonsil'},
    'visium': {'breast': 'Breast cancer · Giotto', 'colorectal': 'Colorectal cancer · Seurat', 'tonsil': 'Tonsil · SpatialExperiment'},
}


def header(collection, prefix):
    paths = {'public': prefix + 'index.html', 'visium-hd': prefix + 'visium-hd/', 'visium': prefix + 'visium/'}
    links = []
    for index, (key, (title, description, _)) in enumerate(COLLECTIONS.items(), 1):
        current = ' aria-current="page"' if key == collection else ''
        links.append(f'<a href="{paths[key]}"{current}><strong>{index} · {title}</strong>'
                     f'<span>{html.escape(description)}</span></a>')
    title, _, introduction = COLLECTIONS[collection]
    return ('<header class="site-header"><p class="eyebrow">BORA · BENCHMARK VIEWER</p>'
            f'<h1>{title}</h1><p class="site-intro">{introduction}</p>'
            '<nav class="benchmark-sections" aria-label="Benchmark collections">' + ''.join(links) + '</nav>'
            '<nav class="links" aria-label="Project">'
            '<a href="https://github.com/tkcaccia/Bora-Analysis">Benchmark code</a>'
            '<a href="https://github.com/tkcaccia/Bora">Bora application</a>'
            f'<a href="{prefix}attribution.html">Datasets and attribution</a></nav></header>')


def footer(prefix):
    return ('<footer class="site-footer"><p>Exploratory benchmark evaluation · Data providers retain their rights. '
            f'<a href="{prefix}attribution.html">Attribution and reuse terms</a> · '
            f'<a href="{prefix}BUILD_PROVENANCE.json">Build provenance</a></p></footer>')


def tissue_nav(collection, tissue):
    links = ['<a href="index.html"' + (' aria-current="page"' if tissue is None else '') + '>All tissues</a>']
    for slug, name in TISSUES[collection].items():
        current = ' aria-current="page"' if slug == tissue else ''
        links.append(f'<a href="{slug}.html"{current}>{html.escape(name)}</a>')
    return '<nav class="tissue-nav" aria-label="Tissues in this benchmark">' + ''.join(links) + '</nav>'


def card(record, details='', disclosure=''):
    slug = record['id'].removeprefix(record['section'] + '-')
    image = record['image'].removeprefix(record['section'] + '/')
    title = html.escape(record['title'])
    origin = 'image-derived' if record['section'] == 'visium-hd' else 'gene-expression'
    _, _, width, height = record['image_bounds_xywh']
    mpp_x, mpp_y = record['native_mpp_xy']
    extra = ('<details><summary>Sample details and downloads</summary>' + details + '</details>') if details else ''
    return (f'<article id="{slug}" class="tissue-card"><a class="card-preview" href="{slug}.html">'
            f'<img loading="lazy" src="{image}" alt="{title}: H&amp;E overview"></a><div class="card-body">'
            f'<h2>{title}</h2><p class="card-meta">{record["source_feature_count"]} anonymous {origin} groups · 9 zooms</p>'
            f'<span class="card-dimensions">{width*mpp_x/1000:.2f} × {height*mpp_y/1000:.2f} mm</span>'
            f'<div class="card-actions"><a class="primary-link" href="{slug}.html">View results →</a>'
            f'<a href="{slug}.html#inspect-geojson">Inspect GeoJSON</a></div>{extra}{disclosure}</div></article>')


def landing_cards(content, collection, records):
    if '<!-- BEGIN TISSUE CARDS -->' in content:
        return content
    cards = []
    for record in records:
        if record['section'] != collection:
            continue
        details = disclosure = ''
        if collection == 'visium':
            slug = record['id'].removeprefix('visium-')
            match = re.search(r'<section id="' + slug + r'" class="tissue">(.*?)</section>', content, re.S)
            if match is None:
                raise ValueError('Missing expression sample details: ' + slug)
            paragraphs = re.findall(r'<p\b.*?</p>', match.group(1), re.S)
            for paragraph in paragraphs:
                if f'href="{slug}.html"' in paragraph:
                    continue
                if 'Coverage limitation:' in paragraph:
                    disclosure += paragraph
                else:
                    details += paragraph
            # Retain the original output-mask preview inside the sample details.
            details += ''.join(re.findall(r'<figure\b.*?</figure>', match.group(1), re.S))
        cards.append(card(record, details, disclosure))
    block = '<!-- BEGIN TISSUE CARDS --><div class="tissue-cards">' + ''.join(cards) + '</div><!-- END TISSUE CARDS -->'
    pattern = r'<div class="tissues">.*</div>' if collection == 'visium-hd' else r'<div class="sample-grid">.*</section></div>'
    result, count = re.subn(pattern, lambda _: block, content, count=1, flags=re.S)
    if count != 1:
        raise ValueError('Missing tissue selection grid: ' + collection)
    return result


def apply(site):
    records = json.loads((site / 'geojson-inspectors.json').read_text())['records']
    pages = [('public', None, site / 'index.html')]
    for section, tissues in TISSUES.items():
        pages.append((section, None, site / section / 'index.html'))
        pages.extend((section, tissue, site / section / (tissue + '.html')) for tissue in tissues)
    for collection, tissue, page in pages:
        prefix = '' if collection == 'public' else '../'
        content = page.read_text()
        content = re.sub(r'<style\b[^>]*>.*?</style>', '', content, flags=re.S)
        # Reuse one base stylesheet and one shared template, keeping scale/inspector styles.
        styles = re.findall(r'<link\b[^>]*rel="stylesheet"[^>]*>', content)
        content = re.sub(r'<link\b[^>]*rel="stylesheet"[^>]*>', '', content)
        retained = [link for link in styles if not re.search(r'href="(?:\.\./)?(?:viewer|benchmark-template)\.css"', link)]
        links = f'<link rel="stylesheet" href="{prefix}viewer.css">' + ''.join(retained)
        links += f'<link rel="stylesheet" href="{prefix}benchmark-template.css">'
        content = content.replace('</head>', links + '</head>', 1)
        if collection == 'visium':
            # Section builders emitted duplicate script tags; one listener per download.
            content = re.sub(r'<script\b[^>]*src="geojson-download\.js"[^>]*></script>', '', content)
            content = content.replace('</head>', '<script src="geojson-download.js" defer></script></head>', 1)
        content = re.sub(r'<body\b([^>]*)>', lambda m: '<body' +
                         re.sub(r'\s(?:class|data-collection)="[^"]*"', '', m.group(1)) +
                         f' class="benchmark-page" data-collection="{collection}">', content, count=1)
        content = re.sub(r'<a class="skip-link"[^>]*>.*?</a>', '', content)
        content = re.sub(r'<header\b.*?</header>', lambda _: '<a class="skip-link" href="#main-content">Skip to results</a>' + header(collection, prefix), content, count=1, flags=re.S)
        title = COLLECTIONS[collection][0] if tissue is None else TISSUES[collection][tissue]
        content = re.sub(r'<title>.*?</title>', lambda _: '<title>' + html.escape(title) + ' · Bora benchmark viewer</title>', content, count=1, flags=re.S)
        content = re.sub(r'<main\b[^>]*>', '<main id="main-content">', content, count=1)
        content = re.sub(r'<!-- BEGIN COLLECTION NAV -->.*?<!-- END COLLECTION NAV -->', '', content, flags=re.S)
        content = re.sub(r'<nav class="jump">.*?</nav>', '', content, flags=re.S)
        local = ''
        if collection != 'public':
            local = tissue_nav(collection, tissue)
            if tissue is None:
                content = re.sub(r'<!-- BEGIN INSPECTOR LINKS -->.*?<!-- END INSPECTOR LINKS -->', '', content, flags=re.S)
                local += '<div class="collection-intro"><h2>Browse results</h2><p>Choose a tissue to view the complete result, inspect its GeoJSON and explore all nine zooms.</p></div>'
                content = landing_cards(content, collection, records)
            elif collection == 'visium-hd':
                local += f'<h2 class="tissue-title">{html.escape(title)}</h2>'
            content = content.replace('<main id="main-content">', '<main id="main-content"><!-- BEGIN COLLECTION NAV -->' + local + '<!-- END COLLECTION NAV -->', 1)
        content = re.sub(r'<footer\b.*?</footer>', '', content, flags=re.S)
        content = content.replace('</body>', footer(prefix) + '</body>', 1)
        page.write_text('\n'.join(line.rstrip() for line in content.splitlines()) + '\n')
    print(json.dumps(dict(template_pages=len(pages),collections=list(COLLECTIONS),scientific_assets_modified=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=Path(__file__).resolve().parents[1] / 'site')
    apply(parser.parse_args().site)
