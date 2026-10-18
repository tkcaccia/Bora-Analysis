"""Draw a native-resolution tumor–stroma boundary audit from saved masks.

The input key decodes the expert-origin coarse map for a diagnostic comparison;
it must never be interpreted as a deployable tile classifier.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tifffile

from benchmark.analyze_cmpb_boundaries import tumor_stroma_edge


def build(case_dir: Path, ontology: dict[str, int], output: Path, mpp: float = 0.5) -> None:
    image = np.asarray(Image.open(case_dir / 'original.native.png').convert('RGB'))
    raw = tifffile.imread(case_dir / 'reference.native.tif')
    truth = np.zeros(raw.shape, np.uint8)
    for value, role in ontology.items():
        truth[raw == int(value)] = int(role)
    final = tifffile.imread(case_dir / 'mask.tif')
    coarse = tifffile.imread(case_dir / 'coarse.working.tif')
    key = json.loads((case_dir / 'key.json').read_text())
    coarse_roles = np.zeros(coarse.shape, np.uint8)
    for anonymous, source in zip(key['input_anonymous_ids'], key['input_source_ids']):
        coarse_roles[coarse == anonymous] = int(ontology.get(str(source), 0))
    if coarse_roles.shape != truth.shape:
        coarse_roles = np.asarray(Image.fromarray(coarse_roles).resize((truth.shape[1], truth.shape[0]), Image.Resampling.NEAREST))
    assert image.shape[:2] == truth.shape == final.shape == coarse_roles.shape
    valid = truth > 0
    edges = [tumor_stroma_edge(x, valid) for x in (truth, coarse_roles, final)]
    # Center the crop on the reference interface rather than selecting by model quality.
    side = min(480, *truth.shape)
    from scipy.ndimage import uniform_filter
    density = uniform_filter(edges[0].astype(np.float32), size=side, mode='constant')
    y, x = np.unravel_index(np.argmax(density), density.shape)
    y0 = int(np.clip(y - side // 2, 0, truth.shape[0] - side))
    x0 = int(np.clip(x - side // 2, 0, truth.shape[1] - side))
    rgb = image[y0:y0+side, x0:x0+side]
    titles = ['Original H&E', 'Expert interface', 'Coarse interface*', 'Bora interface']
    colors = [None, '#ffe100', '#00e5ff', '#ff2ac3']
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), layout='constrained')
    for i, ax in enumerate(axes):
        ax.imshow(rgb)
        if i:
            ax.contour(edges[i-1][y0:y0+side, x0:x0+side], levels=[0.5], colors=[colors[i]], linewidths=1.1)
        ax.set_title(titles[i], fontsize=10)
        ax.axis('off')
        bar = 50 / mpp
        ax.plot([18, 18+bar], [side-20, side-20], color='white', lw=4, solid_capstyle='butt')
        ax.text(18, side-27, '50 µm', color='white', fontsize=8, weight='bold')
    fig.text(.5, .015, '*Coarse class identities are decoded using the expert-origin key solely for this diagnostic.',
             ha='center', va='bottom', fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-dir', type=Path, required=True)
    p.add_argument('--ontology-json', required=True, help='Raw expert values to tissue-role JSON object')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mpp', type=float, default=0.5)
    a = p.parse_args()
    build(a.case_dir, json.loads(a.ontology_json), a.output, a.mpp)
