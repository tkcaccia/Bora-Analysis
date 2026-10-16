"""Two-tissue exploratory atlas: reference, tile mask, Bora mask."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import tifffile

from benchmark.runtime import BORA_ROOT
from manuscript.build_mask_comparison import DATA_ROOT, overlay, choose_crop


RESULTS = DATA_ROOT / 'results/bora-additional-binary-2026-10-16'
CAM = DATA_ROOT / 'results/bora-camelyon-windows-2026-10-16'
OUTPUT = BORA_ROOT / 'output/pdf/Bora_additional_datasets_atlas.pdf'
TARGETS = (
    ('rings-tumor', 'RINGS · prostate tumour'),
    ('camelyon16-subset', 'CAMELYON16 · lymph node tumour'),
)


def binary_crop(truth: np.ndarray, width: int = 560, height: int = 560) -> tuple[slice, slice]:
    h, w = truth.shape
    width, height = min(width, w), min(height, h)
    if h == height and w == width:
        return slice(0, h), slice(0, w)
    a = np.zeros(truth.shape, dtype=np.int64)
    a[:-1] += (truth[:-1] != truth[1:]) * 50
    a[:, :-1] += (truth[:, :-1] != truth[:, 1:]) * 50
    a += (truth > 0)
    integ = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    best = (-1, 0, 0)
    for y in sorted(set(range(0, h - height + 1, 56)) | {h - height}):
        for x in sorted(set(range(0, w - width + 1, 56)) | {w - width}):
            score = (integ[y + height, x + width] - integ[y, x + width]
                     - integ[y + height, x] + integ[y, x])
            if score > best[0]:
                best = (score, y, x)
    return slice(best[1], best[1] + height), slice(best[2], best[2] + width)


def read_binary(dataset: str) -> tuple[np.ndarray, tuple[np.ndarray, ...], str]:
    sample = json.loads((RESULTS / dataset / 'selection.json').read_text())[0]
    case = RESULTS / dataset / sample['id']
    image = tifffile.imread(case / 'image.tif')
    truth = tifffile.imread(case / 'binary_truth.tif')
    coarse = tifffile.imread(case / 'tile_56/coarse.tif')
    refined = tifffile.imread(case / 'tile_56/refined.ome.tif', key=0)
    ys, xs = binary_crop(truth)
    return image[ys, xs], (truth[ys, xs], coarse[ys, xs], refined[ys, xs]), sample['id']


def read_cam() -> tuple[np.ndarray, tuple[np.ndarray, ...], str]:
    sample = json.loads((CAM / 'selection.json').read_text())[0]
    case = CAM / sample['group_id'] / 'camelyon16-subset'
    image = tifffile.imread(case / 'image.tif')
    truth = tifffile.imread(case / 'normalized_truth.tif')
    coarse = tifffile.imread(case / 'error_0/tile_56/coarse.tif')
    refined = tifffile.imread(case / 'error_0/tile_56/refined.ome.tif', key=0)
    ys, xs = choose_crop(truth, coarse, refined, width=560, height=560)
    return image[ys, xs], (truth[ys, xs], coarse[ys, xs], refined[ys, xs]), sample['group_id']


def draw_row(axes, image: np.ndarray, masks: tuple[np.ndarray, ...],
             name: str, sample: str) -> None:
    for col, (ax, labels) in enumerate(zip(axes, masks)):
        ax.imshow(overlay(image, labels), interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#6f7780')
            spine.set_linewidth(.65)
        if col == 0:
            ax.set_ylabel(name, fontsize=10.5, fontweight='semibold', labelpad=10)
        ax.plot([10, 66], [image.shape[0] - 16] * 2, color='black', lw=3,
                solid_capstyle='butt')
    axes[1].text(.5, -.055, sample[:65], ha='center', va='top',
                 transform=axes[1].transAxes, fontsize=7.6, color='#4b5159')


def page(pdf: PdfPages, rows: list[tuple[str, str]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.8), dpi=180,
                             gridspec_kw={'hspace': .27, 'wspace': .04})
    fig.patch.set_facecolor('white')
    for ax, title in zip(axes[0], ('Reference annotation', '56 × 56 tile mask', 'Bora refined mask')):
        ax.set_title(title, fontsize=12, fontweight='semibold', pad=10)
    for i in range(2):
        dataset, name = rows[i]
        image, masks, sample = read_cam() if dataset == 'camelyon16-subset' else read_binary(dataset)
        draw_row(axes[i], image, masks, name, sample)
    fig.text(.5, .019,
             'Same coordinates across each row · H&E beneath 42% opacity masks · black bar = 56 px',
             ha='center', fontsize=9)
    pdf.savefig(fig, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT) as pdf:
        page(pdf, list(TARGETS))
    print(OUTPUT)


if __name__ == '__main__':
    main()
