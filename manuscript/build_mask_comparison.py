"""Draw registered, semi-transparent reference/tile/Bora comparisons.

The main figure uses three independent tissue sources. Each row shows the same
coordinates in every column. No smoothing or resampling is applied to labels.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import tifffile


DATA_ROOT = Path(os.environ.get('BORA_DATA_ROOT', '/mnt/sata_ssd/Bora-benchmark'))
BORA_ROOT = Path(os.environ.get('BORA_ROOT', Path(__file__).resolve().parents[2] / 'Bora'))
PILOT = DATA_ROOT / 'results/bora-pilot-independent-2026-10-16'
COLOURS = ('#008B8B', '#E46C38', '#9163B6', '#35A76E', '#D9A227', '#4363D8', '#E05E8B')
NAMES = {
    'bcss': 'BCSS · breast',
    'tiger': 'TIGER · breast',
    'wsss4luad': 'WSSS4LUAD · lung',
}
LABELS = {
    'bcss': ('tumour', 'stroma', 'necrosis/debris'),
    'tiger': ('invasive tumour', 'tumour stroma', 'healthy glands', 'inflamed stroma', 'rest'),
    'wsss4luad': ('tumour', 'tumour stroma'),
}


def inner_border(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, np.uint8)
    v = (mask[1:] != mask[:-1]) & (mask[1:] > 0) & (mask[:-1] > 0)
    out[1:] |= v
    out[:-1] |= v
    v = (mask[:, 1:] != mask[:, :-1]) & (mask[:, 1:] > 0) & (mask[:, :-1] > 0)
    out[:, 1:] |= v
    out[:, :-1] |= v
    return out


def choose_crop(truth: np.ndarray, coarse: np.ndarray, refined: np.ndarray,
                width: int = 672, height: int = 560) -> tuple[slice, slice]:
    h, w = truth.shape
    if h <= height or w <= width:
        return slice(0, h), slice(0, w)
    edge = inner_border(truth)
    changed = (coarse != refined).astype(np.uint8)
    rank = edge * 4 + changed
    integral = np.pad(rank.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    best = (-1, 0, 0)
    for y in range(0, h - height + 1, 56):
        for x in range(0, w - width + 1, 56):
            score = (integral[y + height, x + width] - integral[y, x + width]
                     - integral[y + height, x] + integral[y, x])
            if score > best[0]:
                best = (score, y, x)
    _, y, x = best
    return slice(y, y + height), slice(x, x + width)


def overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = .42) -> np.ndarray:
    base = rgb.astype(np.float32) / 255
    out = base.copy()
    for value in np.unique(mask):
        if value == 0:
            continue
        colour = np.array(matplotlib.colors.to_rgb(COLOURS[(int(value) - 1) % len(COLOURS)]))
        sel = mask == value
        out[sel] = base[sel] * (1 - alpha) + colour * alpha
    return out


def build(tile_size: int, output: Path, pdf: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(17.2, 12.7), dpi=240,
                             gridspec_kw={'hspace': .47, 'wspace': .035})
    fig.patch.set_facecolor('white')
    headers = ('Reference annotation', f'{tile_size} × {tile_size} tile mask', 'Bora refined mask')
    coordinates = {}
    for row, dataset in enumerate(NAMES):
        folder = PILOT / dataset
        image = tifffile.imread(folder / 'image.tif')
        truth = tifffile.imread(folder / 'normalized_truth.tif')
        case = folder / f'error_0/tile_{tile_size}'
        coarse = tifffile.imread(case / 'coarse.tif')
        refined = tifffile.imread(case / 'refined.ome.tif', key=0)
        ys, xs = choose_crop(truth, coarse, refined)
        coordinates[dataset] = {'x': xs.start, 'y': ys.start,
                                'width': xs.stop - xs.start, 'height': ys.stop - ys.start}
        image = image[ys, xs]
        for col, mask in enumerate((truth[ys, xs], coarse[ys, xs], refined[ys, xs])):
            ax = axes[row, col]
            ax.imshow(overlay(image, mask), interpolation='nearest')
            ax.set_xlim(-.5, image.shape[1] - .5)
            ax.set_ylim(image.shape[0] - .5, -.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#5c6470')
                spine.set_linewidth(.65)
            if row == 0:
                ax.set_title(headers[col], fontsize=13, fontweight='semibold', pad=13)
            if col == 0:
                ax.set_ylabel(NAMES[dataset], fontsize=12, fontweight='semibold', labelpad=14)
            ax.add_patch(Rectangle((8, image.shape[0] - 24), tile_size, 3,
                                   color='black', zorder=10))
        legend = [Patch(facecolor=COLOURS[i], alpha=.65, label=name)
                  for i, name in enumerate(LABELS[dataset])]
        axes[row, 1].legend(handles=legend, loc='upper center',
                            bbox_to_anchor=(.5, -.035), ncol=min(len(legend), 3),
                            frameon=False, fontsize=9, columnspacing=1.25,
                            handlelength=1.1, borderpad=0)
    fig.text(.5, .018,
             'Identical coordinates across each row · H&E under 42% opacity categorical masks · black bar = one tile width',
             ha='center', va='bottom', fontsize=10, color='#30343b')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps({'tile_size': tile_size, 'rows': coordinates,
                                                       'mask_alpha': .42}, indent=2))
    print(output)
    print(pdf)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tile-size', type=int, default=56, choices=(56, 90, 216))
    p.add_argument('--output', type=Path, default=BORA_ROOT / 'docs/figures/multi_tissue_mask_comparison.png')
    p.add_argument('--pdf', type=Path, default=BORA_ROOT / 'output/pdf/Bora_multi_tissue_mask_comparison.pdf')
    args = p.parse_args()
    build(args.tile_size, args.output, args.pdf)


if __name__ == '__main__':
    main()
