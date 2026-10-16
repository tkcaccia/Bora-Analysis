"""Plot source-group changes for the BCSS and TIGER descriptive expansions."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DATA_ROOT = Path('/mnt/sata_ssd/Bora-benchmark/results')
from benchmark.runtime import BORA_ROOT
OUTPUT = BORA_ROOT / 'docs/figures/breast_extension_changes.png'


def main() -> None:
    datasets = [('BCSS', 'bcss', '#276d89'), ('TIGER', 'tiger', '#ae4d42')]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.65), sharex=True)
    for title, dataset, color in datasets:
        path = DATA_ROOT / f'bora-{dataset}-expansion-2026-10-16/summary.json'
        rows = json.loads(path.read_text())['rows']
        groups = sorted({row['group_id'] for row in rows})
        for ax, key_before, key_after, label in [
            (axes[0], 'macro_iou_coarse', 'macro_iou_bora', 'Macro IoU change'),
            (axes[1], 'f1_8_coarse', 'f1_8_bora', 'Class-pair F1 change (8 px)')]:
            for group in groups:
                subset = sorted((r for r in rows if r['group_id'] == group),
                                key=lambda r: r['tile_size'])
                x = [r['tile_size'] for r in subset]
                y = [r[key_after] - r[key_before] for r in subset]
                ax.plot(x, y, color=color, alpha=.23, lw=.85, marker='o', ms=2.5)
            sizes = (56, 90, 216)
            medians = [median(r[key_after] - r[key_before] for r in rows
                              if r['tile_size'] == size) for size in sizes]
            ax.plot(sizes, medians, color=color, lw=2.2, marker='o', ms=5.5,
                    label=f'{title} median (n={len(groups)})')
            ax.set_ylabel(label)
    for ax in axes:
        ax.axhline(0, color='#333333', lw=.8, ls='--')
        ax.set_xticks((56, 90, 216))
        ax.set_xlabel('Tile width (pixels)')
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False, loc='upper right', fontsize=8)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=260, facecolor='white')
    print(OUTPUT)


if __name__ == '__main__':
    main()
