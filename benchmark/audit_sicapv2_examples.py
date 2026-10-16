"""Document three SICAPv2 examples without decoding JPEG masks as classes."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path('/mnt/sata_ssd/Bora-benchmark')
OUTPUT = ROOT / 'results/bora-sicapv2-encoding-audit-2026-10-16'
from benchmark.runtime import BORA_ROOT
FIGURE = BORA_ROOT / 'docs/figures/sicapv2_raw_examples.png'


def main() -> None:
    examples = json.loads((ROOT / 'examples/examples.json').read_text())
    rows = [row for row in examples if row['dataset'] == 'sicapv2']
    if len(rows) != 3:
        raise ValueError('Expected exactly three curated SICAPv2 examples')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(7.8, 10.3))
    records = []
    for i, row in enumerate(rows):
        image = np.asarray(Image.open(ROOT / 'examples' / row['image']).convert('RGB'))
        mask = np.asarray(Image.open(ROOT / 'examples' / row['mask']))
        if mask.ndim != 2 or image.shape[:2] != mask.shape:
            raise ValueError(f'Unexpected SICAP example geometry: {row["sample_id"]}')
        values, counts = np.unique(mask, return_counts=True)
        records.append({'sample_id': row['sample_id'], 'image': row['image'],
                        'mask': row['mask'], 'shape': list(mask.shape),
                        'mask_unique_intensities': len(values),
                        'mask_min': int(values.min()), 'mask_max': int(values.max()),
                        'most_common_intensities': [
                            {'value': int(values[j]), 'pixels': int(counts[j])}
                            for j in np.argsort(counts)[-5:][::-1]],
                        'interpretation': 'raw JPEG intensities only; categorical codes unresolved'})
        axes[i, 0].imshow(image)
        axes[i, 1].imshow(mask, cmap='gray', vmin=0, vmax=255)
        axes[i, 0].set_ylabel(f'Example {i+1}', rotation=0, labelpad=40, va='center')
        for ax in axes[i]:
            ax.axis('off')
    axes[0, 0].set_title('H&E image')
    axes[0, 1].set_title('Supplied raw JPEG mask')
    fig.suptitle('SICAPv2: visual audit only; mask codes unresolved', y=.995)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=190, facecolor='white')
    (OUTPUT / 'audit.json').write_text(json.dumps({
        'source_index': 'examples/examples.json', 'n_examples': len(records),
        'scoring_performed': False, 'reason': 'JPEG intensities do not safely map to tissue classes',
        'examples': records}, indent=2) + '\n')
    print(OUTPUT / 'audit.json')
    print(FIGURE)


if __name__ == '__main__':
    main()
