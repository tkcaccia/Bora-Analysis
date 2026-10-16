"""Make auditable tables and figures from completed Bora pilot records."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path('/mnt/sata_ssd/Bora-benchmark/results/bora-pilot-independent-2026-10-16')


def main() -> None:
    rows = []
    for file in sorted(ROOT.glob('*/error_*/tile_*/result.json')):
        item = json.loads(file.read_text())
        tiled = item['tiling']['border_error_tiles'][f"{item['tile_size']}x{item['tile_size']}"]
        rows.append({
            'dataset': item['dataset'], 'sample_id': item['sample_id'],
            'tile_px': item['tile_size'], 'error_requested_pct': item['error_percent'],
            'error_eligible_tiles': tiled['eligible'], 'error_swapped_tiles': tiled['swapped'],
            'baseline_macro_iou': item['coarse']['macro_iou'],
            'bora_macro_iou': item['refined']['macro_iou'],
            'baseline_pixel_accuracy': item['coarse']['pixel_accuracy'],
            'bora_pixel_accuracy': item['refined']['pixel_accuracy'],
            'baseline_internal_f1_8': item['coarse']['internal_boundary_f1']['8'],
            'bora_internal_f1_8': item['refined']['internal_boundary_f1']['8'],
            'baseline_internal_assd_px': item['coarse']['internal_assd_px'],
            'bora_internal_assd_px': item['refined']['internal_assd_px'],
            'baseline_internal_hd95_px': item['coarse']['internal_hd95_px'],
            'bora_internal_hd95_px': item['refined']['internal_hd95_px'],
            'tiling_total_seconds_all_sizes': item['tiling']['total_seconds'],
            'preparation_seconds': json.loads((ROOT / item['dataset'] / 'sample.json').read_text())['preparation_wall_seconds'],
            'preparation_timing_basis': json.loads((ROOT / item['dataset'] / 'sample.json').read_text())['preparation_timing_basis'],
            'grid_expansion_seconds': item['other_stage_timings']['grid_expansion_seconds'],
            'raster_scoring_seconds': item['other_stage_timings']['two_raster_scores_seconds'],
            'geojson_roundtrip_score_seconds': item['other_stage_timings']['geojson_roundtrip_score_seconds'],
            'ancillary_timing_basis': item['other_stage_timings']['measurement'],
            'bora_and_vector_seconds': item['bora_wall_seconds_including_vectorization'],
            'vector_seconds': item['geojson_conversion_seconds'],
            'repeat_vector_seconds': item['geojson_repeat']['repeat_seconds'],
            'vector_pixel_agreement': item['geojson']['pixel_agreement'],
            'vector_byte_reproducible': item['geojson_repeat']['byte_reproducible'],
            'geojson_macro_iou_vs_truth': item['refined']['macro_iou'] if item['geojson']['pixel_agreement'] == 1 else None,
        })
    if len(rows) != 27:
        raise ValueError(f'Expected 27 pilot conditions, got {len(rows)}')
    with (ROOT / 'pilot-results.csv').open('w', newline='') as handle:
        out = csv.DictWriter(handle, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    (ROOT / 'pilot-summary.json').write_text(json.dumps({'conditions': len(rows),
        'datasets': sorted({r['dataset'] for r in rows}),
        'all_vector_byte_reproducible': all(r['vector_byte_reproducible'] for r in rows),
        'minimum_vector_pixel_agreement': min(r['vector_pixel_agreement'] for r in rows),
        'scope': 'One selected ROI per dataset; descriptive pilot, not a population estimate.'}, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True, constrained_layout=True)
    titles = {'bcss': 'BCSS breast (0.25 µm/px)',
              'tiger': 'TIGER breast (~0.5 µm/px)',
              'wsss4luad': 'WSSS4LUAD lung (spacing unknown)'}
    for ax, dataset in zip(axes, titles):
        for error, color in ((0, '#146c94'), (1, '#ca6702'), (5, '#ae2012')):
            sample = sorted((r for r in rows if r['dataset'] == dataset and
                             r['error_requested_pct'] == error), key=lambda r: r['tile_px'])
            sizes = [r['tile_px'] for r in sample]
            ax.plot(sizes, [r['baseline_macro_iou'] for r in sample], '--o',
                    color=color, alpha=.55, linewidth=1.4)
            ax.plot(sizes, [r['bora_macro_iou'] for r in sample], '-o',
                    color=color, linewidth=2, label=f'{error}% errors')
        ax.set_title(titles[dataset], fontsize=10)
        ax.set_xlabel('Tile width (pixels)')
        ax.set_xticks([56, 90, 216])
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Macro IoU (solid Bora; dashed tile baseline)')
    axes[-1].legend(frameon=False, fontsize=8)
    fig.savefig(ROOT / 'pilot-macro-iou.png', dpi=180)
    plt.close(fig)
    label_names = {'bcss': ['Tumour', 'Stroma', 'Necrosis/debris'],
                   'tiger': ['Invasive tumour', 'Tumour stroma', 'Healthy glands', 'Inflamed stroma', 'Rest'],
                   'wsss4luad': ['Tumour', 'Stroma']}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True, sharey=True)
    for ax, dataset in zip(axes, titles):
        full = json.loads((ROOT / dataset / 'error_0' / 'tile_56' / 'result.json').read_text())
        names = label_names[dataset]
        x = np.arange(len(names))
        baseline = [full['coarse']['per_class_iou'][str(i)] for i in range(1, len(names)+1)]
        refined = [full['refined']['per_class_iou'][str(i)] for i in range(1, len(names)+1)]
        ax.bar(x-.18, baseline, width=.35, label='Tile baseline', color='#94a3b8')
        ax.bar(x+.18, refined, width=.35, label='Bora', color='#146c94')
        ax.set_xticks(x, names, rotation=20, ha='right', fontsize=8)
        ax.set_title(titles[dataset], fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Class IoU at 56 px, 0% error')
    axes[-1].legend(frameon=False, fontsize=8)
    fig.savefig(ROOT / 'pilot-class-iou.png', dpi=180)
    plt.close(fig)
    print(ROOT / 'pilot-results.csv')
    print(ROOT / 'pilot-macro-iou.png')


if __name__ == '__main__':
    main()
