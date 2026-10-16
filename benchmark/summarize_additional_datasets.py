"""Summarize the separate RINGS tumour-region endpoint by tile width."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median

from benchmark.run_additional_datasets import DATASETS, ROOT


def median_or_none(values) -> float | None:
    values = list(values)
    return median(values) if values else None


def main(root: Path = ROOT) -> None:
    summary = {'protocol': 'Three inventory-ordered RINGS tumour-mask source groups, 56/90/216 pixel reference-derived tiles, zero requested border error. Binary tumour vs supplied outside; not pooled with multiclass tissue results.',
               'datasets': {}}
    for dataset in DATASETS:
        folder = root / dataset
        selection = json.loads((folder / 'selection.json').read_text())
        if len(selection) != 3:
            raise ValueError(f'{dataset}: expected three selected records')
        records = []
        for item in selection:
            for size in (56, 90, 216):
                path = folder / item['id'] / f'tile_{size}/result.json'
                records.append(json.loads(path.read_text()))
        by_size = {}
        for size in (56, 90, 216):
            rows = [r for r in records if r['tile_size'] == size]
            diffs = [r['bora']['foreground_iou'] - r['coarse']['foreground_iou']
                     for r in rows if r['bora']['foreground_iou'] is not None
                     and r['coarse']['foreground_iou'] is not None]
            f1 = [r['bora']['boundary_f1']['8'] - r['coarse']['boundary_f1']['8']
                  for r in rows if r['bora']['boundary_f1']['8'] is not None
                  and r['coarse']['boundary_f1']['8'] is not None]
            by_size[str(size)] = {
                'n_pairs': len(rows), 'median_iou_delta': median(diffs) if diffs else None,
                'coarse_target_absent': sum(r['coarse']['foreground_pixels'] == 0 for r in rows),
                'bora_target_absent': sum(r['bora']['foreground_pixels'] == 0 for r in rows),
                'iou_improved': sum(x > 0 for x in diffs),
                'iou_worsened': sum(x < 0 for x in diffs),
                'median_dice_delta': median_or_none(r['bora']['foreground_dice'] -
                                                   r['coarse']['foreground_dice'] for r in rows
                                                   if r['bora']['foreground_dice'] is not None
                                                   and r['coarse']['foreground_dice'] is not None),
                'median_accuracy_delta': median(r['bora']['pixel_accuracy'] -
                                                r['coarse']['pixel_accuracy'] for r in rows),
                'median_boundary_f1_8_delta': median(f1) if f1 else None,
                'f1_improved': sum(x > 0 for x in f1),
                'f1_worsened': sum(x < 0 for x in f1),
                'median_assd_delta_px': median_or_none(r['bora']['assd_px'] - r['coarse']['assd_px']
                                                      for r in rows if r['bora']['assd_px'] is not None
                                                      and r['coarse']['assd_px'] is not None),
                'median_hd95_delta_px': median_or_none(r['bora']['hd95_px'] - r['coarse']['hd95_px']
                                                      for r in rows if r['bora']['hd95_px'] is not None
                                                      and r['coarse']['hd95_px'] is not None),
                'median_wall_seconds': median(r['wall_seconds'] for r in rows),
                'median_geojson_seconds': median(r['geojson_seconds'] for r in rows),
                'all_geojson_exact': all(r['geojson']['pixel_agreement'] == 1 for r in rows),
                'all_repeat_bytes_identical': all(r['geojson_repeat']['byte_reproducible']
                                                  for r in rows)}
        summary['datasets'][dataset] = {'n_images': len(selection),
                                        'n_conditions': len(records),
                                        'by_tile': by_size}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
