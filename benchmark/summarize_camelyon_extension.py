"""Summarize three native CAMELYON16 tumour windows, not complete WSIs."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import tifffile

from benchmark.reanalyze_bora_pilot import pair_metrics
from benchmark.run_camelyon_extension import ROOT


def main() -> None:
    selected = json.loads((ROOT / 'selection.json').read_text())
    rows = []
    for item in selected:
        sample = ROOT / item['group_id'] / 'camelyon16-subset'
        truth = tifffile.imread(sample / 'normalized_truth.tif')
        for size in (56, 90, 216):
            case = sample / 'error_0' / f'tile_{size}'
            record = json.loads((case / 'result.json').read_text())
            coarse = tifffile.imread(case / 'coarse.tif')
            refined = tifffile.imread(case / 'refined.ome.tif').squeeze()
            before, after = pair_metrics(coarse, truth), pair_metrics(refined, truth)
            rows.append({'source_slide': item['group_id'], 'tile_size': size,
                         'macro_iou_coarse': record['coarse']['macro_iou'],
                         'macro_iou_bora': record['refined']['macro_iou'],
                         'pixel_accuracy_coarse': record['coarse']['pixel_accuracy'],
                         'pixel_accuracy_bora': record['refined']['pixel_accuracy'],
                         'f1_8_coarse': before['f1_8'], 'f1_8_bora': after['f1_8'],
                         'assd_coarse_px': before['assd_px_observed_pairs'],
                         'assd_bora_px': after['assd_px_observed_pairs'],
                         'hd95_coarse_px': before['hd95_px_observed_pairs'],
                         'hd95_bora_px': after['hd95_px_observed_pairs'],
                         'retained_classes_coarse': record['coarse']['retained_classes'],
                         'retained_classes_bora': record['refined']['retained_classes'],
                         'wall_seconds': record['bora_wall_seconds_including_vectorization'],
                         'geojson_seconds': record['geojson_conversion_seconds'],
                         'geojson_pixel_agreement': record['geojson']['pixel_agreement'],
                         'repeat_identical': record['geojson_repeat']['byte_reproducible']})
    by_size = {}
    for size in (56, 90, 216):
        subset = [r for r in rows if r['tile_size'] == size]
        delta = [r['macro_iou_bora'] - r['macro_iou_coarse'] for r in subset]
        f1 = [r['f1_8_bora'] - r['f1_8_coarse'] for r in subset
              if r['f1_8_bora'] is not None and r['f1_8_coarse'] is not None]
        by_size[str(size)] = {'n_windows': len(subset),
                              'median_macro_iou_delta': median(delta),
                              'macro_iou_improved': sum(x > 0 for x in delta),
                              'macro_iou_worsened': sum(x < 0 for x in delta),
                              'median_f1_8_delta': median(f1) if f1 else None,
                              'f1_improved': sum(x > 0 for x in f1),
                              'f1_worsened': sum(x < 0 for x in f1),
                              'median_wall_seconds': median(r['wall_seconds'] for r in subset),
                              'all_geojson_exact': all(r['geojson_pixel_agreement'] == 1 for r in subset),
                              'all_repeat_identical': all(r['repeat_identical'] for r in subset)}
    summary = {'scope': 'Three native 4096x4096 tumour windows from three acquired complete slides; not complete-WSI processing.',
               'annotation': 'Expert-derived tumour vs automatically detected normal tissue; 0 ignored.',
               'n_distinct_windows': len(selected), 'n_conditions': len(rows),
               'by_tile': by_size, 'rows': rows}
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
