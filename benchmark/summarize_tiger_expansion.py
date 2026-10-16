"""Summarize a completed source-group extension without rerunning Bora."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np
import tifffile

from benchmark.reanalyze_bora_pilot import pair_metrics
from benchmark.run_tiger_expansion import DEFAULT_OUTPUT


def main(root: Path = DEFAULT_OUTPUT, dataset: str = 'tiger') -> None:
    selection = json.loads((root / 'selection.json').read_text())
    rows = []
    for item in selection:
        if item['dataset'] != dataset:
            raise ValueError(f'Unexpected dataset in selection: {item["dataset"]}')
        sample = root / item['group_id'] / dataset
        truth = tifffile.imread(sample / 'normalized_truth.tif')
        for size in (56, 90, 216):
            case = sample / 'error_0' / f'tile_{size}'
            report = json.loads((case / 'result.json').read_text())
            coarse = tifffile.imread(case / 'coarse.tif')
            refined = tifffile.imread(case / 'refined.ome.tif').squeeze()
            before, after = pair_metrics(coarse, truth), pair_metrics(refined, truth)
            geojson = case / 'refined.geojson'
            if not geojson.is_file() or not report['geojson_repeat']['byte_reproducible']:
                raise ValueError(f'Incomplete vector verification: {case}')
            rows.append({'group_id': item['group_id'], 'tile_size': size,
                         'pixel_accuracy_coarse': report['coarse']['pixel_accuracy'],
                         'pixel_accuracy_bora': report['refined']['pixel_accuracy'],
                         'macro_iou_coarse': report['coarse']['macro_iou'],
                         'macro_iou_bora': report['refined']['macro_iou'],
                         'f1_8_coarse': before['f1_8'], 'f1_8_bora': after['f1_8'],
                         'f1_16_coarse': before['f1_16'], 'f1_16_bora': after['f1_16'],
                         'f1_32_coarse': before['f1_32'], 'f1_32_bora': after['f1_32'],
                         'assd_coarse_px': before['assd_px_observed_pairs'],
                         'assd_bora_px': after['assd_px_observed_pairs'],
                         'hd95_coarse_px': before['hd95_px_observed_pairs'],
                         'hd95_bora_px': after['hd95_px_observed_pairs'],
                         'classes_coarse': report['coarse']['retained_classes'],
                         'classes_bora': report['refined']['retained_classes'],
                         'same_class_ids_as_coarse': bool(np.array_equal(
                             np.unique(coarse[truth > 0]), np.unique(refined[truth > 0]))),
                         'bora_wall_seconds': report['bora_wall_seconds_including_vectorization'],
                         'geojson_seconds': report['geojson_conversion_seconds'],
                         'geojson_bytes': geojson.stat().st_size,
                         'geojson_pixel_agreement': report['geojson']['pixel_agreement']})
    by_size = {}
    for size in (56, 90, 216):
        subset = [r for r in rows if r['tile_size'] == size]
        delta = [r['macro_iou_bora'] - r['macro_iou_coarse'] for r in subset]
        f1 = [r['f1_8_bora'] - r['f1_8_coarse'] for r in subset
              if r['f1_8_bora'] is not None and r['f1_8_coarse'] is not None]
        by_size[str(size)] = {'n_groups': len(subset), 'median_macro_iou_delta': median(delta),
                              'median_pixel_accuracy_delta': median(
                                  r['pixel_accuracy_bora'] - r['pixel_accuracy_coarse'] for r in subset),
                              'macro_iou_improved': sum(x > 0 for x in delta),
                              'macro_iou_worsened': sum(x < 0 for x in delta),
                              'n_defined_f1': len(f1), 'median_f1_8_delta': median(f1) if f1 else None,
                              'f1_improved': sum(x > 0 for x in f1),
                              'f1_worsened': sum(x < 0 for x in f1),
                              'median_wall_seconds': median(r['bora_wall_seconds'] for r in subset),
                              'median_geojson_seconds': median(r['geojson_seconds'] for r in subset),
                              'all_roundtrip_exact': all(r['geojson_pixel_agreement'] == 1 for r in subset),
                              'all_classes_preserved_by_bora': all(r['same_class_ids_as_coarse'] for r in subset)}
    summary = {'selection': ('First ten inventory-ordered eligible non-TCGA TIGER tissue-cells source groups not used in the prior pilot or breast extension; at most 3 million pixels and at least two observed valid classes.'
                             if dataset == 'tiger' else
                             'First eight inventory-ordered eligible BCSS source groups not used in the prior pilot or breast extension; at most 5 million pixels and at least two observed valid classes.'),
               'conditions': '56, 90 and 216 pixels; zero requested border error; frozen adaptive-wand settings',
               'n_distinct_images': len(selection), 'n_results': len(rows), 'by_tile': by_size, 'rows': rows}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--dataset', choices=('tiger', 'bcss'), default='tiger')
    args = parser.parse_args()
    main(args.root, args.dataset)
