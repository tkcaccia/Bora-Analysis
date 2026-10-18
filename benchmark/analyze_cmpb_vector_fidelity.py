"""Summarize saved polygon-fitting diagnostics without rerunning Bora."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.results.read_text())['cases']
    records = []
    for case in cases:
        r = case['vector_record']['compact_boundary_report']
        records.append({
            'case': case['case'],
            'source_group': case['source_group'],
            'selected_tolerance_um': r['selected_tolerance_um'],
            'minimum_class_iou_vs_prefit_prediction': r['agreement']['minimum_class_iou'],
            'changed_tissue_area_fraction_vs_prefit_prediction': r['agreement']['changed_tissue_area_fraction'],
            'parts_before': r['before']['polygon_parts'],
            'parts_after': r['after']['polygon_parts'],
            'segments_before': r['before']['boundary_segments'],
            'segments_after': r['after']['boundary_segments'],
            'fallback': r['original_export_fallback'],
        })
    assert len(records) == 116
    summary = {
        'images': len(records),
        'mean_minimum_class_iou_vs_prefit_prediction': mean(r['minimum_class_iou_vs_prefit_prediction'] for r in records),
        'minimum_class_iou_vs_prefit_prediction': min(r['minimum_class_iou_vs_prefit_prediction'] for r in records),
        'mean_changed_tissue_area_fraction': mean(r['changed_tissue_area_fraction_vs_prefit_prediction'] for r in records),
        'maximum_changed_tissue_area_fraction': max(r['changed_tissue_area_fraction_vs_prefit_prediction'] for r in records),
        'parts_before': sum(r['parts_before'] for r in records),
        'parts_after': sum(r['parts_after'] for r in records),
        'images_with_lost_parts': sum(r['parts_after'] < r['parts_before'] for r in records),
        'segments_before': sum(r['segments_before'] for r in records),
        'segments_after': sum(r['segments_after'] for r in records),
        'fallbacks': sum(r['fallback'] for r in records),
        'comparison_caveat': 'Fitting diagnostics compare the export with the preceding predicted geometry, not expert contours.'
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'summary': summary, 'records': records}, indent=2) + '\n')


if __name__ == '__main__':
    main()
