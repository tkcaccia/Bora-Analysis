"""Quantify raster fidelity of Bora's default accurate GeoJSON profile."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

from benchmark.runtime import require_bora
sys.path.insert(0, str(require_bora()))
from bora.multicpu_geojson import convert
from benchmark.run_bora_pilot import vector_roundtrip

ROOT = Path('/mnt/sata_ssd/Bora-benchmark/results/bora-pilot-independent-2026-10-16')
OUT = ROOT / 'reviewer_reanalysis'


def main() -> None:
    OUT.mkdir(exist_ok=True)
    results=[]
    for dataset in ('bcss','tiger','wsss4luad'):
        case=ROOT/dataset/'error_0'/'tile_56'
        raster=case/'refined.ome.tif'
        expected=np.squeeze(tifffile.imread(raster))
        production=OUT/f'{dataset}_56_accurate.geojson'
        started=time.perf_counter()
        report=convert(raster, production, page=0, min_area=500, simplify=2, workers=2)
        elapsed=time.perf_counter()-started
        audit=vector_roundtrip(production, expected.shape, expected)
        exact=vector_roundtrip(case/'refined.geojson', expected.shape, expected)
        results.append({'dataset':dataset,'tile':56,'profile':'accurate',
                        'conversion_wall_seconds':elapsed,'converter_report':report,
                        'raster_agreement':audit['pixel_agreement'],
                        'feature_count':audit['feature_count'],
                        'bytes':production.stat().st_size,
                        'unsimplified_feature_count':exact['feature_count'],
                        'unsimplified_bytes':(case/'refined.geojson').stat().st_size})
    (OUT/'accurate_profile_audit.json').write_text(json.dumps(results,indent=2))
    print(OUT/'accurate_profile_audit.json')

if __name__=='__main__':main()
