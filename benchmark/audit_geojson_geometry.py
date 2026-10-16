"""Audit feature validity, classes, holes and overlap in pilot GeoJSONs."""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

from shapely.geometry import shape

ROOT=Path('/mnt/sata_ssd/Bora-benchmark/results/bora-pilot-independent-2026-10-16')
OUT=ROOT/'reviewer_reanalysis'


def geometry_audit(path: Path) -> dict:
    obj=json.loads(path.read_text())
    features=obj['features']
    geometries=[shape(f['geometry']) for f in features]
    values=[int(f['properties']['value']) for f in features]
    holes=0
    for geometry in geometries:
        members=geometry.geoms if geometry.geom_type=='MultiPolygon' else [geometry]
        holes+=sum(len(poly.interiors) for poly in members if poly.geom_type=='Polygon')
    overlaps=[(values[i],values[j],float(geometries[i].intersection(geometries[j]).area))
              for i,j in combinations(range(len(geometries)),2)
              if values[i]!=values[j] and geometries[i].intersects(geometries[j])]
    return {'path':str(path),'features':len(features),'values':values,
            'all_valid':all(g.is_valid for g in geometries),
            'empty_geometries':sum(g.is_empty for g in geometries),
            'holes':holes,'max_cross_class_overlap_area':max((a for _,_,a in overlaps),default=0.0)}


def main() -> None:
    reports=[]
    for dataset in ('bcss','tiger','wsss4luad'):
        for error in (0,1,5):
            for tile in (56,90,216):
                path=ROOT/dataset/f'error_{error}'/f'tile_{tile}'/'refined.geojson'
                reports.append({'dataset':dataset,'error':error,'tile':tile,
                                'profile':'unsimplified',**geometry_audit(path)})
        reports.append({'dataset':dataset,'error':0,'tile':56,'profile':'accurate',
                        **geometry_audit(OUT/f'{dataset}_56_accurate.geojson')})
    (OUT/'geometry_audit.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps({'n':len(reports),'all_valid':all(r['all_valid'] for r in reports),
                      'max_overlap':max(r['max_cross_class_overlap_area'] for r in reports),
                      'total_empty':sum(r['empty_geometries'] for r in reports)},indent=2))

if __name__=='__main__':main()
