"""Correct the exploratory tile-readout provenance without editing geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('geojson',type=Path)
    a=p.parse_args()
    path=a.geojson
    old=path.read_bytes()
    data=json.loads(old)
    assert len(data['features'])==2
    assert {int(f['properties']['value']) for f in data['features']}=={1,3}
    before=[f['geometry'] for f in data['features']]
    for feature in data['features']:
        source=feature['properties']['tissue_identity_source']
        assert source in {'Neighbor context, native fastPLS',
                          'Source-trained fastPLS tile predictions; area-majority per refined group'}
        feature['properties']['tissue_identity_source']='Source-trained fastPLS tile predictions; area-majority per refined group'
    assert before==[f['geometry'] for f in data['features']]
    backup=path.with_name('mask.pre-provenance-correction.geojson')
    if not backup.exists():
        backup.write_bytes(old)
    temporary=path.with_suffix('.geojson.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    os.replace(temporary,path)
    print(json.dumps(dict(old_sha256=hashlib.sha256(old).hexdigest(),
                          new_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          geometry_unchanged=True,corrected_features=len(data['features']))))


if __name__=='__main__':
    main()
