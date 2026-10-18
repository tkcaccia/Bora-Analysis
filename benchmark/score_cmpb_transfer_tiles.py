"""Score already-frozen source-trained tile maps; never fit or infer here."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry', type=Path, required=True)
    p.add_argument('--tiles', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    registry = {r['case']:r for r in json.loads(a.registry.read_text())['cases']}
    frozen = json.loads((a.tiles/'PREDICTIONS_FROZEN.json').read_text())
    records=[]
    for r in frozen['records']:
        row=registry[r['case']]
        path=a.tiles/r['case']/'anonymous_tiles.npy'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['tile_file_sha256']
        labels=np.load(path,allow_pickle=False)
        reference=tifffile.imread(row['reference_path'])
        truth=np.zeros(reference.shape,np.uint8)
        for raw,role in row['ontology_raw_to_role'].items():
            truth[reference==int(raw)]=int(role)
        coarse=np.repeat(np.repeat(labels,90,axis=0),90,axis=1)[:truth.shape[0],:truth.shape[1]]
        pred=np.where(coarse==1,1,np.where(coarse==2,3,0)).astype(np.uint8)
        valid=truth>0
        roles=sorted(set(np.unique(truth[valid]))|set(np.unique(pred[valid])))
        scores={str(int(role)):float(np.sum(valid&(truth==role)&(pred==role))/np.sum(valid&((truth==role)|(pred==role))))
                for role in roles if role}
        records.append({'case':r['case'],'source_group':r['source_group'],
                        'macro_iou_all_present_roles':float(np.mean(list(scores.values()))),
                        'role_iou':scores,'valid_pixels':int(valid.sum()),
                        'predicted_groups':r['predicted_groups']})
    output={'source_case':frozen['source_case'],'source_group':frozen['source_group'],
            'source_training_tiles':frozen['training_tiles'],'targets':len(records),
            'mean_coarse_macro_iou':float(np.mean([r['macro_iou_all_present_roles'] for r in records])),
            'scope':'Frozen source-trained tile maps only; no Bora refinement or vector output.',
            'records':records}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))


if __name__=='__main__':main()
