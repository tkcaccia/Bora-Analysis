"""Run the fixed marker watershed comparator on pilot zero-error inputs."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import tifffile

from benchmark.run_bora_pilot import PYTHON, SAMPLES, metrics
from benchmark.reanalyze_bora_pilot import ROOT, OUT, pair_metrics


def main() -> None:
    OUT.mkdir(exist_ok=True)
    results=[]
    for dataset in SAMPLES:
        sample=ROOT/dataset
        truth=tifffile.imread(sample/'normalized_truth.tif')
        for tile in (56,90,216):
            case=sample/'error_0'/f'tile_{tile}'
            target=OUT/f'{dataset}_{tile}_watershed.ome.tif'
            vector=OUT/f'{dataset}_{tile}_watershed.geojson'
            command=[str(PYTHON),'-m','bora.cli','refine',str(sample/'image.tif'),
                str(case/'coarse.tif'),'--output',str(target),'--geojson',str(vector),
                '--backend','watershed','--no-annealed-wand','--no-pyramid',
                '--geojson-profile','custom','--geojson-page','0',
                '--geojson-min-area','0','--geojson-simplify','0']
            if not target.exists():
                from benchmark.runtime import require_bora
                env=os.environ.copy();env['PYTHONPATH']=str(require_bora())
                started=time.perf_counter()
                subprocess.run(command,check=True,env=env,capture_output=True,text=True)
                elapsed=time.perf_counter()-started
            else:
                elapsed=None
            predicted=np.squeeze(tifffile.imread(target))
            results.append({'dataset':dataset,'tile':tile,'macro_iou':metrics(predicted,truth)['macro_iou'],
                'pair_boundary_f1_8':pair_metrics(predicted,truth)['f1_8'],
                'wall_seconds':elapsed,'command':command})
            (OUT/'watershed_comparator.json').write_text(json.dumps(results,indent=2))
            print(dataset,tile,results[-1]['macro_iou'],flush=True)

if __name__=='__main__':main()
