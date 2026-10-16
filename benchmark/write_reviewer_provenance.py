"""Record the exact local code and hardware context for reviewer extensions."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import cv2
import numpy
import rasterio
import scipy
import shapely
import tifffile
from benchmark.runtime import BORA_ROOT

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/mnt/sata_ssd/Bora-benchmark/results')
TILER=Path('/home/chiamaka/.local/bin/tiling-mask-cpp')
FILES=['src/bora/boundary.py','src/bora/cli.py','src/bora/multicpu_geojson.py',
       'benchmark/run_bora_pilot.py','benchmark/reanalyze_bora_pilot.py',
       'benchmark/run_breast_extension.py','benchmark/summarize_wsss_validation.py',
       'benchmark/summarize_breast_extension.py','benchmark/audit_geojson_profile.py',
       'benchmark/audit_geojson_geometry.py','benchmark/audit_watershed_comparator.py',
       'benchmark/plot_reviewer_extension.py']


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cpu='unknown'
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu=line.split(':',1)[1].strip();break
    except OSError:pass
    manifest={'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'git_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
      'sha256_by_path':{name:sha((BORA_ROOT if name.startswith('src/') else ROOT)/name)
                        for name in FILES},
      'bora_git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=BORA_ROOT,text=True).strip(),
      'tiling_mask_executable_sha256':sha(TILER),
      'hardware':{'cpu_model':cpu,'logical_cpu_count':__import__('os').cpu_count(),
                  'platform':platform.platform()},
      'python_packages':{'numpy':numpy.__version__,'scipy':scipy.__version__,
          'opencv':cv2.__version__,'rasterio':rasterio.__version__,
          'tifffile':tifffile.__version__,'shapely':shapely.__version__}}
    for folder in ('bora-pilot-independent-2026-10-16','bora-wsss-validation-2026-10-16',
                   'bora-breast-extension-2026-10-16'):
        path=DATA/folder/'reviewer_provenance.json'
        path.write_text(json.dumps(manifest,indent=2))
        print(path)

if __name__=='__main__':main()
