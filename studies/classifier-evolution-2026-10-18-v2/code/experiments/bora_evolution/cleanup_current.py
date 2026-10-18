"""Version the accepted viewer with native-pixel microscopic-region filling."""
import argparse
from pathlib import Path
from threadpoolctl import threadpool_limits
from experiments.bora_group import bora_geojson,viewer
from experiments.bora_supervised.registry import save,sha

def run(study):
    study=Path(study);base=study.parent;group=base/'bora-anonymous-groups-2026-10-18-v1'
    registry=base/'tissue-supervised-2026-10-18-v1/registry-v1';cycle=group/'classifiers-test-v1'
    vectors=group/'geojson-bora-fill20-v5';output=group/'viewer-bora-fill20-v5'
    bora_geojson.run(registry,group,cycle,vectors,fill_below_pixels=20)
    viewer.run(registry,group,cycle,vectors,output)
    html=(output/'index.html').read_text().replace('<h1>Anonymous tile groups → Bora → tissue identity</h1>',
        '<h1>Anonymous tile groups → Bora → tissue identity</h1><p><b>Updated cleanup:</b> connected tissue regions and enclosed holes with area below 20 native pixels are filled from their surroundings. Exterior background is preserved. Exactly 20 pixels is retained. The published Bora accurate-profile converter then simplifies shared borders.</p>')
    (output/'index.html').write_text(html)
    save(group/'FILL20_COMPLETE.json',dict(viewer=str(output/'index.html'),vector_report_sha256=sha(vectors/'GEOJSON_COMPLETE.json'),threshold_native_pixels=20))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True)
    with threadpool_limits(limits=2):run(p.parse_args().study)
