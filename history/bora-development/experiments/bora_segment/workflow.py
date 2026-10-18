"""One versioned segment-classification cycle after all geometry is frozen."""
import argparse
from pathlib import Path
from threadpoolctl import threadpool_limits
from experiments.bora_supervised.registry import save,sha
from . import targets,train,final,viewer


def run(study_root,geometry):
    study_root,geometry=Path(study_root),Path(geometry)
    if not (geometry/'ALL_GEOMETRY_FROZEN.json').exists():raise ValueError('Geometry must finish first')
    registry=study_root/'registry-v1';labels=geometry/'targets-v1';validation=geometry/'classifiers-validation-v1'
    tile_bundle=study_root/'models/S06-c24'
    targets.run(registry,geometry,labels)
    train.predict(validation,registry,geometry,labels,tile_bundle,registry)
    train.evaluate(validation,registry,geometry,labels)
    viewer.run(registry,geometry,validation,geometry/'viewer-validation')
    final.run(registry,geometry,validation,geometry/'classifiers-followup-test-v1',tile_bundle,registry,labels)
    viewer.run(registry,geometry,geometry/'classifiers-followup-test-v1',geometry/'viewer-followup-test')
    save(geometry/'CYCLE_COMPLETE.json',dict(unit='Whole connected Bora tissue segment',
        validation_results_sha256=sha(validation/'RESULTS.json'),
        followup_test_results_sha256=sha(geometry/'classifiers-followup-test-v1/RESULTS.json'),
        geometry_unchanged=True,test_reuse_disclosed=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study-root',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=2):run(a.study_root,a.geometry)


if __name__=='__main__':main()
