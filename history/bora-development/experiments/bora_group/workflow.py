"""Fit whole-group classifiers and publish every reserved test prediction."""
import argparse
from pathlib import Path
import cv2
from threadpoolctl import threadpool_limits
from experiments.bora_supervised.registry import load,save,sha
from experiments.bora_segment import train,final
from . import targets,bora_geojson,viewer


def run(study_root,geometry):
    study_root,geometry=Path(study_root),Path(geometry);registry=study_root/'registry-v1'
    frozen=load(geometry/'ALL_GEOMETRY_FROZEN.json')
    if not frozen['all_tile_partitions_identical'] or not frozen['all_rgb_bytes_unchanged'] or not frozen['all_anonymous_bora_geometry_identical']:
        raise ValueError('Previous benchmark preservation audit failed')
    labels=geometry/'targets-v1';validation=geometry/'classifiers-validation-v1';test=geometry/'classifiers-test-v1'
    targets.run(registry,geometry,labels)
    train.predict(validation,registry,geometry,labels,study_root/'models/S06-c24',registry)
    train.evaluate(validation,registry,geometry,labels)
    final.run(registry,geometry,validation,test,study_root/'models/S06-c24',registry,labels)
    vectors=geometry/'geojson-bora-accurate-v2'
    bora_geojson.run(registry,geometry,test,vectors)
    viewer.run(registry,geometry,test,vectors,geometry/'viewer-bora-geojson-v2')
    save(geometry/'CYCLE_COMPLETE.json',dict(prediction_unit=frozen['prediction_unit'],
        validation_results_sha256=sha(validation/'RESULTS.json'),test_results_sha256=sha(test/'RESULTS.json'),
        geojson_checks_sha256=sha(vectors/'GEOJSON_COMPLETE.json'),
        all_previous_tile_assignments_preserved=True,all_rgb_bytes_unchanged=True,
        all_test_predictions_before_test_scoring=True,new_clustering_used=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('study-root','geometry'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();cv2.setNumThreads(2)
    with threadpool_limits(limits=2):run(a.study_root,a.geometry)


if __name__=='__main__':main()
