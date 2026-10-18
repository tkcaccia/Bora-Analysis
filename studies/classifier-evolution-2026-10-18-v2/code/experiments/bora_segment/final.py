"""Freeze the validation-selected region families on reserved source groups."""
import argparse
from pathlib import Path
import time
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from experiments.bora_supervised.registry import load,save,sha
from experiments.bora_supervised.predict import load_model
from .train import region_features,tile_voter,evaluate


def run(registry_root,geometry,validation,output,tile_bundle,tile_registry,targets):
    registry_root,geometry,validation,output,tile_registry=map(Path,(registry_root,geometry,validation,output,tile_registry))
    if output.exists():raise ValueError('Preserve follow-up test cycle')
    registry=load(registry_root/'REGISTRY.json');results=load(validation/'RESULTS.json')
    if results['split']!='validation':raise ValueError('Select on validation only')
    seen=set();selected=[]
    for candidate in results['ranking']:
        if candidate['family'] not in seen:seen.add(candidate['family']);selected.append(candidate)
    rows=[r for r in registry['cases'] if r['split']=='test']
    save(output/'PROTOCOL.json',dict(unit=load(geometry/'ALL_GEOMETRY_FROZEN.json').get('prediction_unit','Whole connected Bora region'),selected_on_validation=[r['name'] for r in selected],
        validation_results_sha256=sha(validation/'RESULTS.json'),geometry_sha256=sha(geometry/'ALL_GEOMETRY_FROZEN.json'),
        prior_test_reuse='This is exploratory follow-up on a cohort already scored for tile/superpixel controls.',
        new_test_targets_used_to_select=False))
    models={}
    for candidate in selected:
        name=candidate['name'];folder=output/name;folder.mkdir()
        if candidate['family']=='R01':infer,classes=tile_voter(tile_bundle);feature='Tile vote';startup=None
        else:infer,metadata,startup=load_model(validation/name);feature=metadata['feature'];classes=metadata['classes']
        records=[]
        for row in rows:
            began=time.monotonic()
            if candidate['family']=='R01':
                x=np.load(tile_registry/'cases'/row['case']/'mean.npy');w=sparse.load_npz(geometry/'cases'/row['case']/'tile_overlap.npz')
                operation=lambda:infer(x,w)
            else:
                x=region_features(geometry,row['case'],feature,row['sample']['native_mpp_xy']);operation=lambda:infer(x)
            features_seconds=time.monotonic()-began;operation();times=[]
            for _ in range(5):
                began=time.monotonic();ranks=operation();times.append(time.monotonic()-began)
            path=folder/(row['case']+'.npz');np.savez(path,top5=ranks)
            records.append(dict(case=row['case'],regions=len(ranks),head_seconds=float(np.median(times)),
                feature_loading_and_context_seconds=features_seconds,sha256=sha(path)))
        record=dict(name=name,family=candidate['family'],feature=feature,classes=list(map(int,classes)),
            startup_refit_seconds=startup,predictions=records,selected_on_validation=True)
        save(folder/'PREDICTIONS.json',record);models[name]=sha(folder/'PREDICTIONS.json')
    save(output/'ALL_PREDICTIONS_FROZEN.json',dict(models=models,split='test',geometry_unchanged=True,
        test_annotations_decoded_by_predictor=False,selected_on_validation=True))
    evaluate(output,registry_root,geometry,targets)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('registry-root','geometry','validation','output','tile-bundle','tile-registry','targets'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=2):run(a.registry_root,a.geometry,a.validation,a.output,a.tile_bundle,a.tile_registry,a.targets)


if __name__=='__main__':main()
