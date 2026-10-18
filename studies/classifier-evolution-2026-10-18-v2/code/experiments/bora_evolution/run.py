"""Ten generations: twenty native-PLS mechanisms, five parents, fifteen children."""
import argparse,hashlib,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.bora_supervised.registry import load,save,sha
from .catalog import initial
from .data import Data
from .evolve import instantiate,children
from .evaluation import evaluate_model
from .viewer import publish


def predict(study,data,spec,cycle):
    folder=cycle/spec['id'];done=folder/'PREDICTIONS.json'
    if done.exists():
        result=load(done)
        if result['recipe']!=spec:raise ValueError('Completed candidate recipe changed')
        for p in result['predictions']:
            if sha(folder/(p['case']+'.npz'))!=p['sha256']:raise ValueError('Frozen prediction changed')
        return result
    folder.mkdir(parents=True,exist_ok=True);start=time.monotonic();model=instantiate(spec,data);fit_seconds=time.monotonic()-start;records=[]
    for case in data.evaluation:
        model.predict_case(case);times=[]
        for _ in range(3):
            start=time.monotonic();scores=model.predict_case(case);times.append(time.monotonic()-start)
        if scores.shape!=(len(data.case(case)),len(data.classes)) or not np.isfinite(scores).all():raise ValueError('Invalid native tissue evidence')
        order=np.argsort(-scores,axis=1,kind='stable');ranks=data.classes[order[:,:5]]
        path=folder/(case+'.npz');np.savez_compressed(path,top5=ranks,scores=scores)
        records.append(dict(case=case,sha256=sha(path),warm_head_seconds=float(np.median(times)),predictor_input_seconds=0.,
            warm_measurement_includes_case_tile_loading=('tiles' in model.required)))
    manifest=dict(id=spec['id'],recipe=spec,fit_seconds=fit_seconds,classes=data.classes.tolist(),predictions=records,
        native_head_count=model.heads+int(bool(model.required&{'covariance','quantile','spatial'})),
        additional_structural_stages=model.stages,required_descriptors=sorted(model.required),
        training_only=True,evaluation_targets_read=False,threads=2)
    save(done,manifest);print(__import__('json').dumps(dict(candidate=spec['id'],fit_seconds=fit_seconds,predicted_images=len(records))),flush=True)
    return manifest


def run(study,generations=10):
    study=Path(study)
    if not (study/'PREPROCESSING_COMPLETE.json').exists():raise ValueError('Expanded calibrated cohort must freeze before all cycles')
    import bora.mask_cleanup,bora.multicpu_geojson,bora.compact_geojson
    code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')};protocol=dict(policy=load(Path(__file__).with_name('EXECUTION_POLICY.json')),
        descriptors_sha256=sha(study/'descriptors/COMPLETE.json'),registry_sha256=sha(study/'registry-v2/REGISTRY.json'),geometry_sha256=sha(study/'geometry/ALL_GEOMETRY_FROZEN.json'),code_sha256=code,
        boundary_profile='shared-rdp-3um-class97-area1p5',compact_code_sha256=sha(Path(bora.compact_geojson.__file__)),small_region_fill_native_pixels=20,continuous_polygon_area_checked=True,evaluation_policy_sha256=sha(study/'EVALUATION_POLICY.json') if (study/'EVALUATION_POLICY.json').exists() else None,cleanup_code_sha256=sha(Path(bora.mask_cleanup.__file__)),converter_code_sha256=sha(Path(bora.multicpu_geojson.__file__)),completed_generations_requested=generations,started_utc=None)
    path=study/'EVOLUTION_PROTOCOL.json'
    if path.exists():
        if load(path)!=protocol:raise ValueError('Evolution protocol changed: use a new study version')
    else:save(path,protocol)
    data=Data(study);recipes=initial()
    save(study/'EVALUATION_COHORT.json',dict(cases=data.evaluation,training_groups=len(data.train),classes=data.classes.tolist(),
        development_test_reuse=True,all_ten_generations_same_cohort=True,policy=data.evaluation_policy,
        excluded_by_dataset=[c for c,r in data.rows.items() if r['split']!='train' and r['dataset'] in data.evaluation_policy.get('excluded_datasets',[])],
        excluded_single_role=[c for c,r in data.rows.items() if r['split']!='train' and len(set(r['ontology_raw_to_role'].values())-{0})<2]))
    for generation in range(1,generations+1):
        cycle=study/'cycles'/f'cycle-{generation:02d}';complete=cycle/'CYCLE_COMPLETE.json'
        if complete.exists():
            old=load(cycle/'RESULTS.json');survivors=old['selected_recipes'];recipes=survivors+children(survivors,generation+1);continue
        if len(recipes)!=20 or len({r['id'] for r in recipes})!=20:raise ValueError('Exactly twenty distinct recipes required')
        save(cycle/'CANDIDATES.json',dict(generation=generation,candidates=recipes,retained_parents=5 if generation>1 else 0))
        # Every candidate predicts every reserved image before any fine-mask scoring.
        manifests={}
        for recipe in recipes:
            predict(study,data,recipe,cycle);manifests[recipe['id']]=sha(cycle/recipe['id']/'PREDICTIONS.json')
        save(cycle/'ALL_PREDICTIONS_FROZEN.json',dict(models=manifests,cases=data.evaluation,targets_read=False))
        ranking=[]
        for recipe in recipes:
            folder=cycle/recipe['id'];result_file=folder/'RESULTS.json'
            result=load(result_file) if result_file.exists() else evaluate_model(study,data,folder)
            ranking.append(result);print(__import__('json').dumps(dict(scored=recipe['id'],selection_score=result['selection_score'],iou=result['mean_final_geojson_iou'])),flush=True)
        ranking.sort(key=lambda r:(-r['selection_score'],r['id']));names=[r['id'] for r in ranking[:5]];survivors=[next(s for s in recipes if s['id']==name) for name in names]
        save(cycle/'RESULTS.json',dict(generation=generation,ranking=ranking,selected=names,selected_recipes=survivors,development_test_reuse=True))
        publish(study,data,cycle,ranking)
        save(complete,dict(generation=generation,utc=datetime.now(timezone.utc).isoformat(),selected=names,
            all_twenty_evaluated=True,viewer=str(cycle/'viewer/index.html'),all_predictions_sha256=sha(cycle/'ALL_PREDICTIONS_FROZEN.json'),results_sha256=sha(cycle/'RESULTS.json')))
        save(study/'LATEST_COMPLETE.json',load(complete));print('CYCLE_COMPLETE '+str(generation)+' '+','.join(names),flush=True)
        recipes=survivors+children(survivors,generation+1)
    save(study/'TEN_CYCLES_COMPLETE.json',dict(completed_generations=generations,all_twenty_evaluated_each_generation=True,adaptive_development_study=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--generations',type=int,default=10);a=p.parse_args()
    with threadpool_limits(limits=2):run(a.study,a.generations)
