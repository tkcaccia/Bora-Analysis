"""CPU regression of the packaged classifier against frozen research outputs.

Read-only study access. No feature extraction, GPU work or experiment controller.
Only the audit report and an optional safe model artifact are written.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import tifffile
from threadpoolctl import threadpool_limits
from bora.classifier import NeighborContextClassifier
from bora.groups import boundary_context
from bora._vendor import fastpls


def run(study, output):
    registry=json.loads((study/'registry-v2/REGISTRY.json').read_text())
    xs=[];ys=[]
    for row in registry['cases']:
        if row['split']!='train':continue
        case=row['case']
        x=np.load(study/'descriptors/cases'/case/'neighbor.npy')
        with np.load(study/'targets/cases'/case/'targets.npz',allow_pickle=False) as target:
            keep=(target['identity']>0)&(target['fractions'].sum(1)>0)
            xs.extend(x[keep]);ys.extend(target['identity'][keep])
    x=np.asfortranarray(np.stack(xs),dtype=np.float64);y=np.asarray(ys,np.int64)
    names={int(i):registry['ontology']['roles'][int(i)-1] for i in np.unique(y)}
    start=time.perf_counter();model=NeighborContextClassifier().fit(x,y,names)
    fit=time.perf_counter()-start
    # This identifier is used only to locate immutable historical predictions.
    result=json.loads((study/'cycles/cycle-03/RESULTS.json').read_text())
    reference=next(m for m in result['ranking'] if m['id']=='P04')
    cases=[];context=[];sampled=set()
    rows={r['case']:r for r in registry['cases']}
    for saved in reference['cases']:
        case=saved['case'];old=np.load(study/'descriptors/cases'/case/'neighbor.npy')
        prediction=model.predict(old)
        np.testing.assert_array_equal(prediction,np.asarray(saved['top1']))
        cases.append(case)
        row=rows[case]
        if row['dataset'] not in sampled:
            geo=study/'geometry/cases'/case
            means=np.load(geo/'mean.npy')
            regions=tifffile.imread(geo/'regions.native.tif')
            new=boundary_context(regions,means,row['sample']['native_mpp_xy'])
            np.testing.assert_array_equal(new,old)
            np.testing.assert_array_equal(model.predict(new),prediction)
            context.append(dict(case=case,dataset=row['dataset'],descriptor_bitwise_identical=True))
            sampled.add(row['dataset'])
    output.parent.mkdir(parents=True,exist_ok=True)
    model_path=output.parent/'neighbor-context-training-artifact.npz'
    model.save(model_path)
    loaded=NeighborContextClassifier.load(model_path)
    for saved in reference['cases']:
        a=np.load(study/'descriptors/cases'/saved['case']/'neighbor.npy')
        np.testing.assert_array_equal(loaded.predict(a),np.asarray(saved['top1']))
    report=dict(ok=True,training_shape=list(x.shape),dtype=str(x.dtype),training_classes=len(names),
        components=model.actual_components_,fit_seconds=fit,fastpls_version=fastpls.__version__,
        vendored_module=fastpls.__name__,evaluation_images=len(cases),
        predictions_identical_for_all_images=True,roundtrip_predictions_identical=True,
        physical_context_checks=context,encoder_extraction_performed=False,gpu_work_performed=False,
        source_registry_sha256=hashlib.sha256((study/'registry-v2/REGISTRY.json').read_bytes()).hexdigest())
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    with threadpool_limits(limits=2):run(args.study,args.output)
