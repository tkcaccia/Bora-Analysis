"""Add calibrated fields from unused GCSS and IGNITE source groups."""
import argparse
from pathlib import Path
from experiments.bora_supervised.registry import load,save
from experiments.bora_extension.prepare_tissue_catalog import gcss,ignite


def run(data,registry,output):
    data,registry,output=map(Path,(data,registry,output));old=load(registry/'REGISTRY.json')
    added=[]
    for dataset,function,quota in [('gcss',gcss,40),('ignite',ignite,24)]:
        path=output/dataset/'PLAN.json'
        if path.exists():plan=load(path)
        else:
            skipped={r['sample']['group'] for r in old['cases'] if r['dataset']==dataset}
            options=dict(quota=quota,skip_groups=skipped,case_prefix=dataset+'_e',cache_root=output.parent/'source-cache')
            if dataset=='ignite':options['splits']=('train','test')
            plan=function(data,output,**options)
        added.extend(plan['samples'])
        if not plan['quota_met']:
            # Report the actual accepted scope; never weaken qualification.
            save(output/dataset/'PLAN.json',dict(plan,quota_shortfall_explicit=True))
    save(output/'ACQUISITION_COMPLETE.json',dict(samples=added,accepted=len(added),
        previous_registry=str(registry),new_source_groups_only=True,models_run=False,
        provider_test_images_reserved_from_training=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('data','registry','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.data,a.registry,a.output)


if __name__=='__main__':main()
