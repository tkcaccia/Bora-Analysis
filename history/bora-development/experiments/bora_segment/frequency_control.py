"""Post hoc class-frequency control; no feature fitting or model reselection."""
import argparse
from pathlib import Path
import time
import numpy as np
from experiments.bora_supervised.registry import load,save,sha
from .train import evaluate


def run(registry_root,geometry,output):
    registry_root,geometry,output=map(Path,(registry_root,geometry,output))
    if output.exists():raise ValueError('Preserve the diagnostic control')
    registry=load(registry_root/'REGISTRY.json');training=geometry/'classifiers-validation-v1/R02_TRAINING.json'
    counts=load(training)['eligible_by_dominant_role'];order=np.array(sorted(map(int,counts),key=lambda g:(-counts[str(g)],g)))
    save(output/'PROTOCOL.json',dict(control='Training-frequency ranking, identical for every Bora region',
        chosen_after_main_cycle=True,purpose='Assess whether top-five containment is explained by common tissue roles',
        training_counts_sha256=sha(training),no_test_statistics_used_to_set_ranks=True,primary_model_selection_unchanged=True))
    for split in ('validation','test'):
        root=output/split;folder=root/'TRAINING-FREQUENCY';folder.mkdir(parents=True);records=[]
        save(root/'PROTOCOL.json',dict(selected_on_validation=['TRAINING-FREQUENCY'],post_hoc_diagnostic=True))
        for row in registry['cases']:
            if row['split']!=split:continue
            n=load(geometry/'cases'/row['case']/'GEOMETRY_FROZEN.json')['regions'];began=time.monotonic()
            ranks=np.repeat(order[None,:5],n,axis=0);seconds=time.monotonic()-began
            path=folder/(row['case']+'.npz');np.savez(path,top5=ranks)
            records.append(dict(case=row['case'],regions=n,head_seconds=seconds,sha256=sha(path)))
        save(folder/'PREDICTIONS.json',dict(name='TRAINING-FREQUENCY',family='PRIOR',classes=sorted(order.tolist()),predictions=records))
        save(root/'ALL_PREDICTIONS_FROZEN.json',dict(split=split,models={'TRAINING-FREQUENCY':sha(folder/'PREDICTIONS.json')}))
    # Both reserved partitions have fixed predictions before any diagnostic score.
    for split in ('validation','test'):evaluate(output/split,registry_root,geometry,geometry/'targets-v1')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('registry-root','geometry','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.registry_root,a.geometry,a.output)


if __name__=='__main__':main()
