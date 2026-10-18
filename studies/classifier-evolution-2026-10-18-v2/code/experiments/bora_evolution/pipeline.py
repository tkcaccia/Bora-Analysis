"""Resume-safe preparation chain bound to the exact acquisition process."""
import argparse,os,subprocess,time
from pathlib import Path
from experiments.bora_supervised.registry import load,save


def run(study):
    study=Path(study);base=study.parent;data=Path('/mnt/sata_ssd/Bora-benchmark');cpu=base/'tissue-supervised-2026-10-18-v1/.venv/bin/python'
    gpu=data/'experiments/virchow2-inverse-2026-10-16/.venv/bin/python';receipt=load(study/'acquisition-launch-v2.json')
    while not (study/'acquisition-v2/ACQUISITION_COMPLETE.json').exists():
        stat=Path('/proc')/str(receipt['pid'])/'stat'
        if not stat.exists() or stat.read_text().split()[21]!=receipt['startticks']:
            raise RuntimeError('Acquisition stopped before completion; inspect its existing log, do not relaunch blindly')
        time.sleep(10)
    def execute(label,cmd):
        log=study/(label+'.log');start=time.time();print('STAGE '+label,flush=True)
        with log.open('a') as stream:subprocess.run([str(v) for v in cmd],cwd=study/'code',stdout=stream,stderr=subprocess.STDOUT,check=True)
        save(study/(label+'-complete.json'),dict(command=[str(v) for v in cmd],wall_seconds=time.time()-start))
    if not (study/'features-new/ALL_FEATURES_FROZEN.json').exists():
        execute('features',[gpu,'-u','-m','experiments.bora_extension.features','--sources',study/'acquisition-v2/gcss',study/'acquisition-v2/ignite','--root',study/'features-new','--model-dir',data/'models/Virchow2/3158645804b69e3f3bc4439d4116edddf0840a72'])
    if not (study/'PREPARATION_COMPLETE.json').exists():
        execute('prepare',[cpu,'-u','-m','experiments.bora_evolution.prepare','--study',study,'--old-registry',base/'tissue-supervised-2026-10-18-v1/registry-v1','--old-geometry',base/'bora-anonymous-groups-2026-10-18-v1','--data',data])
    if not (study/'descriptors/COMPLETE.json').exists():execute('descriptors',[cpu,'-u','-m','experiments.bora_evolution.descriptors','--study',study])
    save(study/'PREPROCESSING_COMPLETE.json',dict(ready=True,cycles_not_started_by_preprocessor=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);run(p.parse_args().study)
