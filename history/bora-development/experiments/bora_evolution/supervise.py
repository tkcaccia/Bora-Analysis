"""Start the authorized ten cycles only after the exact preparer completes."""
import argparse,time,subprocess,os
from pathlib import Path
from experiments.bora_supervised.registry import load,save

def run(study):
    study=Path(study)
    while not (study/'PREPROCESSING_COMPLETE.json').exists():
        receipt=load(study/'pipeline-launch.json')
        p=Path('/proc')/str(receipt['pid'])/'stat'
        if not p.exists() or p.read_text().split()[21]!=receipt['startticks']:
            raise RuntimeError('Exact preprocessor terminated before completion; inspect its saved log')
        time.sleep(10)
    while (study/'VECTOR_PROFILE_PENDING.json').exists():time.sleep(10)
    cpu=study.parent/'tissue-supervised-2026-10-18-v1/.venv/bin/python'
    command=[str(cpu),'-u','-m','experiments.bora_evolution.run','--study',str(study),'--generations','10']
    save(study/'CYCLES_DISPATCH.json',dict(command=command,authorized_ten_generations=True))
    with (study/'cycles.log').open('a') as f:subprocess.run(command,cwd=study/'code',stdout=f,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);run(p.parse_args().study)
