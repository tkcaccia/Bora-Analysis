"""Matched controlled refinement comparison using the author's bilateral solver.

Only the eligible validation ROIs are used. Coarse labels and converted RGB
are reference-conditioned legacy inputs; this is a controlled refinement task,
not an independently predicted segmentation result.
"""
import argparse
from pathlib import Path
import json
import time
import subprocess
import cv2
import numpy as np
from PIL import Image
from scipy import sparse
from scipy.sparse.linalg import spsolve,cg
import tifffile
from threadpoolctl import threadpool_limits
from .registry import load,save,sha
from .eligibility import eligible
from .evaluate import pixel_scores


def author_solver(repository):
    notebook=Path(repository)/'notebooks/bilateral_solver.ipynb';nb=load(notebook)
    cells=[]
    for cell in nb['cells']:
        text=''.join(cell.get('source',[]))
        if cell['cell_type']=='code' and any(key in text for key in ('RGB_TO_YUV =','class BilateralGrid','class BilateralSolver')):cells.append(text)
    original='\n\n'.join(cells);scope={'np':np,'xrange':range}
    exec(compile(original,str(notebook),'exec'),scope)
    statuses=[]
    def compatible_cg(a,b,x0=None,M=None,maxiter=None,tol=None):
        x,info=cg(a,b,x0=x0,M=M,maxiter=maxiter,rtol=tol,atol=0)
        statuses.append(dict(info=int(info),relative_residual=float(np.linalg.norm(a@x-b)/max(np.linalg.norm(b),1e-30))))
        return x,info
    scope['cg']=compatible_cg
    return scope,statuses,dict(repository='https://github.com/poolio/bilateral_solver',
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip(),
        notebook_sha256=sha(notebook),
        compatibility='Original definition cells executed unchanged; xrange aliases range; cg(tol) adapts to SciPy cg(rtol), with convergence reporting added.',
        limitation='Author reference notebook implementation is not the optimized implementation used for published runtime claims.')


def diffusion(c,he):
    from experiments.bora_cycles.cycle_six import field_geometry,guide_laplacian
    shape,spacing=field_geometry(c)
    guide=-np.log((cv2.resize(he,shape[::-1],interpolation=cv2.INTER_AREA).astype(float)+1)/256)
    lap=guide_laplacian(guide,spacing)
    mass=cv2.resize(c.weight,shape[::-1],interpolation=cv2.INTER_AREA).ravel()
    rhs=np.stack([cv2.resize(c.weight*(c.labels==g),shape[::-1],interpolation=cv2.INTER_AREA).ravel() for g in range(c.k)],1)
    a=lap+sparse.diags(mass+.015);rhs+=.015/c.k
    q=spsolve(a.tocsc(),rhs).reshape(*shape,c.k)
    if c.k==1:q=q[...,None] if q.ndim==2 else q
    fine=np.stack([cv2.resize(q[...,g],c.shape[::-1],interpolation=cv2.INTER_LINEAR) for g in range(c.k)],-1)
    return fine,dict(name='H&E graph quadratic probability diffusion',
        relation_to_literature='Soft observation graph-Laplacian baseline inspired by random walkers and quadratic image-guided smoothing; not the original hard-seeded random-walker implementation.')


def run(root,output,repository,cg_maxiter=25,methods=None,nonconverged_from=None):
    from bora.evidence import Evidence
    from experiments.bora_cycles.cycle_six import edge_appearance
    from experiments.bora_cycles.cycle_ten import bending_field
    root,output=Path(root),Path(output)
    if output.exists():raise ValueError('Preserve comparator run')
    registry=load(root/'REGISTRY.json');rows=[r for r in registry['cases'] if r['split']=='validation' and eligible(r)]
    if nonconverged_from:
        prior=load(Path(nonconverged_from)/'RESULTS.json')
        remaining={r['case'] for r in prior['cases'] if any(v['info'] for v in r['methods']['FBS-RGB']['details']['conjugate_gradient'])}
        rows=[r for r in rows if r['case'] in remaining]
    author,statuses,provenance=author_solver(repository)
    protocol=dict(task='Controlled reference-quantization refinement; not supervised test performance',
        cases=[r['case'] for r in rows],source_registry_sha256=sha(root/'REGISTRY.json'),
        methods=methods or ['TILES','Bora-AF','Bora-VAF','FBS-HE','FBS-RGB','GRAPH'],
        comparison='Same original H&E, frozen9+19 converted RGB, coarse labels, analysis grid about1um and native tissue support for all methods.',
        fbs=provenance,fbs_parameters=dict(sigma_spatial_um=8,sigma_luma=4,sigma_chroma=4,lam=128,cg_maxiter=cg_maxiter,cg_tol=1e-5,confidence_floor=1e-6),
        repetitions='One warm-up plus three timed refinements. Include image guide/grid construction, solve and native projection; exclude file loading, encoder, conversion and metrics.',
        threads=2,test_annotations_read=False,code_sha256=sha(Path(__file__)))
    if nonconverged_from:protocol.update(convergence_followup_source_sha256=sha(Path(nonconverged_from)/'RESULTS.json'),
        subset_rule='Only cases with a nonzero CG status; no accuracy-based case selection. Converged cases terminate identically at larger iteration limits and can retain their frozen outputs.')
    save(output/'PROTOCOL.json',protocol);records=[]
    for row in rows:
        src=Path(row['original_path']).parent
        image=np.asarray(Image.open(row['original_path']).convert('RGB'))
        rgb=np.asarray(Image.open(src/'ensemble/representation.png').convert('RGB'))
        coarse=tifffile.imread(src/'coarse.tif');support=tifffile.imread(row['tissue_support_path'])>0
        native_support=tifffile.imread(src/'tissue.native.tif')>0
        out=output/'cases'/row['case'];out.mkdir(parents=True)
        entry=dict(case=row['case'],group=row['group'],dataset=row['dataset'],methods={})
        for name in protocol['methods']:
            timings=[];native=None;details={}
            for repeat in range(4):
                began=time.monotonic()
                if name=='TILES':labels=coarse
                else:
                    c=Evidence(rgb,coarse,support,row['sample']['working_mpp_xy'])
                    if c.k==1:q=np.ones((*c.shape,1))
                    elif name=='Bora-AF':q,details=edge_appearance(c,None,image)
                    elif name=='Bora-VAF':q,details=bending_field(c,None,image,'logvariance')
                    elif name=='GRAPH':q,details=diffusion(c,image)
                    else:
                        guide=(cv2.resize(image,c.shape[::-1],interpolation=cv2.INTER_AREA) if name=='FBS-HE'
                               else np.rint(c.rgb*255).astype(np.uint8))
                        grid=author['BilateralGrid'](guide,sigma_spatial=8/np.sqrt(c.mpp.prod()),sigma_luma=4,sigma_chroma=4)
                        target=np.eye(c.k)[c.labels].reshape(-1,c.k);target[~c.known.ravel()]=1/c.k
                        confidence=np.maximum(c.weight.reshape(-1,1),1e-6)
                        before=len(statuses)
                        solver=author['BilateralSolver'](grid,dict(lam=128,A_diag_min=1e-5,cg_tol=1e-5,cg_maxiter=cg_maxiter))
                        q=solver.solve(target,confidence).reshape(*c.shape,c.k)
                        details=dict(bilateral_vertices=grid.nvertices,conjugate_gradient=statuses[before:])
                    if not np.isfinite(q).all():raise ValueError('Nonfinite refinement')
                    labels=c.ids[q.argmax(-1)].astype(np.uint8)
                native=cv2.resize(labels,tuple(row['sample']['native_shape_yx'][::-1]),interpolation=cv2.INTER_NEAREST)
                native[~native_support]=0
                if repeat:timings.append(time.monotonic()-began)
            path=out/(name+'.tif');tifffile.imwrite(path,native,compression='deflate')
            entry['methods'][name]=dict(median_seconds=float(np.median(timings)),repetitions=timings,
                mask_sha256=sha(path),details=details)
        save(out/'PREDICTIONS.json',entry);records.append(entry)
        print(json.dumps(dict(frozen=row['case'])),flush=True)
    save(output/'ALL_PREDICTIONS_FROZEN.json',dict(test_annotations_read=False,
        cases={r['case']:sha(output/'cases'/r['case']/'PREDICTIONS.json') for r in rows}))
    for row,record in zip(rows,records):
        truth=tifffile.imread(row['reference_path'])
        if sha(row['reference_path'])!=row['reference_sha256']:raise ValueError('Reference changed')
        for name,entry in record['methods'].items():
            p=output/'cases'/row['case']/(name+'.tif')
            if sha(p)!=entry['mask_sha256']:raise ValueError('Prediction changed')
            entry['score']=pixel_scores(truth,tifffile.imread(p),len(row['sample']['class_names']))
    ranking=[]
    for name in protocol['methods']:
        entries=[r['methods'][name] for r in records]
        ranking.append(dict(name=name,mean_iou=float(np.mean([r['score']['macro_iou'] for r in entries])),
            mean_seconds=float(np.mean([r['median_seconds'] for r in entries])),
            over_one_second=sum(r['median_seconds']>1 for r in entries),
            fbs_cg_unconverged_class_solves=sum(d['info']!=0 for r in entries for d in r['details'].get('conjugate_gradient',[]))))
    ranking.sort(key=lambda r:-r['mean_iou'])
    save(output/'RESULTS.json',dict(cases=records,ranking=ranking,test_annotations_read=False,
        superiority_claim='Limited to these implementations and controlled development ROIs. No end-to-end foundation-model or optimized-FBS speed superiority is established.'))
    print(json.dumps(ranking),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output','repository'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cg-maxiter',type=int,default=25)
    p.add_argument('--methods',nargs='+',choices=['TILES','Bora-AF','Bora-VAF','FBS-HE','FBS-RGB','GRAPH'])
    p.add_argument('--nonconverged-from',type=Path)
    a=p.parse_args();cv2.setNumThreads(2)
    with threadpool_limits(limits=2):run(a.root,a.output,a.repository,a.cg_maxiter,a.methods,a.nonconverged_from)


if __name__=='__main__':main()
