"""Training-only label-free coordinates describe distributions and architecture."""
import argparse,time
from pathlib import Path
import numpy as np
from scipy import sparse,ndimage
import cv2,tifffile
from threadpoolctl import threadpool_limits
from experiments.bora_supervised.registry import load,save,sha
from experiments.bora_segment.context import boundary_context


def weighted_quantiles(x,w,q):
    result=[]
    for j in range(x.shape[1]):
        order=np.argsort(x[:,j],kind='stable');v=x[order,j];mass=w[order];c=np.cumsum(mass)
        result.extend(np.interp(np.asarray(q)*c[-1],c,v).tolist())
    return result


def profile(z,weights,regions,mpp_xy):
    """Exact group/token mass weights; spatial distances use calibrated grid."""
    n,d=weights.shape[0],z.shape[1];q=[];cov=[];spatial=[]
    h,w=regions.shape;yy,xx=np.indices((h,w));xy=np.column_stack([xx.ravel()/max(w-1,1),yy.ravel()/max(h-1,1)])
    for i in range(n):
        row=weights.getrow(i);at=row.indices;mass=row.data.astype(float);mass/=mass.sum()
        x=z[at];mu=mass@x;centred=x-mu;c=(centred*mass[:,None]).T@centred
        c=.9*c+.1*np.eye(d)*max(np.trace(c)/d,1e-6)
        ev,u=np.linalg.eigh(c);log=(u*np.log(np.maximum(ev,1e-6)))@u.T
        cov.append(log[np.triu_indices(d)]);q.append(weighted_quantiles(x,mass,[.1,.25,.5,.75,.9]))
        # Morphology is an input descriptor, never an expert label.
        dist=ndimage.distance_transform_edt(regions==i+1,sampling=mpp_xy[::-1]).ravel()[at]
        inside=mass*(dist>16.);edge=mass*(dist<=16.)
        a=(inside@x/max(inside.sum(),1e-10));b=edge@x/max(edge.sum(),1e-10)
        local=xy[at]-(mass@xy[at]);moments=np.concatenate([(mass*local[:,j])@x for j in range(2)])
        spatial.append(np.r_[a,b,a-b,moments])
    return dict(quantile=np.asarray(q),covariance=np.asarray(cov),spatial=np.asarray(spatial))


def run(study):
    from sklearn.decomposition import PCA
    study=Path(study);registry=load(study/'registry-v2/REGISTRY.json');geo=study/'geometry';out=study/'descriptors'
    xs=[]
    for r in registry['cases']:
        if r['split']!='train':continue
        x=np.load(geo/'cases'/r['case']/'mean.npy')
        xs.append(x)
    x=np.concatenate(xs).astype(float);mu=x.mean(0);sd=np.maximum(x.std(0),1e-6)
    model=PCA(n_components=12,svd_solver='full').fit((x-mu)/sd)
    contract=dict(components=12,training_registry_sha256=sha(study/'registry-v2/REGISTRY.json'),
        training_only=True,descriptor_projection='Label-free PCA of training group features; all classification heads remain native fastPLS',supervised_targets_read=False,code_sha256=sha(Path(__file__)))
    if (out/'PROJECTION.json').exists() and load(out/'PROJECTION.json')!=contract:raise ValueError('Descriptor projection changed')
    save(out/'PROJECTION.json',contract);np.save(out/'projection_train_x.npy',x);np.savez(out/'projection.npz',mean=mu,scale=sd,components=model.components_)
    for r in registry['cases']:
        dst=out/'cases'/r['case'];done=dst/'DESCRIPTORS.json'
        if done.exists():
            record=load(done)
            if record['projection_sha256']!=sha(out/'PROJECTION.json'):raise ValueError('Projection mismatch')
            for n,h in record['files_sha256'].items():
                if sha(dst/n)!=h:raise ValueError('Descriptor changed')
            continue
        start=time.monotonic();folder=geo/'cases'/r['case']
        if sha(r['feature']['path'])!=r['feature']['sha256']:raise ValueError('Frozen full-token feature cache changed')
        tokens=np.load(r['feature']['path'],mmap_mode='r');flat=tokens.reshape(-1,1280)
        z=np.concatenate([model.transform((np.asarray(flat[i:i+2048],dtype=float)-mu)/sd) for i in range(0,len(flat),2048)])
        regions=tifffile.imread(folder/'regions.native.tif');yy=np.minimum(((np.arange(tokens.shape[0])+.5)*14*regions.shape[0]/r['sample']['working_shape_yx'][0]).astype(int),regions.shape[0]-1);xx=np.minimum(((np.arange(tokens.shape[1])+.5)*14*regions.shape[1]/r['sample']['working_shape_yx'][1]).astype(int),regions.shape[1]-1);grid=regions[yy[:,None],xx[None,:]]
        weights=sparse.load_npz(folder/'token_overlap.npz');mpp=14*np.asarray(r['sample']['working_mpp_xy'])
        profiles=profile(z,weights,grid,mpp);distribution_seconds=time.monotonic()-start;mean=np.load(folder/'mean.npy').astype(float)
        neighbor_start=time.monotonic();profiles['neighbor']=boundary_context(regions,mean,r['sample']['native_mpp_xy']);neighbor_seconds=time.monotonic()-neighbor_start
        dst.mkdir(parents=True,exist_ok=True)
        for name,a in profiles.items():
            if not np.isfinite(a).all():raise ValueError('Nonfinite '+name)
            np.save(dst/(name+'.npy'),a)
        save(done,dict(case=r['case'],seconds=time.monotonic()-start,distribution_seconds=distribution_seconds,neighbor_seconds=neighbor_seconds,projection_sha256=sha(out/'PROJECTION.json'),
            geometry_sha256=sha(folder/'GEOMETRY_FROZEN.json'),files_sha256={n+'.npy':sha(dst/(n+'.npy')) for n in profiles},
            fine_reference_read=False,full_1280_input_channels=True))
        print(__import__('json').dumps(dict(described=r['case'],seconds=time.monotonic()-start)),flush=True)
    save(out/'COMPLETE.json',dict(cases=len(registry['cases']),projection=contract,case_manifest_sha256={r['case']:sha(out/'cases'/r['case']/'DESCRIPTORS.json') for r in registry['cases']}))


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);a=p.parse_args();cv2.setNumThreads(2)
    with threadpool_limits(limits=2):run(a.study)

if __name__=='__main__':main()
