"""Predictor store; evaluation labels are deliberately absent from this class."""
from pathlib import Path
import numpy as np
from scipy import sparse
from experiments.bora_supervised.registry import load,sha
from experiments.bora_supervised.eligibility import eligible


def evaluation_cases(rows,policy=None):
    """Explicit cohort exclusions affect evaluation only, never training folds."""
    policy=policy or {};excluded=set(policy.get('excluded_datasets',[]))
    return [c for c,r in rows.items() if r['split']!='train' and eligible(r) and r['dataset'] not in excluded]


class Data:
    def __init__(self,study):
        self.root=Path(study);self.registry=load(self.root/'registry-v2/REGISTRY.json')
        self.rows={r['case']:r for r in self.registry['cases']};self.geo=self.root/'geometry';self.views={};self.tile_cache={};self.target={}
        self.train=[];self.labels=[];self.fractions=[];self.source_groups=[]
        frozen_descriptors=load(self.root/'descriptors/COMPLETE.json');frozen_geometry=load(self.geo/'ALL_GEOMETRY_FROZEN.json');frozen_targets=load(self.root/'targets/TARGETS_FROZEN.json')
        for case,r in self.rows.items():
            geo=self.geo/'cases'/case
            if sha(geo/'GEOMETRY_FROZEN.json')!=frozen_geometry['cases'][case]:raise ValueError('Frozen geometry record changed')
            gf=load(geo/'GEOMETRY_FROZEN.json')
            for name in ('mean.npy','tile_overlap.npz','regions.native.tif'):
                if sha(geo/name)!=gf['files_sha256'][name]:raise ValueError('Frozen group predictor changed')
            descriptor=self.root/'descriptors/cases'/case
            if sha(descriptor/'DESCRIPTORS.json')!=frozen_descriptors['case_manifest_sha256'][case]:raise ValueError('Frozen descriptor record changed')
            for name,h in load(descriptor/'DESCRIPTORS.json')['files_sha256'].items():
                if sha(descriptor/name)!=h:raise ValueError('Frozen descriptor changed')
            mean=np.asarray(np.load(self.geo/'cases'/case/'mean.npy'),dtype=float)
            folder=self.root/'descriptors/cases'/case
            self.views[case]={'mean':mean}
            for view in ('covariance','quantile','spatial','neighbor'):
                a=np.load(folder/(view+'.npy'))
                self.views[case][view]=a if view=='neighbor' else np.concatenate([mean,a],axis=1)
            if r['split']=='train':
                if sha(self.root/'targets/cases'/case/'targets.npz')!=frozen_targets['cases'][case]['sha256']:raise ValueError('Training target changed')
                with np.load(self.root/'targets/cases'/case/'targets.npz') as t:
                    identity=t['identity'];fraction=t['fractions'];keep=(identity>0)&(fraction.sum(1)>0)
                    self.target[case]=dict(identity=identity.copy(),fractions=fraction.copy())
                    for i in np.flatnonzero(keep):
                        self.train.append((case,int(i)));self.labels.append(identity[i]);self.fractions.append(fraction[i]);self.source_groups.append(r['group'])
        self.labels=np.asarray(self.labels,int);self.fractions=np.asarray(self.fractions,float)
        self.source_groups=np.asarray(self.source_groups);self.classes=np.unique(self.labels)
        policy_path=self.root/'EVALUATION_POLICY.json'
        self.evaluation_policy=load(policy_path) if policy_path.exists() else {}
        self.evaluation=evaluation_cases(self.rows,self.evaluation_policy)
        self._tiles_train=None

    def matrix(self,view='mean',indices=None):
        if indices is None:indices=np.arange(len(self.train))
        return np.asfortranarray(np.stack([self.views[self.train[i][0]][view][self.train[i][1]] for i in indices]))

    def case(self,case,view='mean'):return np.asfortranarray(self.views[case][view])

    def case_weights(self,case):
        r=load(self.geo/'cases'/case/'REGIONS.json');a=np.asarray([v['area_pixels'] for v in r['regions']],float)
        return a/max(a.sum(),1)

    def tiles(self,case):
        # Loading one case bounds resident memory on large calibrated BCSS ROIs.
        return (np.asarray(np.load(self.root/'registry-v2/cases'/case/'mean.npy'),dtype=float),
                sparse.load_npz(self.geo/'cases'/case/'tile_overlap.npz'))

    def tile_training(self,indices=None,cap=320):
        import hashlib
        cases={self.train[i][0] for i in (range(len(self.train)) if indices is None else indices)}
        reservoir={}
        for case in sorted(cases):
            with np.load(self.root/'registry-v2/cases'/case/'targets.npz') as t,np.load(self.root/'registry-v2/cases'/case/'units.npz') as u:
                f=t['fractions'];y=f.argmax(1)+1;keep=(t['coverage']>=.5)&(u['tissue_fraction']>=.5)&(f.sum(1)>0)
                for i in np.flatnonzero(keep):
                    key=hashlib.sha256(f'evolution-tile:{case}:{i}'.encode()).hexdigest()
                    reservoir.setdefault(int(y[i]),[]).append((key,case,int(i)))
        selected=[(case,i,y) for y,rows in sorted(reservoir.items()) for _,case,i in sorted(rows)[:cap]]
        xs=[];ys=[];by_case={}
        for case,i,y in selected:by_case.setdefault(case,[]).append((i,y))
        for case,items in sorted(by_case.items()):
            x=np.load(self.root/'registry-v2/cases'/case/'mean.npy',mmap_mode='r');ids=[i for i,y in items]
            xs.append(np.asarray(x[ids],float));ys.extend(y for i,y in items)
        return np.asfortranarray(np.concatenate(xs)),np.asarray(ys,int)
