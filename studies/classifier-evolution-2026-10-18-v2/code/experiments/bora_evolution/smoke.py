"""Executable native-core integration test on held-out synthetic tissues."""
import json,time
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from .catalog import initial
from .models import Model

class Synthetic:
    def __init__(self):
        rng=np.random.default_rng(8);self.classes=np.arange(1,5);self.train=[];self.labels=[];self.source_groups=[];self.views={}
        prototypes=rng.normal(size=(4,32))*3
        for j in range(60):
            y=np.asarray([1+(j%4),1+((j+1)%4)]);x=prototypes[y-1]+rng.normal(size=(2,32))*.3;case=f's{j}'
            self.views[case]={v:(x if v in ('mean','tiles') else np.c_[x,x[:,:8]**2]) for v in ('mean','neighbor','spatial','quantile','covariance','tiles')}
            if j<48:
                for k in range(2):self.train.append((case,k));self.labels.append(y[k]);self.source_groups.append('source'+str(j))
        self.labels=np.asarray(self.labels);self.fractions=np.eye(4)[self.labels-1];self.source_groups=np.asarray(self.source_groups)
    def matrix(self,view='mean',indices=None):
        ids=range(len(self.train)) if indices is None else indices
        return np.stack([self.views[self.train[i][0]][view][self.train[i][1]] for i in ids])
    def case(self,case,view='mean'):return self.views[case][view]
    def case_weights(self,case):return np.asarray([.5,.5])
    def tiles(self,case):return self.case(case),sparse.eye(2,format='csr')
    def tile_training(self,ids=None):
        ids=np.arange(len(self.train)) if ids is None else np.asarray(ids)
        return self.matrix('mean',ids),self.labels[ids]

def run():
    d=Synthetic();results=[]
    for spec in initial():
        start=time.monotonic();m=Model(spec,d);correct=0
        for j in range(48,60):
            s=m.predict_case(f's{j}');assert s.shape==(2,4) and np.isfinite(s).all(),spec['id']
            correct+=np.count_nonzero(d.classes[np.argmax(s,1)]==np.asarray([1+j%4,1+(j+1)%4]))
        row=dict(id=spec['id'],head_count=m.heads,accuracy=correct/24,seconds=time.monotonic()-start)
        assert row['accuracy']>=.5,row
        results.append(row);print(json.dumps(row),flush=True)
    return results

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
