"""Region context from shared physical boundaries, with no tissue labels."""
import numpy as np
from scipy import sparse


def boundary_context(regions,means,mpp_xy):
    rows=[];cols=[];weights=[];n=len(means)
    for a,b,length in [(regions[:,:-1],regions[:,1:],mpp_xy[1]),
                       (regions[:-1],regions[1:],mpp_xy[0])]:
        use=(a>0)&(b>0)&(a!=b);i=a[use]-1;j=b[use]-1
        rows.extend([i,j]);cols.extend([j,i]);weights.extend([np.full(len(i),length),np.full(len(i),length)])
    graph=sparse.coo_matrix((np.concatenate(weights),(np.concatenate(rows),np.concatenate(cols))),shape=(n,n)).tocsr()
    total=np.asarray(graph.sum(1)).ravel();neighbours=means.copy();has=total>0
    neighbours[has]=(graph@means)[has]/total[has,None]
    return np.concatenate([means,means-neighbours],axis=1)
