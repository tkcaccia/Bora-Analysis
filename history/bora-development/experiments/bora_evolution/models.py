"""Twenty distinct mechanisms, each fitted through the native fastPLS core.

Postprocessing ranks are evidence scores, not calibrated probabilities.
Evaluation labels never enter this module.
"""
import numpy as np
from scipy.special import softmax,expit
from sklearn.cluster import KMeans


def onehot(y,classes):return (np.asarray(y)[:,None]==np.asarray(classes)[None,:]).astype(float)

def standard(x):
    mu=x.mean(0);sd=np.maximum(x.std(0),1e-6);return (x-mu)/sd,mu,sd

def normalized(s):
    s=np.asarray(s,float);return (s-s.mean(1,keepdims=True))/np.maximum(s.std(1,keepdims=True),1e-6)


class Native:
    def __init__(self,x,y,classes=None,kind='reg',components=24,method='simpls'):
        from fastpls import PLS
        self.classes=np.unique(y) if kind=='lda' else classes;self.kind=kind
        self.constant=None;self.response_constant=None;self.heads=1
        if kind!='lda' and np.max(np.std(np.asarray(y).reshape(len(y),-1),axis=0))<1e-12:
            self.response_constant=np.asarray(y).reshape(len(y),-1).mean(0);self.heads=0;return
        if kind=='lda' and len(self.classes)==1:
            self.constant=int(self.classes[0]);self.heads=0;return
        n=max(1,min(components,x.shape[1],len(x)-1))
        options=dict(n_components=n,scaling='autoscaling',backend='cpu',seed=314159,
                     method=method,classifier='lda' if kind=='lda' else None)
        if method=='kernelpls':options.update(kernel='rbf',gamma=1./max(x.shape[1],1))
        self.model=PLS(**options).fit(np.asfortranarray(x,dtype=float),np.asfortranarray(y,dtype=float) if np.asarray(y).ndim==2 else y)
    def scores(self,x,classes):
        if self.response_constant is not None:return np.tile(self.response_constant,(len(x),1))
        if self.constant is not None:return onehot(np.full(len(x),self.constant),classes)
        if self.kind=='lda':
            rank=np.asarray(self.model.predict(np.asfortranarray(x,dtype=float),top=len(self.classes))).reshape(len(x),-1)
            return np.column_stack([(len(self.classes)-np.argmax(rank==c,1))/len(self.classes) if c in self.classes else np.zeros(len(x)) for c in classes])
        return np.asarray(self.model.predict_responses(np.asfortranarray(x,dtype=float))).reshape(len(x),-1)
    def latent(self,x):return np.asarray(self.model.predict_scores(np.asfortranarray(x,dtype=float)))


class Model:
    def __init__(self,spec,data,indices=None):
        self.spec=spec;self.data=data;self.classes=data.classes;self.ids=np.arange(len(data.train)) if indices is None else np.asarray(indices)
        self.engine=spec['engine'];self.view=spec.get('view','mean');self.heads=0;self.stages=0;self.required=set()
        if spec.get('operator'):raise ValueError('Evolution models are fitted by EvolutionModel')
        self.fit()
    def native(self,x,y,**kwargs):
        m=Native(x,y,**kwargs);self.heads+=m.heads;return m
    def fit(self):
        d=self.data;ids=self.ids;x=d.matrix(self.view if self.view!='tiles' else 'mean',ids);y=d.labels[ids];oh=onehot(y,self.classes);e=self.engine
        if self.view in ('covariance','quantile','spatial','neighbor'):self.required.add(self.view)
        if e in ('lda','kernel','opls'):
            self.head=self.native(x,y,kind='lda',method={'kernel':'kernelpls','opls':'opls'}.get(e,'simpls'))
        elif e=='composition':self.head=self.native(x,d.fractions[ids][:,self.classes-1],classes=self.classes)
        elif e in ('tile_vote','attention'):
            tx,ty=d.tile_training(ids);self.head=self.native(tx,ty,kind='lda');self.required.add('tiles');self.stages=1
        elif e=='ecoc':
            rng=np.random.default_rng(314159);self.code=rng.choice([-1.,1.],size=(len(self.classes),max(32,2*len(self.classes))))
            self.code=np.concatenate([self.code,np.eye(len(self.classes))*2-1],1)
            self.head=self.native(x,oh@self.code,classes=self.classes);self.stages=1
        elif e=='embedding':
            self.proto=np.stack([x[y==c].mean(0) if np.any(y==c) else x.mean(0) for c in self.classes]);z,mu,sd=standard(self.proto)
            u,s,v=np.linalg.svd(z,full_matrices=False);self.code=u[:,:min(8,len(u)-1)]*s[:min(8,len(u)-1)]
            self.head=self.native(x,oh@self.code,classes=self.classes);self.stages=1
        elif e=='boost':
            self.chain=[];residual=oh.copy()
            for i in range(3):
                m=self.native(x,residual,classes=self.classes,components=8);self.chain.append(m);residual-=.7*m.scores(x,self.classes)
            self.stages=2
        elif e=='bag':
            rng=np.random.default_rng(314159);self.chain=[];groups=d.source_groups[ids];unique=np.unique(groups)
            for j in range(4):
                drawn=rng.choice(unique,len(unique),replace=True);sample=np.concatenate([np.flatnonzero(groups==g) for g in drawn]);cols=np.sort(rng.choice(x.shape[1],min(384,x.shape[1]),replace=False))
                m=self.native(x[sample][:,cols],y[sample],kind='lda');self.chain.append((m,cols))
            self.stages=1
        elif e=='experts':
            self.global_head=self.native(x,y,kind='lda');z=self.global_head.latent(x);self.gate=KMeans(n_clusters=4,n_init=5,random_state=314159).fit(z)
            self.chain=[]
            for cluster in range(4):
                use=self.gate.labels_==cluster
                self.chain.append(self.native(x[use],y[use],kind='lda') if use.sum()>=12 and len(np.unique(y[use]))>=2 else None)
            self.stages=2
        elif e=='density':
            self.head=self.native(x,y,kind='lda');z=self.head.latent(x);self.densities=[]
            for c in self.classes:
                a=z[y==c];mu=a.mean(0) if len(a) else z.mean(0);var=np.maximum(.8*(a.var(0) if len(a) else z.var(0))+.2*z.var(0),1e-4);self.densities.append((mu,var,len(a)>0))
            self.stages=1
        elif e=='verification':
            rng=np.random.default_rng(314159);self.proto=np.stack([x[y==c].mean(0) if np.any(y==c) else x.mean(0) for c in self.classes]);self.present=np.isin(self.classes,y);a=rng.integers(len(x),size=6*len(x));b=rng.integers(len(x),size=len(a))
            # Include positives even for rare classes; identity pairs are limited.
            for j in range(0,len(a),2):b[j]=rng.choice(np.flatnonzero(y==y[a[j]]))
            self.head=self.native(np.abs(x[a]-x[b]),(y[a]==y[b]).astype(float)[:,None],classes=self.classes);self.stages=1
        elif e=='tree':
            self.tree=[];present=np.unique(y);prototypes={c:x[y==c].mean(0) for c in present}
            def build(classes):
                if len(classes)==1:return int(classes[0])
                p=np.stack([prototypes[c] for c in classes]);u,s,v=np.linalg.svd(p-p.mean(0),full_matrices=False)
                order=np.argsort(u[:,0],kind='stable');left=classes[order[:len(classes)//2]];right=classes[order[len(classes)//2:]];keep=np.isin(y,classes)
                m=self.native(x[keep],np.isin(y[keep],right).astype(float)[:,None],classes=self.classes,components=12)
                return (m,build(left),build(right))
            self.tree=build(present);self.stages=1
        elif e=='pairwise':
            self.chain=[]
            for i,c in enumerate(self.classes):
                for j in range(i):
                    b=self.classes[j]
                    if not np.any(y==c) or not np.any(y==b):continue
                    keep=(y==c)|(y==b);m=self.native(x[keep],(y[keep]==c).astype(float)[:,None],classes=self.classes,components=12)
                    self.chain.append((i,j,m))
            self.stages=1
        elif e=='transport':
            self.head=self.native(x,y,kind='lda');ix=[];iy=[]
            for case in sorted({d.train[i][0] for i in ids}):
                select=np.asarray([j for j,i in enumerate(ids) if d.train[i][0]==case]);a=x[select];w=d.case_weights(case)[[d.train[ids[j]][1] for j in select]];w/=w.sum()
                ix.append(w@a);iy.append(w@oh[select])
            self.composition=self.native(np.asarray(ix),np.asarray(iy),classes=self.classes,components=12);self.stages=1
        else:raise ValueError('Unknown mechanism '+e)
    def scores_x(self,x):
        e=self.engine
        if e in ('lda','kernel','opls','composition','transport'):return self.head.scores(x,self.classes)
        if e in ('ecoc','embedding'):
            p=self.head.scores(x,self.classes);return -np.mean((p[:,None,:]-self.code[None,:,:])**2,2)
        if e=='boost':return sum(.7*m.scores(x,self.classes) for m in self.chain)
        if e=='bag':return np.mean([m.scores(x[:,cols],self.classes) for m,cols in self.chain],0)
        if e=='density':
            z=self.head.latent(x);return np.column_stack([-.5*np.mean(np.log(v)+(z-mu)**2/v,1) if present else np.full(len(x),-1e6) for mu,v,present in self.densities])
        if e=='verification':return np.column_stack([self.head.scores(np.abs(x-p),self.classes).ravel() if self.present[i] else np.full(len(x),-1e6) for i,p in enumerate(self.proto)])
        if e=='experts':
            s=self.global_head.scores(x,self.classes);gate=self.gate.predict(self.global_head.latent(x))
            for c,m in enumerate(self.chain):
                use=gate==c
                if m is not None and use.any():s[use]=.5*s[use]+.5*m.scores(x[use],self.classes)
            return s
        if e=='pairwise':
            s=np.zeros((len(x),len(self.classes)))
            for i,j,m in self.chain:
                p=expit(4*(m.scores(x,self.classes).ravel()-.5));s[:,i]+=p;s[:,j]+=1-p
            return s/max(len(self.classes)-1,1)
        if e=='tree':
            s=np.zeros((len(x),len(self.classes)))
            def visit(node,weight):
                if isinstance(node,int):s[:,np.flatnonzero(self.classes==node)[0]]=weight;return
                m,left,right=node;p=np.clip(m.scores(x,self.classes).ravel(),.01,.99);visit(left,weight*(1-p));visit(right,weight*p)
            visit(self.tree,np.ones(len(x)));return np.log(np.maximum(s,1e-12))
        raise ValueError('Case context required '+e)
    def predict_case(self,case):
        if self.engine in ('tile_vote','attention'):
            tx,w=self.data.tiles(case);s=self.head.scores(tx,self.classes)
            if self.engine=='attention':
                sorted_s=np.sort(s,axis=1);confidence=np.maximum(sorted_s[:,-1]-sorted_s[:,-2],.02);w=w.multiply(confidence[None,:])
            return (w@s)/np.maximum(np.asarray(w.sum(1)),1e-12)
        x=self.data.case(case,self.view);s=self.scores_x(x)
        if self.engine=='transport':
            w=self.data.case_weights(case);target=softmax(self.composition.scores((w@x)[None,:],self.classes).ravel()*3)
            p=softmax(normalized(s),axis=1)
            for _ in range(8):
                current=w@p;p*=np.sqrt(target/np.maximum(current,1e-8))[None,:];p/=p.sum(1,keepdims=True)
            s=np.log(np.maximum(p,1e-12))
        return s
    def training_scores(self):
        cache={};out=[]
        for i in self.ids:
            case,j=self.data.train[i]
            if case not in cache:cache[case]=self.predict_case(case)
            out.append(cache[case][j])
        return np.asarray(out)
