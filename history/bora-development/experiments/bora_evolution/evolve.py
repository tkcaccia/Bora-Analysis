"""Structural descendants; source-held-out teacher evidence drives new stages."""
import hashlib
import numpy as np
from scipy.special import softmax
from sklearn.cluster import KMeans
from .models import Model,Native,onehot,normalized
from .catalog import EVOLUTION_DIRECTIONS


def instantiate(spec,data,indices=None,crossfit=True):
    return EvolutionModel(spec,data,indices,crossfit) if spec.get('operator') else Model(spec,data,indices)


def folds(groups):return np.asarray([int(hashlib.sha256(('evolution-oof:'+g).encode()).hexdigest()[:8],16)%3 for g in groups])


def cross_fitted(spec,data,indices):
    """Every held-out source is absent from native teacher-head fits.

    Inner graph construction may use in-sample training responses; the outer
    holdout remains absent from every native fit. Shared descriptor PCA
    is label-free and fixed on training features; it never reads class targets.
    """
    indices=np.asarray(indices);partition=folds(data.source_groups[indices]);scores=np.zeros((len(indices),len(data.classes)))
    for f in range(3):
        train=indices[partition!=f];hold=np.flatnonzero(partition==f)
        if not len(hold):continue
        if len(train)<5:raise ValueError('Insufficient source-held-out teacher training')
        m=instantiate(spec,data,train,crossfit=False);cache={}
        for j in hold:
            case,k=data.train[indices[j]]
            if case not in cache:cache[case]=m.predict_case(case)
            scores[j]=cache[case][k]
    return normalized(scores)


def children(survivors,generation):
    result=[]
    for rank,parent in enumerate(survivors):
        for child in range(3):
            op,description=EVOLUTION_DIRECTIONS[((generation-2)*5+rank*3+child)%len(EVOLUTION_DIRECTIONS)]
            # Graph depth is bounded by compression. A child at depth4 uses a
            # parent response student, not an exponentially growing ensemble.
            result.append(dict(id=f'C{generation:02d}-{rank+1}{child+1}',name=f'{parent["id"]}: {op.replace("_"," ")}',
                description=description,operator=op,parent_recipe=parent,parents=[parent['id']],generation=generation,
                engine='evolved_native_pls',view='mean',native_backend='fastPLS C++ CPU',
                depth=parent.get('depth',0)+1))
    return result


class EvolutionModel:
    def __init__(self,spec,data,indices=None,crossfit=True):
        self.spec=spec;self.data=data;self.ids=np.arange(len(data.train)) if indices is None else np.asarray(indices)
        self.classes=data.classes;self.op=spec['operator'];self.required=set();self.heads=0;self.stages=1
        parent=spec['parent_recipe'];self.parent=instantiate(parent,data,self.ids,crossfit=False)
        teacher=cross_fitted(parent,data,self.ids) if crossfit else normalized(self.parent.training_scores())
        x=data.matrix('mean',self.ids);y=data.labels[self.ids];target=onehot(y,self.classes);teacher_p=softmax(teacher,axis=1)
        self.required.update(self.parent.required);op=self.op;self.compress=op in ('distillation','source_bagged_student','orthogonal_student')
        if op=='source_bagged_student' and crossfit:
            teacher_sum=np.zeros_like(teacher);counts=np.zeros(len(self.ids));rng=np.random.default_rng(9173)
            source=data.source_groups[self.ids];unique=np.unique(source)
            for repeat in range(8):
                bag=rng.choice(unique,max(2,int(.7*len(unique))),replace=False);use=np.isin(source,bag);held=np.flatnonzero(~use)
                m=instantiate(parent,data,self.ids[use],crossfit=False);cache={}
                for j in held:
                    case,k=data.train[self.ids[j]]
                    if case not in cache:cache[case]=normalized(m.predict_case(case))
                    teacher_sum[j]+=cache[case][k];counts[j]+=1
            use=counts>0;teacher[use]=teacher_sum[use]/counts[use,None];teacher_p=softmax(teacher,axis=1)
        if self.compress:
            method='opls' if op=='orthogonal_student' else 'simpls'
            self.student=Native(x,.65*target+.35*teacher_p,classes=self.classes,method=method)
            self.heads=self.student.heads;self.stages=1;self.required=set();self.parent=None
            return
        if spec.get('depth',0)>=4:
            # Bounded graph: compress its teacher first, then genuinely apply
            # this child's named structural innovation to the student.
            self.parent=StudentParent(Native(x,.65*target+.35*teacher_p,classes=self.classes),data,self.ids)
            self.required=set()
        self.heads=self.parent.heads;self.stages=self.parent.stages+1
        residual=target-teacher_p
        if op=='residual_specialist':
            self.head=Native(x,residual,classes=self.classes,components=16)
        elif op=='context_transport':
            self.head=Model(dict(engine='transport',view='mean'),data,self.ids)
        elif op in ('distribution_expert','spatial_expert'):
            self.view='quantile' if op=='distribution_expert' else 'spatial';self.required.add(self.view)
            self.head=Native(data.matrix(self.view,self.ids),residual,classes=self.classes,components=16)
        elif op=='hierarchical_decoder':self.head=Model(dict(engine='tree',view='mean'),data,self.ids)
        elif op=='verification_decoder':self.head=Model(dict(engine='verification',view='mean'),data,self.ids)
        elif op=='coded_residual':
            rng=np.random.default_rng(271828);self.code=rng.choice([-1.,1.],(len(self.classes),2*len(self.classes)))
            self.head=Native(x,residual@self.code,classes=self.classes,components=16)
        elif op=='local_residual':
            z,mu,sd=__import__('experiments.bora_evolution.models',fromlist=['standard']).standard(x)
            self.mu,self.sd=mu,sd;self.gate=KMeans(n_clusters=3,n_init=5,random_state=314159).fit(z)
            self.chain=[]
            for k in range(3):
                at=self.gate.labels_==k;self.chain.append(Native(x[at],residual[at],classes=self.classes,components=12))
            self.heads+=sum(m.heads for m in self.chain);self.head=None
        elif op=='latent_density_fusion':self.head=Model(dict(engine='density',view='mean'),data,self.ids)
        else:raise ValueError('Unimplemented structural operator '+op)
        if self.head is not None:self.heads+=self.head.heads
    def predict_case(self,case):
        x=self.data.case(case)
        if self.parent is None:return self.student.scores(x,self.classes)
        base=normalized(self.parent.predict_case(case));op=self.op
        if op=='local_residual':
            gate=self.gate.predict((x-self.mu)/self.sd);extra=np.zeros_like(base)
            for k,m in enumerate(self.chain):
                at=gate==k
                if at.any():extra[at]=m.scores(x[at],self.classes)
        elif op in ('distribution_expert','spatial_expert'):extra=self.head.scores(self.data.case(case,self.view),self.classes)
        elif op=='coded_residual':extra=self.head.scores(x,self.classes)@self.code.T/self.code.shape[1]
        elif isinstance(self.head,Model):extra=normalized(self.head.predict_case(case))
        else:extra=self.head.scores(x,self.classes)
        if op=='verification_decoder':
            # Verifier adjudicates the parent's three most plausible tissues.
            allow=np.zeros_like(base,bool);np.put_along_axis(allow,np.argsort(-base,axis=1)[:,:3],True,axis=1)
            return base+.6*np.where(allow,extra,0.)
        return base+.6*extra
    def training_scores(self):
        cache={};out=[]
        for i in self.ids:
            case,j=self.data.train[i]
            if case not in cache:cache[case]=self.predict_case(case)
            out.append(cache[case][j])
        return np.asarray(out)


class StudentParent:
    def __init__(self,head,data,ids):self.head=head;self.data=data;self.ids=ids;self.heads=head.heads;self.stages=1;self.required=set()
    def predict_case(self,case):return self.head.scores(self.data.case(case),self.data.classes)
    def training_scores(self):return self.head.scores(self.data.matrix('mean',self.ids),self.data.classes)
