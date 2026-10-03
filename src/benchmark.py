"""Source-aware sensing benchmark. All randomness is explicit and replayable."""
from __future__ import annotations
from pathlib import Path
import json, itertools, time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

ROOT=Path(__file__).resolve().parents[1]
METHODS=['source_dp','independent_dp','source_mi','myopic','static','random','budget_full']
THRESHOLDS=[0.0001,0.0003,0.001,0.003,0.006,0.01,0.02,0.04,0.08,0.16]
PRICE=0.015

def prepare(name, partition_seed=20271003):
    """Disjoint 100-row blocks with 20-row gaps; fixed 40/20/40 allocation."""
    if name=='synthetic':
        q=np.array([[.10,.16,.05,.08,.02,.04],[.88,.80,.94,.90,.98,.95]])
        return dict(name=name,q=q,prior=.35,groups=np.repeat(np.arange(3),2),costs=np.repeat([1,2,3],2),features=[f'sensor_{i}' for i in range(6)],fit_rows=0,development_rows=0,replay_rows=0)
    if name=='office':
        path=ROOT/'data/raw/occupancy_detection'
        df=pd.concat([pd.read_csv(p,index_col=0) for p in sorted(path.glob('*.txt'))])
        df=df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
        features=['Temperature','Humidity','HumidityRatio','Light','CO2']
        groups=np.array([0,0,0,1,2]);costs=np.array([1,1,1,2,3]);y=df.Occupancy.to_numpy().astype(int)
    elif name=='room':
        df=pd.read_csv(ROOT/'data/raw/room_occupancy/Occupancy_Estimation.csv')
        df=df.sort_values(['Date','Time']).reset_index(drop=True)
        features=['S1_Light','S1_Sound','S2_Light','S2_Sound','S5_CO2','S5_CO2_Slope']
        groups=np.repeat(np.arange(3),2);costs=np.repeat([1,2,3],2)
        y=(df.Room_Occupancy_Count.to_numpy()>0).astype(int)
    else: raise ValueError(name)
    n=len(df);blocks=np.arange(n//120);rng=np.random.default_rng(partition_seed);rng.shuffle(blocks)
    a=int(.4*len(blocks));b=int(.6*len(blocks))
    parts=[(bs[:,None]*120+np.arange(100)).reshape(-1) for bs in [blocks[:a],blocks[a:b],blocks[b:]]]
    fit,dev,pool=parts;z=[];model_specs=[]
    for f in features:
        x=df[[f]].to_numpy()
        clf=DecisionTreeClassifier(max_depth=2,min_samples_leaf=50,class_weight='balanced',random_state=19).fit(x[fit],y[fit])
        z.append(clf.predict(x).astype(int))
        model_specs.append({'feature':f,'children_left':clf.tree_.children_left.tolist(),'children_right':clf.tree_.children_right.tolist(),'threshold':clf.tree_.threshold.tolist(),'value':clf.tree_.value.tolist()})
    z=np.stack(z,axis=1)
    q=np.array([[(z[dev,j][y[dev]==s].sum()+1)/((y[dev]==s).sum()+2) for j in range(len(features))] for s in [0,1]])
    return dict(name=name,q=q,prior=float((y[dev].sum()+1)/(len(dev)+2)),groups=groups,costs=costs,features=features,fit_rows=len(fit),development_rows=len(dev),replay_rows=len(pool),z_dev=z[dev],y_dev=y[dev],z_pool=z[pool],y_pool=y[pool],split_blocks={'fit':blocks[:a].tolist(),'development':blocks[a:b].tolist(),'replay':blocks[b:].tolist()},model_specs=model_specs)

class Beliefs:
    def __init__(self, spec, fault=.08, independent=False):
        self.m=m=len(spec['groups']);self.powers=3**np.arange(m);self.n=3**m
        self.digits=(np.arange(self.n)[:,None]//self.powers)%3
        self.costs=spec['costs'];self.count=(self.digits>0).sum(axis=1)
        prior=spec['prior'];q=spec['q'];g=spec['groups']
        if independent:
            likelihood=(1-fault)*q
            states=np.array([0,1]);weights=np.array([1-prior,prior])
        else:
            fs=np.array(list(itertools.product([0,1],repeat=int(g.max())+1)))
            states=np.repeat([0,1],len(fs));f=np.tile(fs,(2,1))
            weights=np.where(states,prior,1-prior)*np.prod(np.where(f,fault,1-fault),axis=1)
            likelihood=q[states]*(1-f[:,g])
        w=np.tile(weights,(self.n,1))
        for j in range(m):
            d=self.digits[:,j,None]
            w*=np.where(d==0,1,np.where(d==1,1-likelihood[:,j],likelihood[:,j]))
        totals=w.sum(axis=1)
        w=np.divide(w,totals[:,None],out=np.zeros_like(w),where=totals[:,None]>0)
        self.p=w@states;self.pred=w@likelihood
        self.child0=np.arange(self.n)[:,None]+self.powers
        self.child1=np.arange(self.n)[:,None]+2*self.powers
        self.order=np.argsort(-self.count,kind='stable')
        self.spec=spec
    def policy(self,method,tau,max_budget=8,price=PRICE):
        action=np.full((self.n,max_budget+1),-1,dtype=np.int16)
        value=np.zeros((self.n,max_budget+1));mi=np.zeros((self.n,self.m))
        def ent(p):
            p=np.clip(p,1e-15,1-1e-15);return -p*np.log(p)-(1-p)*np.log(1-p)
        if method not in ['source_dp','independent_dp']:
            for h in range(self.n):
                for j in np.flatnonzero(self.digits[h]==0):
                    c0=self.child0[h,j];c1=self.child1[h,j];r=self.pred[h,j]
                    mi[h,j]=max(0,ent(self.p[h])-(1-r)*ent(self.p[c0])-r*ent(self.p[c1]))/self.costs[j]
        static_order=np.argsort(-mi[0],kind='stable')
        random_order=np.random.default_rng(234).permutation(self.m)
        for h in self.order:
            p=self.p[h]
            for budget in range(max_budget+1):
                js=np.flatnonzero((self.digits[h]==0)&(self.costs<=budget))
                if method in ['source_dp','independent_dp']:
                    best=1-p if p<=tau else 0.;a=-2 if p<=tau else -1
                    for j in js:
                        b2=budget-self.costs[j];r=self.pred[h,j]
                        v=(1-r)*value[self.child0[h,j],b2]+r*value[self.child1[h,j],b2]-price*self.costs[j]
                        if v>best+1e-12:best=float(v);a=int(j)
                    value[h,budget]=best;action[h,budget]=a
                else:
                    if method!='budget_full' and p<=tau:action[h,budget]=-2;continue
                    if method!='budget_full' and p>=1-tau:continue
                    if not len(js):
                        action[h,budget]=-2 if p<=tau else -1;continue
                    if method=='source_mi':a=js[np.argmax(mi[h,js])]
                    elif method=='myopic':
                        scores=[]
                        for j in js:
                            r=self.pred[h,j];p0=self.p[self.child0[h,j]];p1=self.p[self.child1[h,j]]
                            v=((1-r)*(1-p0)*(p0<=tau)+r*(1-p1)*(p1<=tau))/self.costs[j]
                            scores.append(v)
                        a=js[np.argmax(scores)] if max(scores)>0 else js[np.argmax(mi[h,js])]
                    else:
                        order=random_order if method=='random' else static_order
                        a=next(j for j in order if j in js)
                    action[h,budget]=a
        return action

def episodes(spec,n,seed,fault=.08,horizon=10,development=False,budgets=(4,6,8),mode='common'):
    rng=np.random.default_rng(seed);m=len(spec['groups']);g=spec['groups'];ng=int(g.max())+1
    if spec['name']=='synthetic':
        y=(rng.random((n,horizon))<spec['prior']).astype(np.int8)
        z=(rng.random((n,horizon,m))<spec['q'][y]).astype(np.int8)
    else:
        zp=spec['z_dev'] if development else spec['z_pool'];yp=spec['y_dev'] if development else spec['y_pool']
        starts=100*rng.integers(0,len(yp)//100,size=n)+rng.integers(0,100-horizon+1,size=n)
        indices=starts[:,None]+np.arange(horizon)
        y=yp[indices];z=zp[indices].copy()
    if mode=='common':
        f=rng.random((n,ng))<fault
        z=np.where(f[:,None,g],0,z)
    elif mode=='independent':
        f=rng.random((n,horizon,m))<fault;z=np.where(f,0,z)
    elif mode=='cross_source':
        f=rng.random((n,1))<fault;z=np.where(f[:,:,None],0,z)
    else:raise ValueError(mode)
    budget=rng.choice(budgets,size=(n,horizon))
    return dict(y=y.astype(np.int8),z=z.astype(np.int8),budget=budget.astype(np.int8))

def evaluate(spec,policy,data,trace=False):
    n,H=data['y'].shape;m=len(spec['groups']);powers=3**np.arange(m)
    y=data['y'].ravel();z=data['z'].reshape(-1,m);b=data['budget'].ravel().copy()
    history=np.zeros(len(y),dtype=int);active=np.ones(len(y),dtype=bool);allow=np.zeros(len(y),dtype=bool)
    calls=np.zeros(len(y),dtype=int);cost=np.zeros(len(y),dtype=int);sources=np.zeros((len(y),int(spec['groups'].max())+1),dtype=bool)
    traces=[]
    for t in range(m+1):
        ids=np.flatnonzero(active)
        if not len(ids):break
        a=policy[history[ids],b[ids]]
        done=a<0;allow[ids[a==-2]]=True;active[ids[done]]=False
        qi=ids[~done];j=a[~done]
        if len(qi):
            if trace:traces.append((qi.copy(),j.copy(),z[qi,j].copy()))
            history[qi]+=(1+z[qi,j])*powers[j]
            b[qi]-=spec['costs'][j];calls[qi]+=1;cost[qi]+=spec['costs'][j];sources[qi,spec['groups'][j]]=True
    assert not active.any()
    unsafe=(allow&(y==1)).reshape(n,H);safe=(allow&(y==0)).reshape(n,H)
    output={'risk':unsafe.any(axis=1).astype(float),'completion':safe.mean(axis=1),'allow':allow.reshape(n,H).mean(axis=1),'calls':calls.reshape(n,H).mean(axis=1),'cost':cost.reshape(n,H).mean(axis=1),'source_count':sources.sum(axis=1).reshape(n,H).mean(axis=1),'opportunities':(y==0).reshape(n,H).mean(axis=1)}
    output['utility']=output['completion']-PRICE*output['cost']
    if trace:output.update(history=history,allow_detail=allow,traces=traces)
    return output

def serializable_spec(spec):
    return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in spec.items() if k not in ['z_dev','y_dev','z_pool','y_pool']}
