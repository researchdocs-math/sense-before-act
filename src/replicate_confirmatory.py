import os
for k in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import json,time,concurrent.futures
from benchmark import *
from run_experiments import upper,stats

def replicate(dataset,method,rep,cfg):
    s=prepare(dataset,20271003+10000*(rep+1));di=cfg['datasets'].index(dataset)
    b=Beliefs(s,independent=method=='independent_dp');ps=[b.policy(method,t,max_budget=8) for t in cfg['thresholds']]
    cal=episodes(s,10000,4000000+10000*rep+di);d=.05/(20*3*7*20);rs=[]
    for p in ps:
        r=evaluate(s,p,cal);rs.append((upper(int(r['risk'].sum()),10000,d),r['utility'].mean()))
    eligible=[i for i,(u,v) in enumerate(rs) if u<=.01 and v>0]
    best=max(eligible,key=lambda i:rs[i][1]) if eligible else -1
    data=episodes(s,5000,5000000+10000*rep+di)
    r=evaluate(s,ps[best] if best>=0 else np.full_like(ps[0],-1),data)
    return {'dataset':dataset,'method':method,'rep':rep,'partition_seed':20271003+10000*(rep+1),'selected_index':best,'calibration_utility':float(rs[best][1]) if best>=0 else 0.,'upper':rs[best][0] if best>=0 else 0.,'test':{k:float(v.mean()) for k,v in r.items()},'spec':serializable_spec(s)}
if __name__=='__main__':
    cfg=json.loads((ROOT/'configs/main.json').read_text());out=[];start=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:
        fs=[ex.submit(replicate,d,m,r,cfg) for r in range(20) for d in cfg['datasets'] for m in cfg['methods']]
        for f in concurrent.futures.as_completed(fs):
            x=f.result();out.append(x);print(x['rep'],x['dataset'],x['method'],x['test'],flush=True)
    (ROOT/'results/confirmatory_replications.json').write_text(json.dumps(sorted(out,key=lambda x:(x['rep'],x['dataset'],x['method'])),indent=2))
    print('elapsed',time.perf_counter()-start)
