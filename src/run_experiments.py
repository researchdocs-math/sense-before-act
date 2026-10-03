from __future__ import annotations
import os
for k in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time,hashlib,platform,sys,argparse,concurrent.futures
import numpy as np
from scipy.stats import beta
from benchmark import ROOT,prepare,Beliefs,episodes,evaluate,serializable_spec

def upper(k,n,delta):
    return 1. if k==n else float(beta.ppf(1-delta,k+1,n-k))

def stats(result):
    out={}
    for key,v in result.items():
        mean=float(np.mean(v));se=float(np.std(v,ddof=1)/np.sqrt(len(v)))
        out[key]={'mean':mean,'se':se,'ci95':[mean-1.96*se,mean+1.96*se]}
    k=int(result['risk'].sum());n=len(result['risk'])
    out['risk'].update(events=k,n=n,ci95=[float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.])
    return out

def task(name,method,cfg):
    start=time.perf_counter();spec=prepare(name);di=cfg['datasets'].index(name)
    belief=Beliefs(spec,fault=cfg['fault_probability'],independent=method=='independent_dp')
    cal=episodes(spec,cfg['calibration_episodes'],cfg['calibration_seed']+di,fault=cfg['fault_probability'],horizon=cfg['episode_length'],budgets=cfg['budgets'])
    total=len(cfg['datasets'])*len(cfg['methods'])*len(cfg['thresholds']);d=cfg['delta']/total
    rows=[];policies=[];raw={}
    for ti,tau in enumerate(cfg['thresholds']):
        policy=belief.policy(method,tau,max_budget=max(cfg['budgets']),price=cfg['cost_price']);policies.append(policy)
        r=evaluate(spec,policy,cal);k=int(r['risk'].sum())
        row={'tau':tau,'calibration':{key:float(v.mean()) for key,v in r.items()},'events':k,'upper':upper(k,len(r['risk']),d)};rows.append(row)
        for key,v in r.items():raw[f'{ti}_{key}']=v.astype(np.float32)
    selected={}
    for alpha in cfg['alphas']:
        eligible=[i for i,r in enumerate(rows) if r['upper']<=alpha]
        if eligible:
            best=max(eligible,key=lambda i:(rows[i]['calibration']['utility'],-rows[i]['tau']))
            selected[str(alpha)]=best if rows[best]['calibration']['utility']>0 else -1
        else:selected[str(alpha)]=-1
    # Selection is now frozen. Test outcomes cannot influence it.
    dest=ROOT/'results/main'/name/method;dest.mkdir(parents=True,exist_ok=True)
    (dest/'selection.json').write_text(json.dumps({'selected':selected,'candidates':rows,'delta_per_candidate':d,'family_size':total},indent=2))
    np.savez_compressed(dest/'calibration.npz',**raw)
    test=episodes(spec,cfg['test_episodes'],cfg['test_seed']+di,fault=cfg['fault_probability'],horizon=cfg['episode_length'],budgets=cfg['budgets'])
    raw_test={};frontier=[]
    for ti,policy in enumerate(policies):
        r=evaluate(spec,policy,test);frontier.append({'tau':cfg['thresholds'][ti],**stats(r)})
        if ti in selected.values():
            for key,v in r.items():raw_test[f'{ti}_{key}']=v.astype(np.float32)
            np.save(dest/f'policy_{ti}.npy',policy)
    reports={}
    for alpha,index in selected.items():
        if index<0:
            policy=np.full_like(policies[0],-1);r=evaluate(spec,policy,test)
            reports[alpha]={'fallback':True,'tau':None,'upper':0.,'test':stats(r)}
            for key,v in r.items():raw_test[f'-1_{key}']=v.astype(np.float32)
        else:reports[alpha]={'fallback':False,'tau':cfg['thresholds'][index],'upper':rows[index]['upper'],'cal_events':rows[index]['events'],'test':frontier[index]}
    stress=[]
    for mode in ['common','independent','cross_source']:
        for f in cfg['stress_faults']:
            data=episodes(spec,cfg['stress_episodes'],cfg['stress_seed']+di,fault=f,horizon=cfg['episode_length'],budgets=cfg['budgets'],mode=mode)
            for alpha,index in selected.items():
                policy=policies[index] if index>=0 else np.full_like(policies[0],-1)
                r=evaluate(spec,policy,data);stress.append({'mode':mode,'fault':f,'alpha':float(alpha),**stats(r)})
    np.savez_compressed(dest/'test_selected.npz',**raw_test)
    report={'dataset':name,'method':method,'selected':reports,'frontier':frontier,'stress':stress,'elapsed_seconds':time.perf_counter()-start,'spec':serializable_spec(spec)}
    (dest/'report.json').write_text(json.dumps(report,indent=2))
    return {'dataset':name,'method':method,'selected':{a:{'tau':r['tau'],'risk':r['test']['risk']['mean'],'completion':r['test']['completion']['mean'],'cost':r['test']['cost']['mean'],'upper':r['upper']} for a,r in reports.items()},'seconds':report['elapsed_seconds']}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    cfg=json.loads((ROOT/'configs/main.json').read_text());start=time.perf_counter()
    files=[ROOT/'src/benchmark.py',ROOT/'src/run_experiments.py',ROOT/'configs/main.json']
    frozen={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'python':sys.version,'platform':platform.platform(),'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'config':cfg}
    (ROOT/'results/frozen_protocol.json').write_text(json.dumps(frozen,indent=2))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures=[ex.submit(task,d,m,cfg) for d in cfg['datasets'] for m in cfg['methods']]
        for future in concurrent.futures.as_completed(futures):print(json.dumps(future.result()),flush=True)
    print('Elapsed seconds:',time.perf_counter()-start,flush=True)
