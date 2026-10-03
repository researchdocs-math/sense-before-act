"""Produce machine-readable comparisons from saved experimental results."""
import json,csv
import numpy as np
from scipy.stats import t
from benchmark import ROOT

def main():
 results=ROOT/'results';summary={'main':[],'confirmatory':[],'paired_main':[]}
 for name in ['synthetic','office','room']:
  for method in ['portfolio','source_dp','independent_dp','source_mi','myopic','static','random','budget_full']:
   r=json.loads((results/f'main/{name}/{method}/report.json').read_text())
   for a,s in r['selected'].items():
    summary['main'].append({'dataset':name,'method':method,'alpha':float(a),'tau':s['tau'],'upper':s['upper'],**{k:v['mean'] for k,v in s['test'].items() if isinstance(v,dict)}})
  pr=json.loads((results/f'main/{name}/portfolio/report.json').read_text())['selected']['0.01'];model=pr['model']
  def raw(m):
   select=json.loads((results/f'main/{name}/{m}/selection.json').read_text())['selected']['0.01']
   f=np.load(results/f'main/{name}/{m}/test_selected.npz');return {k:f[f'{select}_{k}'].astype(float) for k in ['utility','cost','completion','risk']}
  a=raw(model)
  for method in ['independent_dp','source_mi','myopic']:
   b=raw(method)
   for key in a:
    diff=a[key]-b[key];mean=float(diff.mean());se=float(diff.std(ddof=1)/np.sqrt(len(diff)))
    summary['paired_main'].append({'dataset':name,'comparator':method,'metric':key,'mean_difference':mean,'ci95':[mean-1.96*se,mean+1.96*se]})
 reps=json.loads((results/'confirmatory_replications.json').read_text())
 for name in ['synthetic','office','room']:
  selected=[]
  for rep in range(20):
   rows=[x for x in reps if x['dataset']==name and x['rep']==rep and x['method'] in ['source_dp','independent_dp']]
   selected.append(max(rows,key=lambda x:x['calibration_utility']))
  for method in ['portfolio','source_dp','independent_dp','source_mi','myopic','static','random','budget_full']:
   rows=selected if method=='portfolio' else [next(x for x in reps if x['dataset']==name and x['rep']==rep and x['method']==method) for rep in range(20)]
   u=np.array([x['test']['utility'] for x in rows]);delta=np.array([a['test']['utility']-b['test']['utility'] for a,b in zip(selected,rows)])
   se=delta.std(ddof=1)/np.sqrt(20);ci=t.ppf(.975,19)*se
   summary['confirmatory'].append({'dataset':name,'method':method,**{k:float(np.mean([x['test'][k] for x in rows])) for k in rows[0]['test']},'portfolio_utility_difference':float(delta.mean()),'difference_ci95':[float(delta.mean()-ci),float(delta.mean()+ci)],'risk_exceedances':sum(x['test']['risk']>.01 for x in rows),'source_selections':sum(x['method']=='source_dp' for x in selected)})
 (results/'analysis.json').write_text(json.dumps(summary,indent=2))
 for key in ['main','paired_main','confirmatory']:
  with (results/(key+'_summary.csv')).open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(summary[key][0]));writer.writeheader();writer.writerows(summary[key])
 print(json.dumps(summary['confirmatory'],indent=2))
if __name__=='__main__':main()
