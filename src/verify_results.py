import json,sys
import numpy as np
from benchmark import ROOT,prepare,episodes,evaluate
from router import route
expected={'synthetic':(.0048,.368265,2.688975),'office':(.0044,.326955,2.862835),'room':(.00375,.340535,2.990575)}
for name,values in expected.items():
 p=ROOT/f'results/main/{name}'
 report=json.loads((p/'portfolio/report.json').read_text())['selected']['0.01'];result=report['test']
 np.testing.assert_allclose([result[k]['mean'] for k in ['risk','completion','cost']],values,rtol=0,atol=1e-12)
 method=report['model'];s=json.loads((p/method/'selection.json').read_text());idx=s['selected']['0.01']
 assert s['candidates'][idx]['upper']<=.01
 policy=np.load(p/method/f'policy_{idx}.npy');spec=prepare(name);data=episodes(spec,200,930001,horizon=1)
 vector=evaluate(spec,policy,data,trace=True)
 for i in range(200):
  allow,trace,_=route(spec,policy,data['z'][i,0],data['budget'][i,0])
  assert allow==bool(vector['allow_detail'][i]);assert sum(t['cost'] for t in trace)<=data['budget'][i,0]
  assert len({t['sensor'] for t in trace})==len(trace)
traces=json.loads((ROOT/'results/agent/traces.json').read_text())
for c in traces:
 assert not c['executed'] or (c['verified'] and c['permit'])
 assert sum(t['cost'] for t in c['trace'])<=c['budget']
summary=json.loads((ROOT/'results/agent/summary.json').read_text())
assert summary['blocked_attempts']==sum(c['blocked'] for c in traces)
assert summary['executed']==sum(c['executed'] for c in traces)
print('Verified primary metrics, calibration bounds, scalar/vector routing, budgets, and recorded interlock outcomes.')
