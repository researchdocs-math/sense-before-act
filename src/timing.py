import os
for k in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import time,json,platform
import numpy as np
from benchmark import *
from router import route
out=[]
for name in ['synthetic','office','room']:
 s=prepare(name);rep=json.loads((ROOT/f'results/main/{name}/portfolio/report.json').read_text())['selected']['0.01']
 model=rep['model'];tau=rep['tau'];build=[]
 for _ in range(5):
  t=time.perf_counter();b=Beliefs(s,independent=model=='independent_dp');policy=b.policy(model,tau);build.append(time.perf_counter()-t)
 data=episodes(s,20000,7100001,horizon=1);times=[]
 for i in range(20000):
  _,_,us=route(s,policy,data['z'][i,0],data['budget'][i,0]);times.append(us)
 out.append({'dataset':name,'policy_bytes':policy.nbytes,'table_states':len(policy),'build_median_seconds':float(np.median(build)),'routing_median_us':float(np.median(times)),'routing_p95_us':float(np.percentile(times,95)),'routing_p99_us':float(np.percentile(times,99)),'n':len(times),'platform':platform.platform()})
(ROOT/'results/cpu_timing.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
