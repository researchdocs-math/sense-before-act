"""Select between two already-certified planners using calibration utility only."""
import json,shutil
from pathlib import Path
from benchmark import ROOT

def main():
 for name in ['synthetic','office','room']:
  reports={m:json.loads((ROOT/f'results/main/{name}/{m}/report.json').read_text()) for m in ['source_dp','independent_dp']}
  sels={m:json.loads((ROOT/f'results/main/{name}/{m}/selection.json').read_text()) for m in reports}
  out={'dataset':name,'method':'portfolio','selected':{}}
  dest=ROOT/f'results/main/{name}/portfolio';dest.mkdir(exist_ok=True)
  for alpha in ['0.01','0.05']:
   def utility(m):
    i=sels[m]['selected'][alpha]
    return sels[m]['candidates'][i]['calibration']['utility'] if i>=0 else 0.
   model=max(reports,key=lambda m:(utility(m),m=='independent_dp'))
   out['selected'][alpha]=reports[model]['selected'][alpha]|{'model':model,'calibration_utility':utility(model)}
  (dest/'report.json').write_text(json.dumps(out,indent=2))
  print(name,out['selected'])
if __name__=='__main__':main()
