import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from benchmark import prepare,Beliefs,episodes,evaluate
from run_experiments import upper
class Invariants(unittest.TestCase):
    def test_common_fault_floor(self):
        s=dict(prior=.5,q=np.array([[0.,0.],[1.,1.]]),groups=np.array([0,0]),costs=np.ones(2,dtype=int))
        b=Beliefs(s,fault=.1);ind=Beliefs(s,fault=.1,independent=True)
        self.assertAlmostEqual(b.p[4],.1/1.1)
        self.assertAlmostEqual(ind.p[4],.01/1.01)
    def test_bayes_tower(self):
        b=Beliefs(prepare('synthetic'))
        for h in range(b.n):
            for j in np.flatnonzero(b.digits[h]==0):
                r=b.pred[h,j]
                self.assertAlmostEqual(b.p[h],(1-r)*b.p[b.child0[h,j]]+r*b.p[b.child1[h,j]],places=11)
    def test_budget_and_gate(self):
        s=prepare('synthetic');b=Beliefs(s);p=b.policy('source_dp',.01);data=episodes(s,1000,23)
        out=evaluate(s,p,data,trace=True)
        self.assertTrue(np.all(b.p[out['history'][out['allow_detail']]]<=.01+1e-12))
        for h in range(b.n):
            for budget,a in enumerate(p[h]):
                if a>=0:
                    self.assertEqual(b.digits[h,a],0);self.assertLessEqual(s['costs'][a],budget)
    def test_zero_event_bound(self):
        self.assertAlmostEqual(upper(0,100,.01),1-.01**.01)
    def test_partitions(self):
        for name in ['office','room']:
            s=prepare(name);a,b,c=[set(s['split_blocks'][x]) for x in ['fit','development','replay']]
            self.assertFalse(a&b or a&c or b&c)
            self.assertEqual(set(s['y_dev']),{0,1});self.assertEqual(set(s['y_pool']),{0,1})
if __name__=='__main__':unittest.main()
