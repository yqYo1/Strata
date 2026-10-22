"""Admission evidence must reject partial, under-floor or inconsistent trials."""
import copy
from pathlib import Path
import sys
import unittest
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import cpu_iq2_indices as c

def records(arm,ratio=1.):
    samples=[dict(kind='sample',arm=arm,down=down,access=access,rep=rep,passes=10,jobs=10*(168 if access else 6),wall_s=1.2*ratio,cpu_s=1.19*ratio,timer_s=1e-8,warm_s=.001) for down in (20,42) for access in (0,1) for rep in range(5)]
    return [dict(kind='placement',cpu=0,computing_threads=1,arm=arm),*samples,dict(kind='closed',arm=arm,formats=2,experts_per_format=168,index_layout=True,bitwise=True,all_guards=True,inputs_unchanged=True,passed=True)]

class Evidence(unittest.TestCase):
    def test_positive_checks_and_complete_timing(self):
        for arm in (0,1):
            rows=records(arm)
            self.assertEqual(len(c.parse(rows,arm,'measure',0)),20)
            self.assertEqual(c.parse([rows[0],rows[-1]],arm,'check',0),[])

    def test_refusals(self):
        variants=[]
        def add(edit):
            rows=records(1);edit(rows);variants.append(rows)
        add(lambda x:x.pop())
        add(lambda x:x.pop(1))
        add(lambda x:x.insert(1,copy.deepcopy(x[1])))
        add(lambda x:x[0].update(cpu=1))
        add(lambda x:x[-1].update(bitwise=False))
        add(lambda x:x[-1].update(index_layout=False))
        add(lambda x:x[-1].update(inputs_unchanged=False))
        for field,value in [('arm',0),('wall_s',.99),('wall_s',float('nan')),('cpu_s',float('inf')),('passes',0),('passes',1000000),('jobs',1),('timer_s',1e-6),('warm_s',0)]:
            add(lambda x,k=field,v=value:x[1].update({k:v}))
        for rows in variants:
            with self.assertRaises((RuntimeError,KeyError,IndexError)):c.parse(rows,1,'measure',0)

    def test_independent_format_and_pair_gates(self):
        checks=[dict(arm=arm,samples=c.parse(records(arm,.95 if arm else 1),arm,'measure',0)) for arm in c.ORDER]
        self.assertTrue(all(r['performance_gate_passed'] for r in c.reduce(checks)))
        bad=copy.deepcopy(checks)
        for item in bad:
            if item['arm']:
                for sample in item['samples']:
                    if sample['down']==42:sample['wall_s']=1.3
        result=c.reduce(bad)
        self.assertTrue(result[0]['performance_gate_passed']);self.assertFalse(result[1]['performance_gate_passed'])
        with self.assertRaises(RuntimeError):c.reduce(checks[:-1])

if __name__=='__main__':unittest.main()
