"""Reject incomplete phase/route evidence and per-format hidden regressions."""
import copy
from pathlib import Path
import sys
import unittest
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import iq2_indices_pool as c

def records(arm,process=0,ratio=1.):
    samples=[]
    for down in (20,42):
        for mix in (0,1):
            for access in (0,1):
                for rep in range(5):
                    jobs,entries,ones=(24,36,16) if mix else (28,35,22)
                    calls=10*(168//jobs if access else 1);rows=calls*640*ones
                    samples.append(dict(kind='sample',process=process,arm=arm,down=down,mixture=mix,access=access,rep=rep,passes=10,batch_calls=calls,wall_s=1.2*ratio,cpu_s=6*ratio,GU_s=.7*ratio,quant_s=.1*ratio,Down_s=.2*ratio,row_ops=calls*640*entries,NT1_rows=rows,paired_rows=rows,wrapped_rows=rows,other_rows=0,min_thread_rows=1,timer_s=1e-8,warm_pass_s=.001))
    return [dict(kind='placement',workers=5,host_works=True,host_tid=100,threads=[[100+i,i] for i in range(6)]),*samples,dict(kind='validation',contract_cases=0,experts_per_format=168,formats=2,mixtures=2,bitwise=True,prefill_isolated=True,index_layout=True,passed=True),dict(kind='closed',passed=True)]

class Evidence(unittest.TestCase):
    def test_complete_numeric_and_timing_framing(self):
        for arm in (0,1):
            rows=records(arm)
            self.assertEqual(len(c.parse(rows,arm,'measure',range(6),0)),40)
            check=[rows[0],dict(rows[-2],contract_cases=60),rows[-1]]
            self.assertEqual(c.parse(check,arm,'check',range(6),0),[])

    def test_reject_incomplete_or_forged_route(self):
        variants=[]
        def add(edit):
            rows=records(1);edit(rows);variants.append(rows)
        add(lambda x:x.pop())
        add(lambda x:x.pop(1))
        add(lambda x:x.insert(1,copy.deepcopy(x[1])))
        add(lambda x:x[0]['threads'].__setitem__(0,[101,0]))
        add(lambda x:x[-2].update(prefill_isolated=False))
        add(lambda x:x[-2].update(index_layout=False))
        add(lambda x:x[-2].update(bitwise=False))
        for k,v in [('arm',0),('process',1),('wall_s',.999),('wall_s',float('nan')),('cpu_s',float('inf')),('GU_s',3),('paired_rows',0),('wrapped_rows',0),('other_rows',1),('min_thread_rows',0),('passes',0),('batch_calls',1),('timer_s',1e-6),('warm_pass_s',0)]:
            add(lambda x,key=k,val=v:x[1].update({key:val}))
        for rows in variants:
            with self.assertRaises((RuntimeError,KeyError,IndexError)):c.parse(rows,1,'measure',range(6),0)

    def test_independent_format_and_bad_pair_gates(self):
        checks=[dict(arm=arm,samples=c.parse(records(arm,i//2,.95 if arm else 1),arm,'measure',range(6),i//2)) for i,arm in enumerate(c.ORDER)]
        self.assertTrue(all(d['performance_gate_passed'] for d in c.reduce(checks)))
        bad=copy.deepcopy(checks)
        for item in bad:
            if item['arm']:
                for sample in item['samples']:
                    if sample['down']==42:sample['wall_s']=1.3
        result=c.reduce(bad);self.assertTrue(result[0]['performance_gate_passed']);self.assertFalse(result[1]['performance_gate_passed'])
        pair=copy.deepcopy(checks)
        for sample in pair[-1]['samples']:
            if sample['down']==20:sample['wall_s']=1.206
        # Overall geomean stays under0.97,but the last process pair is slower.
        d=c.reduce(pair)[0];self.assertLess(d['geomean'],.97);self.assertFalse(d['performance_gate_passed'])
        with self.assertRaises(RuntimeError):c.reduce(checks[:-1])

if __name__=='__main__':unittest.main()
