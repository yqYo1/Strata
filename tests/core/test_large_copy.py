#!/usr/bin/env python3
"""Copy diagnostic admission must reject incomplete/invalid evidence."""
import copy,json,sys,unittest
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import large_copy as b
class CopyTests(unittest.TestCase):
    def rows(self):
        identity=dict(kind='identity',source='vector',bytes=b.BYTES,chunk=0,samples=6,unpressured=True,
            physical_bus_traffic_measured=False,GPU_service_measured=False,source_mod_65536=16,
            destination_mod_65536=0,setup_wall_s=1.,setup_cpu_s=.5,timer_max_s=.000001)
        samples=[dict(kind='sample',index=i,first_copy=i==0,copies=1,logical_bytes=b.BYTES,
            payload_match=True,guards_match=True,wall_s=.25,cpu_s=.225,
            stage_wall_s=.01,stage_cpu_s=.005,submit_wall_s=.02,submit_cpu_s=.02,
            drain_wall_s=.22,drain_cpu_s=.2)for i in range(6)]
        return [identity,*samples,dict(kind='closed',passed=True,owners_retired=True)]
    def parse(self,rows):return b.parse('\n'.join(json.dumps(x)for x in rows),'vector',0,6,b.BYTES,True)
    def test_valid_measurement_keeps_first_and_all_subsequent_samples(self):
        result=self.parse(self.rows());self.assertEqual(len(result['samples']),6)
        self.assertTrue(result['samples'][0]['first_copy']);self.assertEqual(result['subsequent_median_s'],.25)
    def test_reject_bad_identity_payload_closure_and_order(self):
        for variant in ('identity','guard','payload','bytes','copies','order','first','footer','geometry'):
            with self.subTest(variant=variant):
                r=self.rows()
                if variant=='identity':r[0]['source']='host-usm'
                if variant=='guard':r[1]['guards_match']=False
                if variant=='payload':r[2]['payload_match']=False
                if variant=='bytes':r[1]['logical_bytes']-=1
                if variant=='copies':r[1]['copies']=2
                if variant=='order':r[2]['index']=0
                if variant=='first':r[2]['first_copy']=True
                if variant=='footer':r.pop()
                if variant=='geometry':r[0].pop('source_mod_65536')
                with self.assertRaises((RuntimeError,KeyError)):self.parse(r)
    def test_reject_short_timer_overhead_nonfinite_and_nonconserving_clocks(self):
        for variant in ('short','overhead','nan','phase','cpu'):
            with self.subTest(variant=variant):
                r=self.rows()
                if variant=='short':r[1].update(wall_s=.05,stage_wall_s=.01,submit_wall_s=.02,drain_wall_s=.02,cpu_s=.03,stage_cpu_s=.01,submit_cpu_s=.01,drain_cpu_s=.01)
                if variant=='overhead':r[0]['timer_max_s']=.01
                if variant=='nan':r[1]['wall_s']=float('nan')
                if variant=='phase':r[1]['drain_wall_s']=.1
                if variant=='cpu':r[1]['cpu_s']=r[1]['drain_cpu_s']=2.
                with self.assertRaises(RuntimeError):self.parse(r)
    def test_selection_needs_three_cohorts_and_every_pair_faster(self):
        runs=[dict(passed=True,processes=[dict(source=s,chunk=c,subsequent_median_s=.3 if (s,c)==('vector',0)else .25)for s,c in b.CONFIGS])for _ in range(3)]
        self.assertIsNotNone(b.decision(runs)['selected'])
        for r in runs[0]['processes']:
            if (r['source'],r['chunk'])!=('vector',0):r['subsequent_median_s']=.3/1.01
        self.assertIsNone(b.decision(runs)['selected'])
        with self.assertRaises(RuntimeError):b.decision(runs[:2])
if __name__=='__main__':unittest.main()
