"""Main-authored capacity admission/decision controls; repository LICENSE."""
import copy
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import cpu_pool_capacity as cap


class CapacityTests(unittest.TestCase):
    def plan(self,index):
        arm=cap.ARMS[index]
        return dict(index=index,kernel=arm,pool_workers=5 if arm=='reference' else 1,
            root='/tmp/capacity-owned-'+str(index),heads=1,adapt_swaps=0)

    def test_configuration_is_only_worker_delta(self):
        row=dict(configuration=dict(exe=str(cap.P/'tools/sycl/engine_once.py'),
            args=['--pool-workers','5','--spec','4','--pcie-frac','0','--adapt-swaps','0']))
        original=copy.deepcopy(row);values=[]
        for index in (0,1):
            plan=self.plan(index);value=cap.configuration(row,plan)
            self.assertEqual(value['args'][1],str(plan['pool_workers']))
            value['args'][1]='<workers>';value.pop('log',None);values.append(value)
        self.assertEqual(row,original);self.assertEqual(*values)
        with mock.patch.object(cap.kv.base,'environment',return_value={
                'STRATA_SYCL_DECODE_KV_COALESCED':'1','STRATA_SYCL_KV_COPY_GROUPS':'1'}):
            self.assertEqual(cap.environment(row,self.plan(0)),cap.environment(row,self.plan(1)))
            self.assertEqual(set(cap.environment(row,self.plan(1)).values()),{'0'})
        bad=copy.deepcopy(row);bad['configuration']['exe']='/tmp/strata'
        with self.assertRaises(RuntimeError):cap.configuration(bad,self.plan(0))
        bad=copy.deepcopy(row);bad['configuration']['args']+=['--pool-workers','5']
        with self.assertRaises(RuntimeError):cap.configuration(bad,self.plan(0))
        for field,value in [('pool_workers',True),('pool_workers',2),('kernel','reference'),('index',0)]:
            plan=self.plan(1);plan[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(RuntimeError):cap.workers(plan)

    def test_actual_pool_footer_is_required(self):
        for workers in (1,5):
            line=f'strata generate: {workers} expert-pool workers + the host thread'
            out=cap.pool_binding('unrelated\n'+line+'\n',workers)
            self.assertEqual(out['potential_participants'],workers+1)
            self.assertEqual(out['source_defined_row_tasks_per_native_GU_or_Down_phase'],3*(workers+1))
            for wrong in ['',line+'\n'+line,line.replace(' + the host thread',''),
                    line+' (UNUSED: --no-pool)',line.replace(str(workers),str(6-workers))]:
                with self.subTest(workers=workers,text=wrong),self.assertRaises(RuntimeError):
                    cap.pool_binding(wrong,workers)

    def cohort(self):
        rows=[]
        for i,arm in enumerate(cap.ARMS):
            worker=5 if arm=='reference' else 1
            samples=[dict(native_generated_tokens=1024,exact_native_decode_tps=10 if worker==5 else 5,
                prompt_ms=1000,ttft_s=1.2,acceptance_rate=.8 if j==0 else .34,
                request_sha256=str(j)*64) for j in range(2)]
            rows.append(dict(kernel=arm,passed=True,complete=True,ownership_closed=True,
                native_exit_code=0,fresh_faults=[],native_identity=dict(pid=100+i,start_ticks=200+i),
                samples=samples,API_native_binding=[dict(cpu_pool=dict(workers=worker),
                    routes=[dict(CPU_entries=200,CPU_unique_jobs=150)]) for _ in range(2)]))
        return rows

    def test_decision_keeps_directions_work_and_invalid_cohorts(self):
        rows=self.cohort();out=cap.decision(rows)
        for workload in cap.kv.WORKLOADS:
            value=out['workloads'][workload]
            self.assertTrue(value['fewer_workers_repeatedly_slower'])
            self.assertTrue(value['reference_MTP_class_confirmed'])
            self.assertEqual(value['geometric_mean_TG_ratio'],.5)
            self.assertTrue(all(p['worker1_over_worker5_jobs_per_emitted']==1 for p in value['pairs']))
        self.assertFalse(out['physical_saturation_qualified'])
        mixed=copy.deepcopy(rows);mixed[2]['samples'][0]['exact_native_decode_tps']=20
        self.assertTrue(cap.decision(mixed)['workloads']['regular-code']['mixed_or_equal_directions'])
        invalid=[rows[:-1]]
        bad=copy.deepcopy(rows);bad[1]['samples'][0]['exact_native_decode_tps']=float('nan');invalid.append(bad)
        bad=copy.deepcopy(rows);bad[1]['API_native_binding'][0]['cpu_pool']['workers']=5;invalid.append(bad)
        bad=copy.deepcopy(rows);bad[1]['samples'][0]['request_sha256']='wrong';invalid.append(bad)
        bad=copy.deepcopy(rows);bad[1]['native_identity']=copy.deepcopy(bad[0]['native_identity']);invalid.append(bad)
        bad=copy.deepcopy(rows);bad[1]['API_native_binding'][0]['routes'][0]['CPU_entries']=0;invalid.append(bad)
        for bad in invalid:
            with self.subTest(case=invalid.index(bad)),self.assertRaises(RuntimeError):cap.decision(bad)


if __name__=='__main__':
    unittest.main()
