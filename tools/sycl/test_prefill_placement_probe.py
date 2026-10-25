"""Host-only negative controls for observed placement attribution. Repository LICENSE."""
import copy
from pathlib import Path
import unittest
import prefill_placement_probe as p


class Attribution(unittest.TestCase):
    def fixture(self, owned):
        cells=117488
        sample=dict(actual_prompt_tokens=cells+1,actual_read_tokens=cells+1)
        warm='strata serve: prompt 74 tokens = 0 reused + 74 read, 1 generated\n'
        body=f'strata prefill: layer-major candidate, tokens={cells} chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_total={cells*20480} residual_gpu_rows={cells} residual_gpu={cells*20480} residual_host=0 residual_staging=0\n'
        body+='strata MoE window configuration: requested_mode=10 effective_mode=10\nstrata prefill dense: phase=restored residency=Full activations=48\n'
        body+=f'strata trace: read {cells} tokens (batched) in 120000.0 ms\n'
        if not owned:body+='strata trace: refilled 1124 slots on 1 stage(s) in 432.1 ms\n'
        return warm+body+f'strata serve: prompt {cells+1} tokens = 0 reused + {cells+1} read, 24 generated\n',[sample]

    def test_cache_tail_is_not_owned(self):
        log,samples=self.fixture(False)
        self.assertEqual(p.request_scopes(log,samples,False)[0]['expert_refill_ms'],432.1)
        with self.assertRaises(RuntimeError):p.request_scopes(log,samples,True)
        log,samples=self.fixture(True)
        self.assertEqual(p.request_scopes(log,samples,True)[0]['expert_refill_ms'],0)
        with self.assertRaises(RuntimeError):p.request_scopes(log,samples,False)

    def test_reject_spill_fallback_missing_restore_and_wrong_extent(self):
        log,samples=self.fixture(False)
        for changed in (log.replace('residual_host=0','residual_host=4096'),log.replace('chunk=8192','chunk=4096'),
            log.replace('activations=48','activations=47'),log.replace('read 117488 tokens','read 117487 tokens')):
            with self.assertRaises(RuntimeError):p.request_scopes(changed,samples,False)

    def test_owned_lease_has_no_borrowed_lifetime(self):
        row=dict(environment_template={'STRATA_MEASUREMENT_ROOT':'/old','STRATA_SYCL_CACHE_SCRATCH_LEASE':'1'})
        self.assertEqual(p.environment(row,Path('/new'),True)['STRATA_SYCL_CACHE_SCRATCH_LEASE'],'0')
        self.assertEqual(p.environment(row,Path('/new'),False)['STRATA_SYCL_CACHE_SCRATCH_LEASE'],'1')

    def test_host_spans_must_be_wholly_contained(self):
        interval=dict(request_begin_monotonic_ns=10,first_output_monotonic_ns=50)
        row=dict(begin=11,end=12,cpu_khz={'policy0':4200000},gpu_act_mhz=2750,gpu_requested_mhz=2750,
                 throttle=0,pswpin=5,pswpout=7,pgmajfault=20,memory_available=80<<30)
        rows=[dict(row,begin=9,end=12,pswpin=999),row,dict(row,begin=30,end=31,pswpin=8),dict(row,begin=45,end=51,pswpin=999)]
        value=p.host_summary(rows,interval)
        self.assertEqual(value['samples'],2);self.assertEqual(value['system_vm_delta']['pswpin'],3)
        with self.assertRaises(RuntimeError):p.host_summary(rows[:2],interval)


if __name__=='__main__':unittest.main()
