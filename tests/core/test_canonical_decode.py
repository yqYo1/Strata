#!/usr/bin/env python3
"""Reject short/unbound decode evidence and prefill/decode tradeoffs."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import canonical_decode as c

class Decisions(unittest.TestCase):
    def rows(self,effect=1.001):
        rows=[]
        for i,arm in enumerate(c.ORDER):
            samples=[]
            for j,name in enumerate(c.WORKLOADS):
                tps=(14 if j==0 else 9)*(effect if arm else 1)
                n=2048;accepted=800 if j==0 else 300
                samples.append(dict(workload=name,run=0,order=j,ok=True,eligible=True,speed_eligible=True,
                    stream_complete=True,finish_reason='stop',reused_tokens=0,prompt_tokens=33214,
                    read_tokens=33214,usage_generated_tokens=n,generated_tokens=n,generated_tokens_source='usage',
                    output_chars=1024,reasoning_chars=200,completion_includes_thinking=True,
                    drafts_offered=1000,drafts_accepted=accepted,acceptance_rate=accepted/1000,
                    decode_tps_source='engine',decode_ms=n*1000/tps,decode_tps=round(tps,1),total_s=300,
                    request_sha256=str(j)*64))
            rows.append(dict(arm=arm,passed=True,complete=True,ownership_closed=True,
                fault_probe_successful=True,boot_unchanged=True,fresh_faults=[],
                native_identity=dict(pid=i+100,start_ticks=i+1000),requests=samples))
        return rows
    def test_exact_clocks_and_independent_workloads(self):
        result=c.summarize(self.rows())
        self.assertTrue(result['decode_preservation_screen']);self.assertTrue(result['MTP_pair_confirmed'])
        for w in result['workloads'].values():self.assertAlmostEqual(w['ratio_geomean'],1.001)
        self.assertFalse(result['candidate70_each_workload_process']);self.assertFalse(result['adoption_qualified'])
        self.assertFalse(result['zero_effect_statistically_proven'])
    def test_one_slower_pair_cannot_be_paid_for_by_other_workload(self):
        rows=self.rows(1.05);s=rows[2]['requests'][0];s['decode_ms']=s['generated_tokens']*1000/13.9
        result=c.summarize(rows)
        self.assertFalse(result['decode_preservation_screen'])
        self.assertTrue(result['workloads']['varied-prose']['no_decline_screen'])
    def test_classification_requires_every_sample_in_both_arms(self):
        for i,j,accepted in((0,0,590),(5,1,410)):
            rows=self.rows();s=rows[i]['requests'][j];s.update(drafts_accepted=accepted,acceptance_rate=accepted/1000)
            result=c.summarize(rows);self.assertFalse(result['MTP_pair_confirmed'])
            self.assertFalse(result['decode_preservation_screen'])
    def test_failed_reused_reordered_short_or_mismatched_evidence_rejected(self):
        mutations=[lambda r:r[0].update(passed=False),lambda r:r[0].update(boot_unchanged=False),
            lambda r:r[0].update(fresh_faults=['fault']),
            lambda r:r[1].update(native_identity=r[0]['native_identity']),
            lambda r:r[1]['requests'][0].update(request_sha256='changed'),
            lambda r:r[2]['requests'][0].update(reasoning_chars=0),
            lambda r:r[2]['requests'][0].update(generated_tokens=24,usage_generated_tokens=24),
            lambda r:r[2]['requests'][0].update(decode_ms=float('nan')),
            lambda r:r[2]['requests'][0].update(reused_tokens=1),
            lambda r:r[2]['requests'].reverse(),lambda r:r.reverse(),lambda r:r.pop()]
        for mutate in mutations:
            rows=self.rows();mutate(rows)
            with self.assertRaises(RuntimeError):c.summarize(rows)
    def test_target_requires_every_candidate_workload_process(self):
        rows=self.rows(10);self.assertTrue(c.summarize(rows)['candidate70_each_workload_process'])
        s=rows[5]['requests'][1];s['decode_ms']=s['generated_tokens']*1000/69.9
        self.assertFalse(c.summarize(rows)['candidate70_each_workload_process'])
    def test_native_binding_gpu_extent_and_clean_observers(self):
        text='strata generate: expert cache 1256 slots, fixture\n'
        text+='strata serve: the prompt path borrows 1124 CUDA0 cache slots fixture\n';events=[]
        for i,n in enumerate((10,20,30,40,33214,33288)):
            if i>=4:
                cells=n-1;size=cells*20480
                text+=f'strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows={cells} mixed_fp16_bytes=0\n'
                text+=f'strata prefill: layer-major candidate, tokens={cells} chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_total={size} residual_gpu_rows={cells} residual_gpu={size} residual_host=0 residual_staging=0 layer_cache=1\n'
            text+=f'strata serve: prompt {n} tokens = 0 reused + {n} read in 1 ms (1 tok/s), 1024 generated in 1 ms (1 tok/s), drafts accepted 800 of 1000, 0 checkpoints\n'
            events.append(dict(kind='result',sample=dict(prompt_tokens=n,reused_tokens=0,read_tokens=n,generated_tokens=1024,
                prompt_ms=1,decode_ms=1,drafts_accepted=800,drafts_offered=1000)))
        with patch.object(c.base.history,'admitted_loan',return_value=dict(first=132,lent=1124)):
            good=c.native_routes(text,events);self.assertEqual(len(good['requests']),2)
            for bad in(text.replace('residual_gpu_rows=33213','residual_gpu_rows=1'),
                        text.replace('effective_mode=10','effective_mode=2'),
                        text+'strata canonical restore: unexpected\n',
                        text.replace('prompt 33214 tokens','prompt 33215 tokens'),
                        text.replace('generated in 1 ms','generated in 4 ms'),
                        text.replace('accepted 800 of','accepted 799 of'),
                        text.replace('cache 1256 slots','cache 128 slots')):
                with self.assertRaises(RuntimeError):c.native_routes(bad,events)
    def test_environment_has_only_prefill_selector_difference(self):
        setup=c.base.setup_reader.pilot();root=c.B/'decode-host-environment-not-executed'
        a,b=[c.environment(setup,root,x)for x in(0,1)]
        self.assertEqual({k for k in a.keys()|b.keys()if a.get(k)!=b.get(k)},
            {'STRATA_SYCL_PREFILL_CANONICAL_CHUNK64'})
        for e in(a,b):
            self.assertEqual(e['STRATA_IQ2_PAIR'],'1');self.assertEqual(e['STRATA_MTP_BATCH_F16'],'0')
            self.assertEqual(e['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'],'1');self.assertEqual(e['EnableDirectSubmission'],'0')
            self.assertEqual(e['STRATA_SYCL_PREFILL_CANONICAL_RESTORE_AUDIT'],'0')
            self.assertNotIn('UR_ENABLE_LAYERS',e);self.assertNotIn('STRATA_TRACE',e)
            self.assertNotIn('STRATA_SNAPSHOT_VERIFY',e);self.assertNotIn('STRATA_CPU_WORK_CENSUS',e)
if __name__=='__main__':unittest.main()
