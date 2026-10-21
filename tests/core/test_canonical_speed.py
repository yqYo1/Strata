#!/usr/bin/env python3
"""Protect independent repetitions and reject misleading improvement claims."""
import copy
import sys
import unittest
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import canonical_speed as c

class Decisions(unittest.TestCase):
    def rows(self,effect=1.0014):
        rows=[]
        for i,arm in enumerate(c.ORDER):
            samples=[]
            # Candidate first use is deliberately much slower. It must remain
            # visible without being pooled with the subsequent-request median.
            for run,tps in enumerate([500 if arm else 1500,*([1000*(effect if arm else 1)]*3)]):
                samples.append(dict(run=run,ok=True,eligible=True,stream_complete=True,
                    thinking_requested=True,reasoning_observed=True,actual_prompt_tokens=117489,
                    actual_read_tokens=117489,actual_reused_tokens=0,generated_tokens=24,
                    prompt_tps_source='engine',prompt_ms=117489000/tps,request_sha256=str(run)))
            rows.append(dict(arm=arm,passed=True,ownership_closed=True,fresh_faults=[],
                native_identity=dict(pid=i+10,start_ticks=i+100),requests=samples))
        return rows
    def test_first_use_does_not_mask_subsequent_effect(self):
        result=c.summarize(self.rows())
        self.assertTrue(result['candidate_faster_screen']);self.assertAlmostEqual(result['ratio_geomean'],1.0014)
        self.assertEqual(result['first_use_tok_s'],[1500,500,500,1500,1500,500])
        self.assertFalse(result['sustained_decode_qualified']);self.assertFalse(result['adoption_qualified'])
    def test_tiny_effect_or_one_slower_pair_is_not_improvement(self):
        self.assertFalse(c.summarize(self.rows(1.0001))['candidate_faster_screen'])
        rows=self.rows()
        for sample in rows[2]['requests'][1:]:sample['prompt_ms']=117489000/999
        self.assertFalse(c.summarize(rows)['candidate_faster_screen'])
    def test_failed_reordered_reused_or_mismatched_repetitions_rejected(self):
        rows=self.rows();bad=[]
        a=copy.deepcopy(rows);a[0]['passed']=False;bad.append(a)
        a=copy.deepcopy(rows);a[0],a[1]=a[1],a[0];bad.append(a)
        a=copy.deepcopy(rows);a[1]['native_identity']=a[0]['native_identity'];bad.append(a)
        a=copy.deepcopy(rows);a[1]['requests'][2]['request_sha256']='changed';bad.append(a)
        bad.append(rows[:-1])
        for a in bad:
            with self.assertRaises(RuntimeError):c.summarize(a)
    def test_invalid_or_reused_samples_cannot_qualify(self):
        for key,value in [('actual_reused_tokens',1),('actual_read_tokens',117488),
                          ('generated_tokens',0),('prompt_ms',float('nan')),('eligible',False)]:
            a=self.rows();a[1]['requests'][1][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):c.summarize(a)
    def test_target_requires_every_candidate_process(self):
        rows=self.rows();self.assertTrue(c.summarize(rows)['candidate1000_each_process'])
        for s in rows[5]['requests'][1:]:s['prompt_ms']=117489000/999.9
        self.assertFalse(c.summarize(rows)['candidate1000_each_process'])
    def test_launch_environments_hold_decode_and_safety_fixed(self):
        setup=c.base.setup_reader.pilot();root=c.B/'clean-host-environment-not-executed'
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
