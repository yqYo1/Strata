#!/usr/bin/env python3
"""CPU-only negative controls for runtime crossover misattribution."""
import copy
import unittest
import prefill_runtime_compare as c


class Controls(unittest.TestCase):
    def test_provider_must_be_selected_and_exclusive(self):
        expected='/private/libze_intel_gpu.so.old';other='/system/libze_intel_gpu.so.new'
        c.check_maps({expected},{expected},expected)
        for seen,allowed in [({other},{expected,other}),({expected,other},{expected,other}),(set(),{expected}),({'/alien'}, {expected})]:
            with self.assertRaises(RuntimeError):c.check_maps(seen,allowed,expected)

    def test_loader_startup_without_driver_is_not_execution_proof(self):
        c.check_maps(set(),set(),'/private/driver',False)
        with self.assertRaises(RuntimeError):c.check_maps(set(),set(),'/private/driver')

    def sample(self):
        return dict(ok=True,eligible=True,stream_complete=True,thinking_requested=True,reasoning_observed=True,
            request_sha256='wire',actual_prompt_tokens=117489,actual_read_tokens=117489,actual_reused_tokens=0,
            generated_tokens=24,prompt_tps_source='engine',prompt_ms=120000)

    def test_workload_and_timing_changes_rejected(self):
        good=[self.sample() for _ in range(4)];self.assertEqual(len(c.rates(good,['wire']*4)),4)
        for key,value in [('actual_read_tokens',117488),('actual_reused_tokens',1),('generated_tokens',23),
            ('thinking_requested',False),('prompt_tps_source','client'),('prompt_ms',float('nan')),('request_sha256','other')]:
            bad=copy.deepcopy(good);bad[1][key]=value
            with self.assertRaises(RuntimeError):c.rates(bad,['wire']*4)

    def processes(self,old=1003,new=974):
        return [dict(arm=arm,passed=True,ownership_closed=True,tok_s=[500,*([old if arm==0 else new]*3)]) for arm in c.ORDER]

    def test_first_use_does_not_enter_subsequent_comparison(self):
        a=self.processes();x=c.summarize(a);a[0]['tok_s'][0]=100000
        self.assertEqual(x,c.summarize(a));self.assertTrue(x['old_faster_screen'])

    def test_inconsistent_or_small_effect_is_unresolved(self):
        self.assertFalse(c.summarize(self.processes(1000,999))['old_faster_screen'])
        a=self.processes();a[5]['tok_s']=[500,900,900,900]
        self.assertFalse(c.summarize(a)['old_faster_screen'])

    def test_failed_or_reordered_process_cannot_qualify(self):
        for edit in ('failed','order','incomplete'):
            a=self.processes()
            if edit=='failed':a[2]['passed']=False
            elif edit=='order':a[2]['arm']=1
            else:a.pop()
            with self.assertRaises(RuntimeError):c.summarize(a)


if __name__=='__main__':unittest.main()
