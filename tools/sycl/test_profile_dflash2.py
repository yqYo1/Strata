"""CPU refusal controls for current-model attribution, without GPU work."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import profile_dflash2 as profile


def cases():
    result=[]
    for label,value in zip(profile.ORDER,(10.,11.,12.)):
        result.append(dict(label=label,samples=[dict(request_sha256=str(i)*64,generated_tokens=24,
            prompt_ms=value,decode_ms=value,ttft_s=value) for i in (1,2)]))
    return result


class ProfileDflash2Test(unittest.TestCase):
    def test_warmup_and_both_exact_request_positions_are_required(self):
        rows=[]
        for phase,run in (('warm-up/chat',None),('prefill-only',0),('prefill-only',1)):
            index=len(rows);digest=str(index+1)*64
            rows.extend([dict(kind='start',phase=phase,run=run,turn=None,index=index,
                max_tokens=24,thinking=True,request_sha256=digest),
                dict(kind='result',phase=phase,run=run,turn=None,request_index=index,
                     sample=dict(ok=True,request_sha256=digest))])
        self.assertEqual(profile.journal(rows)['warmups'],1)
        self.assertEqual(profile.journal(rows)['long_requests'],2)
        bad=copy.deepcopy(rows);bad[2]['thinking']=False
        with self.assertRaises(RuntimeError):profile.journal(bad)
        with self.assertRaises(RuntimeError):profile.journal(rows[2:])
        bad=copy.deepcopy(rows);bad[-1]['sample']['request_sha256']='e'*64
        with self.assertRaises(RuntimeError):profile.journal(bad)

    def test_observer_cost_keeps_each_phase_and_request_separate(self):
        rows=cases();rows[1]['samples'][0]['prompt_ms']=17
        decision=profile.observer_decision(rows)
        self.assertFalse(decision['observer_screen_passed'])
        first,second=decision['requests']
        self.assertFalse(first['phases']['prompt_ms']['screen_passed'])
        self.assertTrue(first['phases']['decode_ms']['screen_passed'])
        self.assertTrue(first['phases']['ttft_s']['screen_passed'])
        self.assertTrue(second['observer_screen_passed'])
        rows=cases();rows[1]['samples'][0]['generated_tokens']=23
        self.assertFalse(profile.observer_decision(rows)['requests'][0]['fixed_24_output_work'])

    def test_changed_body_or_unclocked_sample_never_qualifies(self):
        rows=cases();self.assertTrue(profile.observer_decision(rows)['observer_screen_passed'])
        bad=copy.deepcopy(rows);bad[2]['samples'][1]['request_sha256']='f'*64
        with self.assertRaises(RuntimeError):profile.observer_decision(bad)
        for value in (0.,float('nan'),True):
            bad=copy.deepcopy(rows);bad[1]['samples'][0]['decode_ms']=value
            with self.assertRaises(RuntimeError):profile.observer_decision(bad)

    def test_named_device_families_keep_overlap_and_unknown_kernels(self):
        def event(name,start,duration):return dict(ph='X',cat='gpu_op',name=name,ts=start,dur=duration)
        events=[event('strata::df2_gemm',1010,20),event('strata::df2_attention',1020,20),
                event('unidentified_kernel',1015,20),event('zeCommandListAppendMemoryCopy(D2M)[8]',1025,2)]
        result=profile.families(events,1000000,1040000)
        self.assertEqual(result['named_dflash2_kernel']['count'],2)
        self.assertEqual(result['named_dflash2_kernel']['clipped_sum_us'],40)
        self.assertEqual(result['named_dflash2_kernel']['interval_union_us'],30)
        self.assertEqual(result['other_kernel']['count'],1)
        self.assertEqual(result['copy']['count'],1)
        self.assertLess(result['named_dflash2_kernel']['interval_union_us'],
                        result['named_dflash2_kernel']['clipped_sum_us'])


if __name__=='__main__':unittest.main()
