"""Offline width/count/eligibility and independent phase-selection regressions."""
import copy
import unittest
import mtp_width_bench as m


def fixture(width):
    offered=0 if width==1 else (width-1)*4
    accepted=0 if width==1 else 2
    emitted=5 if width==1 else 7
    row=dict(prompt_tokens=33000,reused_tokens=0,read_tokens=33000,generated_tokens=emitted,
        drafts_accepted=accepted,drafts_offered=offered)
    text=(f'strata serve: prompt 33000 tokens = 0 reused + 33000 read in 40000 ms (825 tok/s), '
        f'{emitted} generated in 20000 ms (1 tok/s), drafts accepted {accepted} of {offered}, 0 checkpoints\n')
    if width==1:text+='strata resident MTP control: mtp_max_t=1 cumulative_draft_rounds=0\n'
    return text,[row]


class WidthEvidence(unittest.TestCase):
    def test_three_fixed_widths_and_actual_output_conservation(self):
        for width in (1,2,4):
            text,rows=fixture(width)
            result,=m.native_work(text,rows,width)
            self.assertEqual(result['target_windows_inferred_from_offered'],5)
            self.assertEqual(result['target_rows_inferred'],1+4*width)

    def test_reject_bad_aggregate_work_observers_and_unbound_requests(self):
        for width in (1,2,4):
            text,rows=fixture(width)
            for bad in (text+text,text+'strata trace: window 32999 1\n',text.replace('of 0,','of 1,')):
                if bad==text:continue
                with self.subTest(width=width),self.assertRaises(RuntimeError):m.native_work(bad,rows,width)
            bad=copy.deepcopy(rows);bad[0]['generated_tokens']+=4
            with self.assertRaises(RuntimeError):m.native_work(text,bad,width)

    def test_no_draft_preserves_speed_and_agreement_labels(self):
        sample=dict(eligible=False,speed_eligible=True,ok=True,stream_complete=True,finish_reason='stop',
            reused_tokens=0,read_tokens=33000,prompt_tokens=33000,generated_tokens=1200,
            usage_generated_tokens=1200,generated_tokens_source='usage',output_chars=600,reasoning_chars=100,
            completion_includes_thinking=True,decode_tps_source='engine',drafts_offered=0,drafts_accepted=0,
            acceptance_rate=None,decode_ms=120000.,decode_tps=10.,prompt_ms=40000.,total_s=160.)
        self.assertTrue(m.sample_eligible(sample,1));self.assertFalse(sample['eligible'])
        self.assertIsNone(sample['acceptance_rate']);self.assertEqual(sample['drafts_offered'],0)
        for key,value in [('speed_eligible',False),('read_tokens',10),('reasoning_chars',0),
                ('acceptance_rate',0.),('drafts_offered',1),('drafts_accepted',True),('generated_tokens',100),
                ('decode_ms',float('nan'))]:
            with self.subTest(key=key):self.assertFalse(m.sample_eligible(dict(sample,**{key:value}),1))

    def test_decode_gain_does_not_excuse_prefill_decline(self):
        processes=[]
        for width in m.ORDER:
            requests=[]
            for i,_ in enumerate(m.WORKLOADS):
                requests.append(dict(generated_tokens=1200,decode_ms=120000. if width==4 else 100000.,
                    read_tokens=33000,prompt_ms=40000. if width!=2 else 41000.,acceptance_rate=.85 if i==0 else .35))
            processes.append(dict(width=width,passed=True,requests=requests))
        result=m.comparison(processes);candidate=result['candidates']['2']['regular-code']
        self.assertTrue(candidate['decode_gain_gate']);self.assertFalse(candidate['prefill_preservation_gate'])
        self.assertFalse(candidate['production_adopted']);self.assertTrue(result['reference_classification'])
        processes[7]['requests'][0]['decode_ms']=121000.
        self.assertFalse(m.comparison(processes)['candidates']['2']['regular-code']['decode_gain_gate'])
        with self.assertRaises(RuntimeError):m.comparison(processes[:-1])


if __name__=='__main__':unittest.main()
