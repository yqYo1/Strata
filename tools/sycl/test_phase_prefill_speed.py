#!/usr/bin/env python3
"""CPU refusal controls for independent prefilling evidence; no GPU."""
import copy
import math
import unittest
import phase_prefill_speed as m


def rows():
    result = []
    for i, arm in enumerate(m.ORDER):
        result.append(dict(arm=arm, passed=True, complete=True, ownership_closed=True, fresh_faults=[],
            native_identity=dict(pid=100+i,start_ticks=1000+i), requests=[dict(run=j,ok=True,
                eligible=True,stream_complete=True,thinking_requested=True,reasoning_observed=True,
                actual_prompt_tokens=117489,actual_read_tokens=117489,actual_reused_tokens=0,
                generated_tokens=24,prompt_tps_source='engine',prompt_ms=117489000/(1005 if arm else 1000),
                request_sha256=str(j)*64) for j in range(4)]))
    return result


class EvidenceControls(unittest.TestCase):
    def test_prefill_gain_never_qualifies_decode_or_adoption(self):
        r=m.summarize(rows())
        self.assertTrue(r['candidate_faster_screen'])
        self.assertFalse(r['decode_preservation_qualified'])
        self.assertFalse(r['adoption_qualified'])

    def test_one_declining_pair_fails_even_when_average_improves(self):
        r=rows()
        for s in r[1]['requests']:s['prompt_ms']=117489000/999.9
        d=m.summarize(r)
        self.assertGreater(d['ratio_geomean'],1)
        self.assertFalse(d['prefill_no_decline_screen'])
        self.assertFalse(d['candidate_faster_screen'])

    def test_first_use_cannot_raise_subsequent_median(self):
        r=rows()
        for p in r:
            p['requests'][0]['prompt_ms']=1 if p['arm'] else 1e9
        for actual,expected in zip(m.summarize(r)['subsequent_median_tok_s'],[1000,1005,1005,1000,1000,1005]):
            self.assertAlmostEqual(actual,expected)

    def test_short_reused_incomplete_unclocked_or_changed_requests_reject(self):
        for changes in ({'actual_read_tokens':32768},{'actual_reused_tokens':1},
            {'stream_complete':False},{'reasoning_observed':False},{'prompt_ms':math.nan},
            {'prompt_ms':0},{'request_sha256':'f'*64}):
            r=rows();r[1]['requests'][2].update(changes)
            with self.subTest(changes=changes),self.assertRaises(RuntimeError):m.summarize(r)

    def test_reused_process_or_failed_owner_reject(self):
        for field in ('identity','owner','fault'):
            r=copy.deepcopy(rows())
            if field=='identity':r[1]['native_identity']=r[0]['native_identity']
            elif field=='owner':r[1]['ownership_closed']=False
            else:r[1]['fresh_faults']=['new fault']
            with self.subTest(field=field),self.assertRaises(RuntimeError):m.summarize(r)


if __name__=='__main__':unittest.main()
