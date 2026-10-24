"""Host-only eligibility and actual-route corruption controls for MTP pilot."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import strata_bench as client
import long_generation_bench as lg
from test_strata_bench import response
from test_mtp_resident_serve import sample_trace
import mtp_net_bench as pilot


def sample(cap):
    raw=response(offered=0 if cap else 1600,accepted=0 if cap else 1200)
    row=client.digest(raw);row.update(lg.sample_eligibility(raw,row),output_chars=len(raw['text']),reasoning_chars=len(raw['reasoning_text']))
    return row


class Pilot(unittest.TestCase):
    def test_real_client_zero_draft_eligibility_stays_distinct(self):
        for cap in (0,1):
            row=sample(cap);self.assertTrue(pilot.sample_eligible(row,cap))
            self.assertFalse(pilot.sample_eligible(row,1-cap))
            for key,value in [('usage_generated_tokens',1023),('stream_complete',False),('reasoning_chars',0),
                              ('output_chars',511),('read_tokens',32767),('decode_ms',float('nan'))]:
                bad=copy.deepcopy(row);bad[key]=value
                with self.subTest(cap=cap,key=key):self.assertFalse(pilot.sample_eligible(bad,cap))

    def test_timing_floor_is_independent_of_response_validity(self):
        row=sample(1);self.assertFalse(pilot.timing_eligible(row))
        row.update(prompt_ms=10000,decode_ms=10000);self.assertTrue(pilot.timing_eligible(row))
        for value in [9999.9,False,float('inf'),float('nan')]:
            row['decode_ms']=value;self.assertFalse(pilot.timing_eligible(row))

    def test_original_short_token_fallback_and_both_verifier_paths(self):
        for cap in (0,1):
            text,rows=sample_trace(cap);out=pilot.request_routes(text,rows,cap)
            self.assertEqual([p['draft_rounds_delta']for p in out],[0,0]if cap else[31,31])
            with self.assertRaises(RuntimeError):
                pilot.request_routes(text.replace('path=token mode=2 cells=32','path=batched mode=2 cells=32'),rows,cap)
            bad=copy.deepcopy(rows);bad[0]['drafts_accepted']=4
            with self.assertRaises(RuntimeError):pilot.request_routes(text,bad,cap)


if __name__=='__main__':unittest.main()
