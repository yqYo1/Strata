#!/usr/bin/env python3
"""Offline false-positive checks for the long cache-capacity evidence gate."""
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import cache_capacity as c

ROWS = 33210
SAMPLE = dict(prompt_tokens=ROWS+1,read_tokens=ROWS+1,generated_tokens=2931)


def fixture():
    result = []
    for generation,tokens,generated in ((1,100,24),(2,300,1),(3,1000,1),(4,1100,8),(5,ROWS+1,2931)):
        long = generation == 5
        chunk,first,capacity,need = (8192,165,2320512000,2319958528) if long else (256,1183,241280000,240304384)
        lent = 1280-first
        def publication(phase,ready):
            return f'strata cache loan: phase={phase} generation={generation} chunk={chunk} first={first} lent={lent} decode_ready={ready}'
        result += [f'strata loan audit: phase=layout chunk={chunk} first={first} slots=1280 bytes={capacity} need={need} lent={lent} mask_match=1',
            'strata loan audit: phase=lend table_entries=24576 table_match=1 free_bytes=400000000 total_bytes=10666115072',publication('masked',0)]
        if long:
            def dense(phase,state,activations):
                return f'strata prefill dense: phase={phase} residency={state} activations={activations} full_bytes=2116951040 bank_bytes=50918400 snapshot_bytes=2116951040 overlay_rows=0 session_recapture=0 verifier=1 free_bytes=400000000 total_bytes=10666115072'
            result += [dense('retired','Full',0),dense('bank','Bank',0),
                f'strata prefill: layer-major candidate, tokens={ROWS} chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_total={ROWS*20480} residual_gpu_rows={ROWS} residual_gpu={ROWS*20480} residual_host=0 residual_staging=0 layer_cache=1022361600; extra allocations, GPU headroom=536870912, RAM headroom=8589934592',
                f'strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows={ROWS} mixed_fp16_bytes=0',
                'strata prefill cache binding: generation=5 parent_match=1 execution_distinct=1']
            for layer in range(48):
                result += [f'strata MoE mixed owners: phase=entry layer={layer} first=0 rows={ROWS} primary=0x1000 primary_bytes={ROWS*5120} region=0x2000 region_bytes=1180698624 tail=(nil) tail_offset=1180698624 tail_bytes=0 residual=0x3000 residual_bytes={ROWS*20480} bo=0x4000 queue=0x5000 direct_owners=0 previous_attention=checked_drain',
                    f'strata MoE cache loan: generation=5 layout_generation=4 layer={layer} first=0 rows={ROWS} used=2311569920 capacity=2320512000',
                    f'strata MoE mixed owners: phase=retired layer={layer} first=0 rows={ROWS} route_status=0 shared_HC=checked_drain callbacks=after_return']
            result.append(dense('restored','Full',48))
        result += [f'strata loan audit: phase=payload rows=1280 bytes=2639104000 lent={lent} same_slots=1 payload_match=1',
            'strata loan audit: phase=refill table_entries=24576 table_match=1 free_bytes=138100736 total_bytes=10666115072',publication('ready',1),
            f'strata serve: prompt {tokens} tokens = 0 reused + {tokens} read in 100 ms (1.0 tok/s), {generated} generated in 100 ms']
    return '\n'.join(result)+'\n'


class CapacityEvidence(unittest.TestCase):
    def test_completed_official_sequence(self):
        result = c.long_evidence(fixture(),SAMPLE)
        self.assertEqual(result['prefill_rows'],ROWS)
        self.assertEqual(result['exact_payload_and_map_cycles'],5)
        self.assertEqual(result['checked_retirements'],48)
        self.assertEqual(result['min_observed_free_bytes'],138100736)

    def test_reject_partial_stale_and_changed_measured_work(self):
        text = fixture()
        edits = [('generation=5 layout_generation=4 layer=7','generation=4 layout_generation=4 layer=7'),
                 ('phase=ready generation=5','phase=ready generation=4'),
                 ('effective_mode=10','effective_mode=2'),
                 ('residual_host=0','residual_host=20480'),
                 ('payload_match=1','payload_match=0'),
                 ('table_match=1','table_match=0'),
                 ('execution_distinct=1','execution_distinct=0'),
                 ('tail_bytes=0','tail_bytes=5120'),
                 ('route_status=0','route_status=4'),
                 ('callbacks=after_return','callbacks=before_return'),
                 ('previous_attention=checked_drain','previous_attention=unchecked'),
                 ('direct_owners=0','direct_owners=1'),
                 ('residency=Bank activations=0','residency=Full activations=0'),
                 ('residency=Full activations=48','residency=Full activations=47')]
        for before,after in edits:
            with self.subTest(edit=before),self.assertRaises(RuntimeError):
                c.long_evidence(text.replace(before,after,1),SAMPLE)
        for key in SAMPLE:
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                c.long_evidence(text,dict(SAMPLE,**{key:SAMPLE[key]+1}))
        with self.assertRaises(RuntimeError): c.long_evidence(text.rsplit('\n',2)[0]+'\n',SAMPLE)

    def test_reject_retirement_moved_to_another_request(self):
        text = fixture()
        line = next(v for v in text.splitlines() if 'phase=retired layer=47' in v)
        changed = line+'\n'+text.replace(line+'\n','')
        with self.assertRaises(RuntimeError): c.long_evidence(changed,SAMPLE)

    def test_reject_refill_ready_before_payload_completion(self):
        text = fixture()
        line = next(v for v in text.splitlines() if 'phase=ready generation=5' in v)
        changed = text.replace(line+'\n','').replace('strata prefill cache binding:',line+'\nstrata prefill cache binding:')
        with self.assertRaises(RuntimeError): c.long_evidence(changed,SAMPLE)


if __name__ == '__main__': unittest.main()
