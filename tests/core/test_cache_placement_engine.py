#!/usr/bin/env python3
"""Refuse stale admission, mis-sized loans and unsupported placement claims."""
import copy
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import cache_placement_engine as engine


def loan(dense=False):
    pairs, sizes = [(0,e) for e in range(129)], [257]*48
    lines = [
        'strata loan audit: phase=layout chunk=256 first=128 slots=129 bytes=512 need=500 lent=1 mask_match=1',
        'strata loan audit: phase=lend table_entries=24576 table_match=1 free_bytes=100 total_bytes=200',
        'strata cache loan: phase=masked generation=5 chunk=256 first=128 lent=1 decode_ready=0']
    if dense:
        lines += ['strata prefill dense: phase=retired residency=Full activations=0',
                  'strata prefill dense: phase=bank residency=Bank activations=0']
    lines += ['strata loan audit: phase=payload rows=129 bytes=66048 lent=1 same_slots=1 payload_match=1']
    if dense: lines += ['strata prefill dense: phase=restored residency=Full activations=48']
    lines += ['strata loan audit: phase=refill table_entries=24576 table_match=1 free_bytes=100 total_bytes=200',
              'strata cache loan: phase=ready generation=5 chunk=256 first=128 lent=1 decode_ready=1']
    return '\n'.join(lines)+'\n', pairs, sizes


def process(byte_count=1000, gpu=100, milliseconds=2000):
    return dict(passed=True,complete=True,census=dict(logical_job_blob_bytes=byte_count,rows=100,
        cpu_entries=1000-gpu,vram_entries=gpu),request=dict(request_sha256='fixed',read_tokens=33214,
        decode_ms=milliseconds,generated_tokens=1024))


class PlacementEngineTests(unittest.TestCase):
    def test_actual_offsets_and_short_vs_dense_loan(self):
        for dense in (False,True):
            text,pairs,sizes = loan(dense)
            row = engine.loan_cycle(text,pairs,sizes,dense_required=dense)
            self.assertEqual((row['cache_bytes'],row['capacity'],row['lent']), (66048,512,1))
            self.assertEqual(row['restored_dense_layers'],48 if dense else None)
        with self.assertRaises(RuntimeError): engine.loan_cycle(*loan(),dense_required=True)

    def test_refuses_payload_map_offset_mask_generation_and_publication_defects(self):
        text,pairs,sizes = loan(True)
        variants = [text.replace('slots=129','slots=1256'),text.replace('bytes=512','bytes=513'),
            text.replace('bytes=66048','bytes=66047'),text.replace('need=500','need=513'),
            text.replace('first=128','first=127'),text.replace('mask_match=1','mask_match=0'),
            text.replace('table_match=1','table_match=0'),text.replace('payload_match=1','payload_match=0'),
            text.replace('same_slots=1','same_slots=0'),text.replace('phase=ready generation=5','phase=ready generation=6'),
            text.replace('activations=48','activations=47'),text.replace('free_bytes=100','free_bytes=201'),
            '\n'.join(text.splitlines()[:-1]),text+text]
        for changed in variants:
            with self.subTest(changed=changed[:100]), self.assertRaises(RuntimeError):
                engine.loan_cycle(changed,pairs,sizes,dense_required=True)

    def test_startup_requires_exact_slots_fill_and_placement(self):
        common = ('strata generate: expert cache 1241 slots, 2.40 GiB of VRAM; policy is\n'
            'strata generate: pre-filled 1241 of 1241 slots from the profile; slot 0 verified\n'
            'strata generate: 5 expert-pool workers + the host thread\n')
        global_text = common+'                 PROFILE, ranked by routing frequency, no eviction.\n'
        engine.startup(global_text,dict(slots=1241),True)
        for text,candidate in [(global_text,False),(global_text.replace('1241','1240'),True),
            (global_text.replace('slot 0 verified','unverified'),True),
            (global_text+'R4.2g PER-LAYER:\n',True),(global_text+'UR_RESULT_ERROR_UNKNOWN\n',True)]:
            with self.assertRaises(RuntimeError): engine.startup(text,dict(slots=1241),candidate)
        engine.startup(common+'R4.2g PER-LAYER:\n',dict(slots=1241),False)

    def test_candidate_config_changes_only_admission_and_cache_argument(self):
        setup = dict(config=dict(args=['--expert-profile','old','--expert-cache','1256',
            '--expert-cache-per-layer','--adapt-swaps','0','--pcie-frac','0','--prefill','8192']))
        original = copy.deepcopy(setup)
        root = Path('/tmp/placement-unit-input')
        geometry = dict(candidate=dict(expert_cache_argument=967))
        base = engine.config(setup,root,False,geometry)
        candidate = engine.config(setup,root,True,geometry)
        args = candidate['args']; args.insert(base['args'].index('--expert-cache-per-layer'),'--expert-cache-per-layer')
        args[args.index('--expert-cache')+1] = '1256'
        self.assertEqual(base,candidate); self.assertEqual(setup,original)
        setup['config']['args'][setup['config']['args'].index('--pcie-frac')+1] = '0.1'
        with self.assertRaises(RuntimeError): engine.config(setup,root,True,geometry)

    def test_screen_criteria_do_not_imply_adoption_or_clean_speed(self):
        passed = engine.decide([process(),process(940,110,1999)])
        self.assertTrue(passed['advance_to_new_clean_plan'])
        self.assertFalse(passed['adoption_qualified']); self.assertFalse(passed['clean_speed_qualified'])
        for candidate in (process(951,110,1999),process(940,100,1999),process(940,110,2001)):
            self.assertFalse(engine.decide([process(),candidate])['advance_to_new_clean_plan'])
        wrong = process(940,110,1999); wrong['request']['request_sha256'] = 'other'
        self.assertFalse(engine.decide([process(),wrong])['advance_to_new_clean_plan'])
        wrong['passed'] = False
        with self.assertRaises(RuntimeError): engine.decide([process(),wrong])


if __name__ == '__main__':
    unittest.main()
