#!/usr/bin/env python3
"""Negative evidence cases for the six-process CPU capacity discriminator."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import cpu_capacity as c
from test_generation_engine import sample


def cohort(ratios=(1.4, 1.4, 1.4)):
    rows = [dict(cohort=i, workers=w, passed=True, ownership_closed=True, fresh_faults=[],
        runtime_pool_confirmed=True, observer_records_absent=True,
        pool_execution_evidence=dict(observed_decode_misses=True),
        machine_identity=dict(boot_id='boot', device=dict(kernel='kernel', driver='xe'), guarded_runtime_sha256='runtime'),
        model_assets_fingerprint='model', source_commit='source', binary_sha256='binary', benchmark_sha256='benchmark',
        native_identity=dict(pid=100+i, start_ticks=1000+i), warmups=dict(successful_warmups=4),
        request=dict(sample(), request_sha256='a'*64)) for i, w in enumerate(c.ORDER)]
    for (_, reduced), ratio in zip(c.comparison_plan()['pairs'], ratios):
        rows[reduced]['request']['decode_ms'] *= ratio
        rows[reduced]['request']['decode_tps'] /= ratio
    return rows


class CapacityEvidenceTests(unittest.TestCase):
    def test_consistent_effect_and_opposite_direction(self):
        self.assertEqual(c.summarize_processes(cohort())['decision'], 'reduced_capacity_slower')
        self.assertEqual(c.summarize_processes(cohort((.7,.7,.7)))['decision'], 'reduced_capacity_faster')

    def test_small_or_inconsistent_effect_remains_inconclusive(self):
        for ratios in ((1.05,1.04,1.06), (1.5,.9,1.5), (1.,1.,1.)):
            self.assertTrue(c.summarize_processes(cohort(ratios))['decision'].startswith('inconclusive'))

    def test_no_partial_reordered_or_reused_processes(self):
        original = cohort()
        for variant in ('partial','reordered','identity'):
            rows = copy.deepcopy(original)
            if variant == 'partial': rows.pop()
            if variant == 'reordered': rows[0], rows[1] = rows[1], rows[0]
            if variant == 'identity': rows[5]['native_identity'] = rows[0]['native_identity']
            with self.subTest(variant=variant), self.assertRaises(RuntimeError): c.summarize_processes(rows)

    def test_failed_observed_or_short_samples_do_not_qualify(self):
        for key, value in (('passed',False), ('ownership_closed',False), ('fresh_faults',['fault']),
                           ('observer_records_absent',False), ('runtime_pool_confirmed',False)):
            rows = cohort(); rows[1][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError): c.summarize_processes(rows)
        rows = cohort(); rows[1]['request']['read_tokens'] = 100
        with self.assertRaises(RuntimeError): c.summarize_processes(rows)

    def test_changed_actual_work_is_retained_but_inconclusive(self):
        for key, value in (('drafts_offered',1100), ('generated_tokens',2050), ('request_sha256','b'*64)):
            rows = cohort(); rows[1]['request'][key] = value
            if key == 'drafts_offered': rows[1]['request']['acceptance_rate'] = 600/1100
            if key == 'generated_tokens': rows[1]['request']['usage_generated_tokens'] = value
            result = c.summarize_processes(rows)
            self.assertFalse(result['performance_eligible']); self.assertEqual(result['decision'],'inconclusive_work_mismatch')

    def test_different_generated_text_is_not_a_correctness_failure(self):
        rows = cohort()
        for i, row in enumerate(rows): row['request']['output_sha256'] = str(i)*64
        self.assertTrue(c.summarize_processes(rows)['performance_eligible'])

    def test_actual_runtime_marker_must_include_the_host(self):
        c.verify_pool('strata generate: 1 expert-pool workers + the host thread\n', 1)
        for text in ('strata generate: 5 expert-pool workers + the host thread\n',
                     'strata generate: 1 expert-pool workers\n',
                     'strata generate: 1 expert-pool workers + the host thread (UNUSED: --no-pool)\n'):
            with self.assertRaises(RuntimeError): c.verify_pool(text, 1)

    def test_cross_process_model_platform_and_source_must_match(self):
        for key in ('model_assets_fingerprint','source_commit','binary_sha256','benchmark_sha256','machine_identity'):
            rows = cohort(); rows[1][key] = 'changed'
            with self.subTest(key=key), self.assertRaises(RuntimeError): c.summarize_processes(rows)
        rows = cohort(); rows[1]['pool_execution_evidence']['observed_decode_misses'] = False
        with self.assertRaises(RuntimeError): c.summarize_processes(rows)

    def test_asset_guard_rejects_writes_with_restored_mtime_and_wrong_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'weights'; p.write_bytes(b'old')
            with patch.object(c, 'model_paths', return_value=[p]):
                pins = c.model_pins({}, full=True)
                self.assertEqual(c.model_pins({}, pins), pins)
                wrong = copy.deepcopy(pins); wrong[str(p)]['sha256'] = '0'*64
                with self.assertRaises(RuntimeError): c.model_pins({}, wrong, full=True)
                before = p.stat(); p.write_bytes(b'new'); os.utime(p, ns=(before.st_atime_ns,before.st_mtime_ns))
                with self.assertRaises(RuntimeError): c.model_pins({}, pins)

    def test_pool_execution_evidence_is_bound_to_long_request_and_cpu_route(self):
        row = dict(generated_tokens=2048); config = dict(args=['--pcie-frac','0'])
        line = '[strata] done: 2048 tokens in 200 s (10.2 tok/s) (stop, cancel=False), expert cache 2.0% hit\n'
        self.assertTrue(c.request_pool_evidence(line,row,config)['observed_decode_misses'])
        for log in ('', line+line, line.replace('2048','24'), line.replace('2.0%','100.0%')):
            with self.assertRaises(RuntimeError): c.request_pool_evidence(log,row,config)
        with self.assertRaises(RuntimeError): c.request_pool_evidence(line,row,dict(args=['--pcie-frac','.2']))


if __name__ == '__main__': unittest.main()
