#!/usr/bin/env python3
"""Check cohort order, workload eligibility and independent overhead decisions."""
import copy
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
import cpu_service_overhead as subject


def cohort():
    sample = dict(ok=True, eligible=True, speed_eligible=True, stream_complete=True,
        finish_reason='stop', prompt_tokens=40000, read_tokens=40000, reused_tokens=0,
        generated_tokens=2000, usage_generated_tokens=2000, generated_tokens_source='usage',
        output_chars=2048, reasoning_chars=8192, completion_includes_thinking=True,
        drafts_offered=3000, drafts_accepted=2400, acceptance_rate=.8,
        prompt_ms=50000., decode_ms=200000., decode_tps=10., decode_tps_source='engine',
        total_s=250., request_sha256='a' * 64)
    return [dict(index=i, arm=arm, native_identity=dict(pid=1000 + i, start_ticks=100 + i),
        source_commit='pinned-source', binary_sha256='original' if arm == 'original' else 'observer',
        passed=True, ownership_closed=True, native_exit_code=0, fresh_faults=[], boot_unchanged=True,
        effective_environment_verified=True, sample=copy.deepcopy(sample),
        work=dict(cpu_entries=20000, format_nt_jobs=[dict(gu=18, down=20, jobs_NT=[12000, 4000, 0, 0])]),
        service=dict(same_request_census=True)) for i, arm in enumerate(subject.ORDER)]


class OverheadDecisionTests(unittest.TestCase):
    def test_balanced_three_arm_cohort_qualifies_overhead_only(self):
        value = subject.summarize(cohort())
        self.assertTrue(value['overhead_qualified'])
        self.assertTrue(value['observed_strong_MTP'])
        self.assertFalse(value['inference_adopted'])
        self.assertFalse(value['clean_speed_qualified'])
        self.assertEqual(value['comparisons']['on_vs_off']['decode']['ratios'], [1., 1., 1.])

    def test_wrong_order_missing_arm_or_reused_process_is_rejected(self):
        for change in ('order', 'missing', 'pid'):
            rows = cohort()
            if change == 'order': rows[0], rows[1] = rows[1], rows[0]
            elif change == 'missing': rows.pop()
            else: rows[1]['native_identity'] = rows[0]['native_identity']
            with self.subTest(change=change), self.assertRaises(RuntimeError): subject.summarize(rows)

    def test_compile_enabled_off_overhead_is_separate_from_on_overhead(self):
        rows = cohort()
        for row in rows:
            if row['arm'] != 'original': row['sample']['decode_ms'] *= 1.1
        decision = subject.summarize(rows)
        self.assertFalse(decision['comparisons']['off_vs_original']['accepted'])
        self.assertTrue(decision['comparisons']['on_vs_off']['accepted'])
        self.assertFalse(decision['overhead_qualified'])

    def test_enabled_observer_and_prefill_have_independent_limits(self):
        for phase, multiplier in (('decode_ms', 1.2), ('prompt_ms', 1.1)):
            rows = cohort()
            for row in rows:
                if row['arm'] == 'on': row['sample'][phase] *= multiplier
            decision = subject.summarize(rows)['comparisons']['on_vs_off']
            self.assertFalse(decision['accepted'])
            self.assertFalse(decision['decode' if phase == 'decode_ms' else 'prefill']['accepted'])

    def test_changed_mtp_or_cpu_work_makes_overhead_confounded(self):
        for change in ('acceptance', 'work', 'format'):
            rows = cohort()
            if change == 'acceptance':
                rows[2]['sample'].update(drafts_accepted=1800, acceptance_rate=.6)
            elif change == 'work': rows[2]['work']['cpu_entries'] *= 2
            else: rows[2]['work']['format_nt_jobs'][0]['jobs_NT'] = [4000, 8000, 0, 0]
            decision = subject.summarize(rows)['comparisons']['on_vs_off']
            self.assertTrue(decision['decode']['accepted'])
            self.assertFalse(decision['work_comparable'])
            self.assertFalse(decision['accepted'])

    def test_ineligible_unbound_failed_or_unconserved_data_is_rejected(self):
        for change in ('short', 'failure', 'source', 'provider', 'request', 'service', 'binary', 'clock'):
            rows = cohort()
            if change == 'short': rows[2]['sample']['generated_tokens'] = 24
            elif change == 'failure': rows[2]['passed'] = False
            elif change == 'source': rows[2]['source_commit'] = 'changed'
            elif change == 'provider': rows[2]['effective_environment_verified'] = False
            elif change == 'request': rows[2]['sample']['request_sha256'] = 'b' * 64
            elif change == 'service': rows[2]['service']['same_request_census'] = False
            elif change == 'binary': rows[2]['binary_sha256'] = 'changed'
            else: rows[2]['sample']['prompt_ms'] = 0.
            with self.subTest(change=change), self.assertRaises(RuntimeError): subject.summarize(rows)


if __name__ == '__main__':
    unittest.main()
