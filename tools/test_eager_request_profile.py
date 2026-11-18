"""Check the real benchmark parser, observation pairing and RAW gate order."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'sycl'))
from tools import strata_bench
import eager_request_profile as profile


class EagerRequestProfileTests(unittest.TestCase):
    def test_clean_and_trace_share_the_actual_benchmark_command(self):
        plan = dict(root='/tmp/unused', port=18765, read=profile.READ)
        clean = profile.command(dict(plan, arm='clean'))
        self.assertEqual(clean, profile.command(dict(plan, arm='trace')))
        args = strata_bench.parse_arguments(clean[2:])
        self.assertTrue(args.think and args.prefill_only)
        self.assertEqual(args.prefill_exact_tokens, 65537)
        self.assertFalse(args.prefill_short_diagnostic)
        self.assertEqual(args.runs, 3)

    def test_warmed_overhead_failure_cannot_be_hidden_by_first_use(self):
        processes = []
        for i, value in enumerate((120., 100.)):
            traced = profile.ARMS[i] == 'trace'
            samples = [dict(prompt_ms=v * 1000, ttft_s=v, total_s=v + 2)
                for v in (110., value, value)]
            row = dict(index=i, passed=True, complete=True, ownership_closed=True,
                clean_speed=not traced, native_identity=dict(pid=i + 1, start_ticks=100), samples=samples)
            if traced:
                row['resources'] = {}
            processes.append(row)
        result = profile.decision(processes)
        self.assertTrue(result['observer_cost']['first_long']['prompt_ms']['within_10_percent'])
        self.assertEqual(result['observer_cost']['warmed']['prompt_ms']['trace_over_clean'], 1.2)
        self.assertFalse(result['fresh_pair_within_observer_budget'])
        self.assertFalse(result['full_original_observer_cohort_qualified'])

    def test_request_gates_require_begin_and_end_for_each_request(self):
        records = {}
        for i in range(3):
            profile.observe_gate(records, i, 'begin', i * 10 + 1)
            with self.assertRaises((ValueError, RuntimeError)):
                profile.observe_gate(records, i + 1, 'begin', i * 10 + 2)
            profile.observe_gate(records, i, 'end', i * 10 + 3)
        self.assertEqual(records['external_request_raw_windows'],
            [{'begin': 1, 'end': 3}, {'begin': 11, 'end': 13}, {'begin': 21, 'end': 23}])


if __name__ == '__main__':
    unittest.main()
