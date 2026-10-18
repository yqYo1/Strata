#!/usr/bin/env python3
"""Offline tests against pseudoreplication and altered source-view decisions."""
import statistics
import sys
import unittest

sys.dont_write_bytecode = True
from bench_engine import source_view_pair_statistics


def arms(base, candidate):
    result = []
    for cohort in range(3):
        for name, values in (('A', base[cohort]), ('B', candidate[cohort])):
            result.append(dict(label=name + str(cohort), first_use_tps=1000 if name == 'A' else 500,
                subsequent_tps=values, subsequent_median=statistics.median(values),
                requested_expert_H2D_bytes=[10, 10, 10, 10], output_hashes=['output'], reasoning_hashes=['thought']))
    return result


class ProcessStatisticsTests(unittest.TestCase):
    def test_large_request_ratio_does_not_inflate_independent_process_gain(self):
        # One request doubles while the process median improves only0.4%.
        # Treating nine request ratios as independent would incorrectly pass.
        data = arms([[1, 100, 100]] * 3, [[2, 100.4, 100.4]] * 3)
        result = source_view_pair_statistics(data)
        self.assertAlmostEqual(result['paired_median_speed_geomean'], 1.004)
        self.assertFalse(result['performance_selection_gate_passed'])

    def test_two_large_gains_do_not_hide_a_regressing_process(self):
        result = source_view_pair_statistics(arms([[1000]*3]*3,
            [[1200]*3, [1200]*3, [999]*3]))
        self.assertGreater(result['paired_median_speed_geomean'], 1.005)
        self.assertFalse(result['performance_selection_gate_passed'])

    def test_cold_regression_is_retained_without_changing_frozen_warm_gate(self):
        result = source_view_pair_statistics(arms([[950]*3]*3, [[996]*3]*3))
        self.assertTrue(result['performance_selection_gate_passed'])
        self.assertEqual([p['first_use_ratio'] for p in result['pairs']], [0.5]*3)
        self.assertFalse(result['prefill1000_target_passed'])

    def test_target_requires_each_candidate_process_median(self):
        result = source_view_pair_statistics(arms([[950]*3]*3,
            [[1005]*3, [1005]*3, [999.9]*3]))
        self.assertGreater(statistics.mean(result['candidate_process_medians']), 1000)
        self.assertTrue(result['performance_selection_gate_passed'])
        self.assertFalse(result['prefill1000_target_passed'])

    def test_output_and_requested_payload_are_descriptive(self):
        data = arms([[950]*3]*3, [[1001]*3]*3)
        data[1]['requested_expert_H2D_bytes'][0] += 1
        data[1]['output_hashes'] = ['different']
        result = source_view_pair_statistics(data)
        self.assertTrue(result['performance_selection_gate_passed'])
        self.assertFalse(result['requested_expert_payload_matches_in_all_pairs'])
        self.assertFalse(result['pairs'][0]['output_hashes_match'])

    def test_duplicate_arm_is_not_six_processes(self):
        data = arms([[950]*3]*3, [[996]*3]*3)
        data[-1]['label'] = 'B0'
        with self.assertRaisesRegex(RuntimeError, 'six unique'):
            source_view_pair_statistics(data)


if __name__ == '__main__':
    unittest.main()
