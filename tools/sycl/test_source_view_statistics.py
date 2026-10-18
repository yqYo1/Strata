#!/usr/bin/env python3
"""Offline tests against pseudoreplication and altered source-view decisions."""
import statistics
import sys
import unittest

sys.dont_write_bytecode = True
from bench_engine import qualified_build_source_matches, source_view_pair_statistics


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


class BuildQualificationTests(unittest.TestCase):
    def setUp(self):
        self.build = dict(source_commit='admission', binary_origin_source_commit='native',
            complete=True, passed=True, ownership_closed=True, active=False, fixtures={'fixture': {}},
            previous_output_comparison=dict(controller_only_admission=True,
                native_and_recipe_inputs_match=True, engine_bytes_match=True,
                fixture_bytes_match={'fixture': True}))

    def test_exact_and_proved_native_origin(self):
        self.assertTrue(qualified_build_source_matches('admission', self.build))
        self.assertTrue(qualified_build_source_matches('native', self.build))
        self.assertFalse(qualified_build_source_matches('nearby-revision', self.build))

    def test_native_or_recipe_change_cannot_reuse_old_gate(self):
        self.build['previous_output_comparison']['native_and_recipe_inputs_match'] = False
        self.assertFalse(qualified_build_source_matches('native', self.build))

    def test_binary_or_fixture_change_cannot_reuse_old_gate(self):
        self.build['previous_output_comparison']['engine_bytes_match'] = False
        self.assertFalse(qualified_build_source_matches('native', self.build))
        self.build['previous_output_comparison']['engine_bytes_match'] = True
        self.build['previous_output_comparison']['fixture_bytes_match']['fixture'] = False
        self.assertFalse(qualified_build_source_matches('native', self.build))

    def test_incomplete_fixture_proof_cannot_reuse_old_gate(self):
        self.build['previous_output_comparison']['fixture_bytes_match'] = {}
        self.assertFalse(qualified_build_source_matches('native', self.build))

    def test_active_or_failed_build_cannot_qualify_either_revision(self):
        self.build['active'] = True
        self.assertFalse(qualified_build_source_matches('admission', self.build))
        self.assertFalse(qualified_build_source_matches('native', self.build))
        self.build['active'] = False
        self.build['passed'] = False
        self.assertFalse(qualified_build_source_matches('admission', self.build))


if __name__ == '__main__':
    unittest.main()
