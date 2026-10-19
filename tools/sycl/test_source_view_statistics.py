#!/usr/bin/env python3
"""Offline tests against pseudoreplication and altered source-view decisions."""
import statistics
import sys
import unittest

sys.dont_write_bytecode = True
from bench_engine import (FUSION_FIELDS, down_pair_statistics, parse_counter_lines, qualified_build_source_matches,
                          source_view_pair_statistics, validate_window_work)
from build_engine import binary_origin_source


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


class DownStatisticsTests(unittest.TestCase):
    def test_down_and_source_view_keep_different_prospective_gates(self):
        data=arms([[1000]*3]*3,[[1003]*3]*3)
        self.assertTrue(down_pair_statistics(data)['performance_selection_gate_passed'])
        self.assertFalse(source_view_pair_statistics(data)['performance_selection_gate_passed'])

    def test_exact_boundary_is_not_a_strict_gain(self):
        self.assertFalse(down_pair_statistics(arms([[1000]*3]*3,[[1002]*3]*3))[
            'performance_selection_gate_passed'])

    def test_one_regressing_process_is_not_hidden_by_large_gains(self):
        result=down_pair_statistics(arms([[1000]*3]*3,[[1200]*3,[1200]*3,[999]*3]))
        self.assertGreater(result['paired_median_speed_geomean'],1.002)
        self.assertFalse(result['performance_selection_gate_passed'])

    def test_target_is_separate_and_each_candidate_process_matters(self):
        result=down_pair_statistics(arms([[950]*3]*3,[[999.9]*3,[1001]*3,[1001]*3]))
        self.assertTrue(result['performance_selection_gate_passed'])
        self.assertFalse(result['prefill1000_target_passed'])

    def test_median_cannot_override_original_samples(self):
        data=arms([[1000]*3]*3,[[1001]*3]*3);data[1]['subsequent_median']=1100
        with self.assertRaises(RuntimeError):down_pair_statistics(data)

    def test_nonfinite_or_incomplete_samples_are_rejected(self):
        for values in ([1003,float('nan'),1003],[1003,float('inf'),1003],[1003,1003]):
            data=arms([[1000]*3]*3,[values,[1003]*3,[1003]*3])
            with self.assertRaises(RuntimeError):down_pair_statistics(data)


class WorkCounterTests(unittest.TestCase):
    def setUp(self):
        self.prefix = 'strata MoE Down fusion:'
        self.valid = self.prefix+' fused_jobs=0 fused_rows=0 separate_jobs=10000 min_rows=2334'
        self.work = dict(windows='96', dequants='1000', jobs='10000',
            source_H2D_bytes=str(117488*48*10*4), GPU_resident_windows_only='1',
            operator_max='8192', maximum_rows='81920')
        self.gu = dict(fused_jobs='1000', fused_rows='5000000', separate_jobs='9000', min_rows='2334')
        self.down = dict(fused_jobs='0', fused_rows='0', separate_jobs='10000', min_rows='2334')

    def check_work(self):
        validate_window_work(self.work, self.gu, self.down, 117489, (81920,35568))

    def test_valid_disabled_and_selective_paths(self):
        counts = parse_counter_lines(self.valid, self.prefix, FUSION_FIELDS)
        self.assertEqual(counts, [self.down])
        self.check_work()
        self.down.update(fused_jobs='500', fused_rows='2000000', separate_jobs='9500')
        self.check_work()

    def test_duplicate_unknown_missing_and_malformed_fields_are_rejected(self):
        for text in (self.valid+' fused_jobs=1', self.valid+' surprise=1',
                     self.valid.replace(' fused_rows=0',''), self.valid.replace('jobs=0','jobs=-1'),
                     self.valid.replace('rows=0','rows=0.0'), self.valid.replace(': ',':'),
                     self.valid+' bare-token'):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                parse_counter_lines(text, self.prefix, FUSION_FIELDS)

    def test_gu_and_down_job_conservation_are_independent(self):
        for route in (self.gu,self.down):
            original=route['separate_jobs'];route['separate_jobs']=str(int(original)-1)
            with self.assertRaises(RuntimeError):self.check_work()
            route['separate_jobs']=original

    def test_fused_row_bounds_do_not_accept_impossible_totals(self):
        for rows in ('2333999','8192001',str(117488*48*10+1)):
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                self.gu['fused_rows']=rows;self.check_work()

    def test_window_transfer_and_geometry_mismatches_are_rejected(self):
        for key in self.work:
            original=self.work[key]
            self.work[key]='0'
            with self.subTest(key=key), self.assertRaises(RuntimeError):self.check_work()
            self.work[key]=original

    def test_impossible_total_job_capacity_is_rejected(self):
        self.work['jobs']='1000';self.gu.update(fused_jobs='0',fused_rows='0',separate_jobs='1000')
        self.down['separate_jobs']='1000'
        with self.assertRaises(RuntimeError):self.check_work()


class BinaryIdentityTests(unittest.TestCase):
    def test_matching_engine_keeps_origin_after_fixture_or_controller_changes(self):
        previous=dict(passed=True,source_commit='admission',binary_origin_source_commit='original-native',
            binary=dict(sha256='same-engine'),previous_output_comparison=dict(controller_only_admission=False))
        self.assertEqual(binary_origin_source(previous,'same-engine','new-fixture'),'original-native')

    def test_changed_or_unproved_engine_gets_current_source(self):
        previous=dict(passed=True,source_commit='old-native',binary=dict(sha256='old'))
        self.assertEqual(binary_origin_source(previous,'new','current'),'current')
        previous['passed']=False
        self.assertEqual(binary_origin_source(previous,'old','current'),'current')

    def test_original_legacy_identity_and_first_build(self):
        previous=dict(passed=True,source_commit='old-native',binary=dict(sha256='same'))
        self.assertEqual(binary_origin_source(previous,'same','current'),'old-native')
        self.assertEqual(binary_origin_source(None,'new','current'),'current')


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
