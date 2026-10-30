#!/usr/bin/env python3
"""Host rejection controls for the selected/split comparison gates."""
import copy
import io
import json
import sys
import unittest

import compare_selected_split as comparison


class ComparisonGates(unittest.TestCase):
    def setUp(self):
        self.processes = [dict(arm=arm, passed=True, complete=True, ownership_closed=True,
            fresh_faults=[], native_identity=dict(pid=100+i, start_ticks=200+i),
            subsequent_median_ms=120000 if arm == 'selected' else 117000)
            for i, arm in enumerate(comparison.ORDER)]
        self.log = '\n'.join([
            'strata prefill: layer-major candidate, tokens=117488 chunk=8192 residual_dtype=bf16 '
                'residual_gpu_rows=117488 residual_host=0',
            'strata MoE expert bank: effective_slots=384 staged=1',
            'strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows=81920',
            'strata MoE accumulator window: maximum_rows=81920 windows=96 dequants=33000 jobs=37000 source_H2D_bytes=1',
            'strata MoE expert bank complete: final_drain=checked'])

    def test_speed_screen_and_target_are_separate_from_adoption(self):
        result = comparison.decision(self.processes)
        self.assertTrue(result['performance_screen_passed'])
        self.assertTrue(result['all_three_split_warm_medians_ge1000'])
        self.assertFalse(result['inference_adopted'])
        self.assertFalse(result['sustained_decode_qualified'])

    def test_one_slower_pair_refuses_even_with_good_geometric_mean(self):
        self.processes[-1]['subsequent_median_ms'] = 120001
        result = comparison.decision(self.processes)
        self.assertLess(result['geometric_mean_time_ratio'], .995)
        self.assertFalse(result['performance_screen_passed'])

    def test_tiny_gain_does_not_advance(self):
        for p in self.processes:
            if p['arm'] == 'split':
                p['subsequent_median_ms'] = 119900
        self.assertFalse(comparison.decision(self.processes)['performance_screen_passed'])

    def test_one_subtarget_process_fails_target(self):
        self.processes[-1]['subsequent_median_ms'] = 118000
        self.assertFalse(comparison.decision(self.processes)['all_three_split_warm_medians_ge1000'])

    def test_incomplete_faulted_reordered_and_reused_processes_are_refused(self):
        for mutation in ('incomplete', 'fault', 'order', 'identity', 'nan'):
            rows = copy.deepcopy(self.processes)
            if mutation == 'incomplete':
                rows.pop()
            elif mutation == 'fault':
                rows[0]['fresh_faults'] = ['test fault']
            elif mutation == 'order':
                rows[0], rows[1] = rows[1], rows[0]
            elif mutation == 'identity':
                rows[-1]['native_identity'] = rows[0]['native_identity']
            else:
                rows[-1]['subsequent_median_ms'] = float('nan')
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                comparison.decision(rows)

    def test_selected_route_is_independent_of_split_route(self):
        route = comparison.selected_routes(self.log, 1)[0]
        self.assertEqual((route['slots'], route['maximum_window_rows'], route['work']['windows']),
                         (384, 81920, 96))

    def test_selected_wrong_extent_placement_and_drain_are_refused(self):
        for old, new in [('residual_gpu_rows=117488', 'residual_gpu_rows=65536'),
                         ('effective_slots=384', 'effective_slots=128'),
                         ('windows=96', 'windows=48'),
                         ('final_drain=checked', 'final_drain=unchecked')]:
            with self.subTest(field=old), self.assertRaises(RuntimeError):
                comparison.selected_routes(self.log.replace(old, new), 1)

    def test_selected_split_marker_or_duplicate_route_is_refused(self):
        for log in (self.log+'\nstrata MoE split admission: active=1', self.log+'\n'+self.log):
            with self.assertRaises(RuntimeError):
                comparison.selected_routes(log, 1)


if __name__ == '__main__':
    transcript = io.StringIO()
    result = unittest.TextTestRunner(stream=transcript).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ComparisonGates))
    print(json.dumps(dict(cases=result.testsRun, failures=len(result.failures),
                          errors=len(result.errors), passed=result.wasSuccessful())))
    if not result.wasSuccessful():
        sys.stderr.write(transcript.getvalue())
        sys.exit(1)
