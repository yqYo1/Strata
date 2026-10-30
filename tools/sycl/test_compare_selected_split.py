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
            '[native16-down] fusion=large min_rows=2334',
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

    def test_selected_requires_its_source_accurate_startup_marker(self):
        marker = '[native16-down] fusion=large min_rows=2334'
        for log in (self.log.replace(marker, ''), self.log.replace(marker, marker+' split_kernel=original')):
            with self.assertRaises(RuntimeError):
                comparison.selected_routes(log, 1)

    def continuation_log(self, arm='split', n=98303):
        tail=(n-65536)*2560*4
        rows=[f'strata prefill: layer-major candidate, tokens={n} chunk=8192 residual_dtype=bf16 residual_gpu_rows={n} residual_host=0',
            f'strata MoE expert bank: effective_slots={128 if arm=="split" else 384} staged=1',
            f'strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows={n if arm=="split" else 81920}',
            f'strata MoE accumulator window: maximum_rows={n if arm=="split" else 81920} windows={48 if arm=="split" else 96} dequants=1000 jobs=37000 source_H2D_bytes=1',
            'strata MoE expert bank complete: final_drain=checked',
            'strata MoE Down fusion: fused_jobs=7000 fused_rows=17000000 separate_jobs=30000 min_rows=2334']
        if arm=='split':
            rows += [f'strata MoE split admission: active=1 rows={n} primary_rows=65536 tail_payload_bytes={tail} extra_allocation_bytes={(tail+255)//256*256+256} prefill_slots=128 max_single_allocation_bytes=4294967296 decode_changed=0',
                     f'strata MoE split complete: rows={n} primary_rows=65536 tail_bytes={tail} guard=checked final_drain=checked decode_changed=0']
        return '\n'.join(rows)

    def test_continuation_uses_dynamic_extents_and_per_arm_placement(self):
        for arm in ('selected','split'):
            for n in (90113,98303,114431):
                route=comparison.continuation_route(self.continuation_log(arm,n),arm,n+1)
                self.assertEqual(route['engine_rows'],n)
                self.assertEqual(route['maximum_window_rows'],n if arm=='split' else 81920)
                if arm=='split':self.assertEqual(route['split']['tail_rows'],n-65536)
        with self.assertRaises(RuntimeError):comparison.continuation_route(self.continuation_log(),'split',90113)
        with self.assertRaises(RuntimeError):comparison.continuation_route(self.continuation_log(),'split',114433)

    def test_continuation_refuses_bad_tail_guard_decode_and_fallback(self):
        log=self.continuation_log()
        for old,new in [('active=1','active=0'),('effective_slots=128','effective_slots=384'),
                        ('maximum_rows=98303','maximum_rows=117488'),('primary_rows=65536','primary_rows=65535'),
                        ('tail_bytes=335534080','tail_bytes=335534081'),('extra_allocation_bytes=335534336','extra_allocation_bytes=335534337'),
                        ('guard=checked','guard=unchecked'),('decode_changed=0','decode_changed=1'),
                        ('final_drain=checked','final_drain=unchecked'),('residual_host=0','residual_host=1'),
                        ('separate_jobs=30000','separate_jobs=29999')]:
            self.assertIn(old,log)
            with self.subTest(field=old),self.assertRaises(RuntimeError):
                comparison.continuation_route(log.replace(old,new),'split',98304)
        with self.assertRaises(RuntimeError):comparison.continuation_route(log+'\n'+log,'split',98304)

    def test_continuation_does_not_accept_client_counts_as_native_extent(self):
        with self.assertRaises(RuntimeError):comparison.continuation_route(self.continuation_log(),'split',98305)
        log=self.continuation_log('selected')
        with self.assertRaises(RuntimeError):comparison.continuation_route(log+'\nstrata MoE split admission: active=1','selected',98304)
        with self.assertRaises(RuntimeError):comparison.continuation_route(log.replace('effective_slots=384','effective_slots=128'),'selected',98304)


if __name__ == '__main__':
    transcript = io.StringIO()
    result = unittest.TextTestRunner(stream=transcript).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ComparisonGates))
    print(json.dumps(dict(cases=result.testsRun, failures=len(result.failures),
                          errors=len(result.errors), passed=result.wasSuccessful())))
    if not result.wasSuccessful():
        sys.stderr.write(transcript.getvalue())
        sys.exit(1)
