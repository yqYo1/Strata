#!/usr/bin/env python3
"""Offline regressions for failed observation and compact timing preservation."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import measure_down_scatter as down


class PublicComparisonTests(unittest.TestCase):
    def setUp(self):
        self.checks=[]
        for index,(arm,process) in enumerate(down.PERF_ORDER):
            events=[]
            for rows in (64,160,2333,2334,2335,5963,8192):
                for irregular in (0,1):
                    events.append(dict(kind='selected_route',enabled=arm,rows=rows,
                        source_first=irregular,source_last=rows,window_rows=78451,
                        selected=bool(arm and rows>=2334),false_route_no_write=True,expected_mask=0))
                    for round in range(-1,6):
                        events.append(dict(kind='sample',rows=rows,irregular=irregular,process=process,
                            round=round,arm=arm,order=0,public_dispatch=1,fused_route=bool(arm and rows>=2334),
                            repeat=256 if rows<256 else 16,window_rows=78451,accumulator_bytes=78451*2560*4,
                            observed_fused_dispatches=(256 if rows<256 else 16) if arm and rows>=2334 else 0,
                            queue_batch_ms=.9*(256 if rows<256 else 16),
                            inputs_immutable=True,status='PASS',wall_ms=1.,queue_interval_ms=.9))
            self.checks.append(dict(mode='perf',command=['fixture','--bench-public',str(process),str(arm)],
                startup_down_selector='large' if arm else '0',
                passed=True,ownership_closed=True,exit_code=0,native_identity=dict(pid=100+index,start_ticks=1000+index),events=events))

    def test_complete_six_process_comparison(self):
        self.assertEqual(len(down.public_timing_samples(self.checks)),504)

    def test_partial_cohort_and_changed_order_are_rejected(self):
        with self.assertRaises(RuntimeError):down.public_timing_samples(self.checks[:2])
        self.checks[1],self.checks[2]=self.checks[2],self.checks[1]
        with self.assertRaises(RuntimeError):down.public_timing_samples(self.checks)

    def test_reused_or_missing_process_identity_is_rejected(self):
        self.checks[-1]['native_identity']=self.checks[0]['native_identity']
        with self.assertRaises(RuntimeError):down.public_timing_samples(self.checks)
        self.checks[-1]['native_identity']=dict(pid=None,start_ticks=None)
        with self.assertRaises(RuntimeError):down.public_timing_samples(self.checks)

    def test_low_level_arm_and_old_small_repeat_are_rejected(self):
        sample=next(e for e in self.checks[0]['events'] if e['kind']=='sample' and e['round']==0)
        sample['public_dispatch']=0
        with self.assertRaises(RuntimeError):down.public_timing_samples(self.checks)
        sample['public_dispatch']=1;sample['repeat']=16
        with self.assertRaises(RuntimeError):down.public_timing_samples(self.checks)

    def test_duplicate_measured_shape_and_missing_warmup_are_rejected(self):
        events=self.checks[0]['events'];at=next(i for i,e in enumerate(events) if e['kind']=='sample' and e['round']==0)
        original=events[at];events[at]=copy.deepcopy(events[at+1])
        with self.assertRaises(RuntimeError):down.public_timing_samples(self.checks)
        events[at]=original
        events.remove(next(e for e in events if e['kind']=='sample' and e['round']==-1))
        with self.assertRaises(RuntimeError):down.public_timing_samples(self.checks)

    def test_wrong_startup_route_is_rejected(self):
        self.checks[0]['events'][0]['enabled']=1
        with self.assertRaises(RuntimeError):down.public_timing_samples(self.checks)

    def test_duplicate_preflight_route_placement_is_rejected(self):
        routes=[e for e in self.checks[0]['events'] if e['kind']=='selected_route']
        routes[1]['source_first']=0
        with self.assertRaises(RuntimeError):down.public_timing_samples(self.checks)

    def test_actual_timed_hook_returns_must_match_the_arm(self):
        sample=next(e for e in self.checks[1]['events'] if e['kind']=='sample' and e['rows']==2334 and e['round']==0)
        for observed in (0,sample['repeat']-1,sample['repeat']+1):
            sample['observed_fused_dispatches']=observed
            with self.assertRaises(RuntimeError):down.public_timing_samples(self.checks)

    def test_duration_floor_includes_warmup_and_checks_marker_interval(self):
        sample=next(e for e in self.checks[0]['events'] if e['kind']=='sample')
        sample['queue_batch_ms']=down.MIN_TIMING_BATCH_MS
        sample['queue_interval_ms']=sample['queue_batch_ms']/sample['repeat']
        self.assertEqual(len(down.public_timing_samples(self.checks)),504)
        sample['queue_batch_ms']=down.MIN_TIMING_BATCH_MS-1e-6
        sample['queue_interval_ms']=sample['queue_batch_ms']/sample['repeat']
        with self.assertRaises(RuntimeError):down.public_timing_samples(self.checks)
        sample['queue_batch_ms']=3.;sample['queue_interval_ms']=1.
        with self.assertRaises(RuntimeError):down.public_timing_samples(self.checks)

    def test_compaction_retains_observed_route_and_granularity_totals(self):
        record=dict(checks=self.checks)
        result=down.compact_result(record)
        for check,(arm,_) in zip(result['checks'],down.PERF_ORDER):
            observed=check['timed_dispatch_observations']
            self.assertEqual(observed['batches_including_warmup'],98)
            self.assertEqual(observed['fused_calls'],896 if arm else 0)
            self.assertEqual(observed['fused_calls']+observed['fallback_calls'],8288)
            self.assertEqual(observed['route_mismatches'],0)
            self.assertEqual(observed['ineligible_batches'],0)


class EvidenceTests(unittest.TestCase):
    def observe(self, *, journal='', driver_error=None, boot=None, **flags):
        record = dict(passed=True, ownership_closed=True)
        record.update(flags)
        boot = boot or Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        with patch.object(down.subprocess, 'check_output',
                **({'side_effect': journal} if isinstance(journal, Exception) else {'return_value': journal})), \
             patch.object(down.guards, 'driver_guard',
                **({'side_effect': driver_error} if driver_error else {'return_value': {'device': 'mock'}})):
            down.postchecks(record, boot, 'test-cursor')
        return record

    def test_clean_observation(self):
        result = self.observe()
        self.assertTrue(result['passed'])
        self.assertTrue(result['postchecks_complete'])
        self.assertEqual(result['fresh_faults'], [])

    def test_missing_journal_is_unknown_and_failed(self):
        result = self.observe(journal=OSError('journal unavailable'))
        self.assertFalse(result['passed'])
        self.assertFalse(result['postchecks_complete'])
        self.assertIsNone(result['fresh_faults'])
        self.assertEqual(result['postcheck_errors'][0]['check'], 'kernel_journal')
        self.assertTrue(result['ownership_closed'])
        self.assertTrue(result['boot_unchanged'])
        self.assertIsNotNone(result['driver_after'])

    def test_driver_failure_does_not_erase_fault(self):
        fault = 'xe 0000:05:00.0: GPU fault test'
        result = self.observe(journal=fault, driver_error=RuntimeError('wedged'))
        self.assertFalse(result['passed'])
        self.assertEqual(result['fresh_faults'], [fault])
        self.assertEqual(result['postcheck_errors'][0]['check'], 'driver_guard')

    def test_boot_change_and_failed_input_remain_failed(self):
        self.assertFalse(self.observe(boot='another-boot')['passed'])
        self.assertFalse(self.observe(ownership_closed=False)['passed'])
        record = dict(passed=False, ownership_closed=True)
        with patch.object(down.subprocess, 'check_output', return_value=''), \
             patch.object(down.guards, 'driver_guard', return_value={}):
            down.postchecks(record, Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'cursor')
        self.assertFalse(record['passed'])

    def test_missing_boot_observation_is_failed(self):
        with patch.object(down.Path, 'read_text', side_effect=OSError('boot unavailable')):
            result = self.observe(boot='known-boot')
        self.assertFalse(result['passed'])
        self.assertFalse(result['postchecks_complete'])
        self.assertFalse(result['boot_unchanged'])
        self.assertEqual(result['postcheck_errors'][0]['check'], 'boot_identity')

    def test_compaction_keeps_failed_samples_order_and_closure(self):
        events = [dict(kind='sample', status='PASS', inputs_immutable=True,
            process=2, rows=2334, irregular=1, round=n, order=a, arm=a,
            wall_ms=1.1 + a, queue_interval_ms=1.0 + a) for n in (-1, 0, 1) for a in (0, 1)]
        closure = dict(observation_complete=True, direct_child_reaped=True, session_empty=True)
        record = dict(passed=False, error='later observation failed', checks=[
            dict(mode='perf', passed=True, closure=closure, events=events),
            dict(mode='perf', passed=False, closure=closure, events=[])])
        original = copy.deepcopy(record)
        result = down.compact_result(record)
        self.assertEqual(record, original)
        self.assertFalse(result['passed'])
        self.assertEqual(result['error'], original['error'])
        self.assertNotIn('events', result['checks'][0])
        self.assertEqual(result['checks'][1]['closure'], closure)
        samples = [dict(zip(result['timing_sample_columns'], s)) for s in result['timing_samples']]
        self.assertEqual([(s['round'], s['order']) for s in samples], [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertEqual([s['queue_interval_ms'] for s in samples], [1.0, 2.0, 1.0, 2.0])


if __name__ == '__main__':
    unittest.main()
