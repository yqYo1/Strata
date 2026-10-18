#!/usr/bin/env python3
"""Offline regressions for failed observation and compact timing preservation."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import measure_down_scatter as down


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
