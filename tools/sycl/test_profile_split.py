#!/usr/bin/env python3
"""Host rejection controls for owned split resource attribution."""
import copy
import unittest
from unittest.mock import patch

import profile_split as profile


class TraceGates(unittest.TestCase):
    def setUp(self):
        self.pid, self.device = 1234, 9000
        self.events = [dict(ph='M', name='process_name', pid=self.device,
            args=dict(name='DEVICE<host>[Intel(R) Arc(TM) B570 Graphics] 0:5:0:0 #0'))]
        for i, name, start, duration in ((1, 'alpha', 11000, 2000), (2, 'beta', 12000, 2000),
                (3, 'zeCommandListAppendMemoryCopy(H2D)[4096]', 14000, 100),
                (4, 'zeCommandListAppendBarrier', 15000, 0)):
            self.events += [dict(ph='s', cat='Flow_H2D_0', id=i, pid=self.pid),
                dict(ph='X', cat='gpu_op', pid=self.device, ts=start, dur=duration,
                     name=name, args=dict(id=i))]
        self.events += [dict(ph='X', cat='cpu_op', pid=self.pid, ts=9500, dur=2000, name='API1'),
                        dict(ph='X', cat='cpu_op', pid=self.pid, ts=14000, dur=500, name='API2')]

    def parse(self, events=None):
        return profile.trace_summary(self.events if events is None else events,
                                     self.pid, 10_000_000, 20_000_000)

    def test_overlapping_service_is_not_added_to_wall_and_payload_is_named(self):
        result = self.parse()
        self.assertEqual(result['gpu_union_us'], 3100)
        self.assertEqual(result['host_api_union_in_window_us'], 2000)
        self.assertEqual(result['controlled_request_wall_us'], 10000)
        self.assertEqual(result['operation_counts'], dict(kernel=2, copy=1, barrier=1))
        self.assertEqual(result['observed_copy_fill_payload_bytes'],
                         {'zeCommandListAppendMemoryCopy(H2D)': 4096})
        self.assertIn('No exact native prefill endpoint', result['limits'])

    def test_invalid_numeric_service_is_refused(self):
        for key, value in (('ts', True), ('ts', float('nan')), ('dur', float('inf')),
                           ('dur', -1), ('dur', 0)):
            events = copy.deepcopy(self.events)
            events[2][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                self.parse(events)

    def test_outside_window_and_wrong_pci_are_refused(self):
        for mutation in ('before', 'after', 'device'):
            events = copy.deepcopy(self.events)
            if mutation == 'before':
                events[2]['ts'] = 8000
            elif mutation == 'after':
                events[2]['ts'] = 19900
            else:
                events[0]['args']['name'] = 'DEVICE<host>[B570] 0:6:0:0 #0'
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                self.parse(events)

    def test_missing_ambiguous_and_foreign_host_sources_are_refused(self):
        for mutation in ('missing', 'duplicate', 'foreign'):
            events = copy.deepcopy(self.events)
            if mutation == 'missing':
                del events[1]
            elif mutation == 'duplicate':
                events.append(copy.deepcopy(events[1]))
            else:
                events[1]['pid'] = self.pid+1
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                self.parse(events)

    def test_duplicate_gpu_id_and_foreign_cpu_process_are_refused(self):
        for mutation in ('id', 'host'):
            events = copy.deepcopy(self.events)
            if mutation == 'id':
                events[4]['args']['id'] = 1
            else:
                events[-1]['pid'] = self.pid+1
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                self.parse(events)

    def test_missing_payload_and_empty_stream_are_refused(self):
        events = copy.deepcopy(self.events)
        events[6]['name'] = 'zeCommandListAppendMemoryCopy(H2D)'
        for value in (events, []):
            with self.assertRaises(RuntimeError):
                self.parse(value)

    def test_invalid_window_identity_is_refused(self):
        for native, begin, end in ((True, 1, 2), (1234, True, 2), (1234, 2, 1)):
            with self.subTest(identity=(native, begin, end)), self.assertRaises(RuntimeError):
                profile.trace_summary(self.events, native, begin, end)


class StartupGates(unittest.TestCase):
    def test_missing_mutated_and_extra_startup_variables_are_refused(self):
        expected = dict(ZES_ENABLE_SYSMAN='0', UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1')
        profile.same_environment(dict(expected, PWD='/work'), expected)
        for value in ({}, dict(expected, ZES_ENABLE_SYSMAN='1'),
                      dict(expected, UNITRACE_KernelMetrics='1')):
            with self.assertRaises(RuntimeError):
                profile.same_environment(value, expected)

    def test_profile_flags_do_not_replace_safety_or_target_preload(self):
        row = dict(native_validation_probe=dict(library=dict(path='/probe.so')))
        profiling = dict(collector=dict(path='/collector.so'), session='Owned123',
                         environment=dict(UNITRACE_SystemTime='1'))
        env = dict(ZES_ENABLE_SYSMAN='0', EnableDirectSubmission='0')
        with patch.object(profile.engine_once, 'native_environment',
                          return_value=dict(env, LD_PRELOAD='/probe.so')):
            actual = profile.expected_target_environment(row, env, profiling)
        self.assertEqual(actual['ZES_ENABLE_SYSMAN'], '0')
        self.assertEqual(actual['EnableDirectSubmission'], '0')
        self.assertEqual(actual['LD_PRELOAD'], '/collector.so:/probe.so')
        self.assertEqual(actual['STRATA_UNITRACE_TARGET_PRELOAD'], '/probe.so')
        self.assertEqual(actual['UNITRACE_Session'], 'Owned123')
        self.assertEqual(actual['UNITRACE_ChromeEventBufferSize'], '128')
        self.assertEqual(actual['UNITRACE_VERSION'],
                         '2.4.1 (4b6d74294cdc72766c00a9211fe2e4261ee94244)')
        self.assertEqual(actual['UNITRACE_LD_PRELOAD_OLD'], '/probe.so')
        for key, value in [('UNITRACE_VERSION', 'other'), ('UNITRACE_LD_PRELOAD_OLD', '')]:
            with self.subTest(variable=key), self.assertRaises(RuntimeError):
                profile.same_environment(dict(actual, **{key:value}), actual)
        self.assertEqual(env, dict(ZES_ENABLE_SYSMAN='0', EnableDirectSubmission='0'))

    def test_session_name_cannot_address_another_path(self):
        self.assertTrue(profile.session_path('Owned123').name.endswith('_Owned123'))
        for value in ('', '../x', 'x/y', 'x y'):
            with self.assertRaises(RuntimeError):
                profile.session_path(value)


if __name__ == '__main__':
    unittest.main()
