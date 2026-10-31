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

    def test_shape_groups_and_observed_gaps_are_reduced_without_losing_work(self):
        events = copy.deepcopy(self.events)
        base = 'strata::experimental::vllm_xe2_native16::DenseKernel<half,bfloat16,Tile1>'
        events[2]['name'] = base+'[SIMD16 {1;20;64} {128;1;1}]'
        events[4]['name'] = base+'[SIMD16 {1;20;32} {128;1;1}]'
        result = profile.family_summary(events, 10_000_000, 20_000_000)
        families = {r['name']:r for r in result['families']}
        self.assertEqual(sum(r['count'] for r in families.values()), 4)
        self.assertEqual(families['native16_dense_gemm']['count'], 2)
        self.assertAlmostEqual(families['native16_dense_gemm']['service_sum_s'], .004)
        self.assertEqual(families['copy:H2D']['payload_bytes'], 4096)
        self.assertEqual(result['largest_normalized_kernel_groups'][0]['name'], base)
        self.assertEqual(result['largest_normalized_kernel_groups'][0]['count'], 2)
        self.assertAlmostEqual(result['kernel_interval_union_s'], .003)
        self.assertAlmostEqual(result['observed_gpu_gap_total_s'], .0069)
        self.assertAlmostEqual(result['largest_internal_observed_gaps'][0]
                               ['host_api_overlap_s']['API2'], .0004)

    def test_fused_and_attention_symbols_have_explicit_source_families(self):
        for symbol, family in [('x::SplitDownScatterKernel<false>', 'native16_down_fused'),
            ('x::GuSwiGLUKernel<Tile1,false>', 'native16_gu_fused'),
            ('prefix_batch_impl<33>', 'qsa_prefix'), ('unknown_symbol', 'other_kernel')]:
            self.assertEqual(profile.kernel_family(symbol), family)


class StatusGates(unittest.TestCase):
    def test_status_partition_conserves_and_rejects_errors_or_missing_regions(self):
        data = dict(version='1', available='1', complete='1', scopes='1', completed='1',
            open_scopes='0', open_parts='0', cpu_s='1', wall_s='2',
            unclassified_cpu_s='.1', unclassified_wall_s='.2')
        for name in ('tag', 'precheck', 'queue_context', 'enqueue', 'observer', 'drain', 'postcheck', 'after_drain', 'branch'):
            data.update({name+'_count':'1', name+'_throws':'0', name+'_cpu_s':'.1',
                name+'_wall_s':'.2', name+'_wait_cpu_s':'.05', name+'_wait_wall_s':'.1'})
        def text(value):
            return 'strata prefill status: '+' '.join(k+'='+v for k,v in value.items())
        result = profile.status_summary(text(data))
        self.assertEqual(result['scopes'], 1)
        self.assertAlmostEqual(result['parts'][0]['outside_known_wait_cpu_s'], .05)
        for mutation in ('throw', 'sum', 'missing', 'open'):
            other = dict(data)
            if mutation == 'throw': other['enqueue_throws'] = '1'
            elif mutation == 'sum': other['cpu_s'] = '1.1'
            elif mutation == 'missing': del other['enqueue_throws']
            else: other['open_scopes'] = '1'
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                profile.status_summary(text(other))


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
