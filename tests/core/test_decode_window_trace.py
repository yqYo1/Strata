"""Window accounting and refusal cases; repository LICENSE, Main-authored."""
import copy
import importlib.util
import math
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[2] / 'tools/sycl/decode_window_trace.py'
spec = importlib.util.spec_from_file_location('decode_window_trace', path)
window = importlib.util.module_from_spec(spec)
spec.loader.exec_module(window)


def event(name, start, duration, work=1, **extra):
    return dict(ph='X', cat='gpu_op', name=name, ts=start, dur=duration,
                pid=99, args=dict(id=work), **extra)


def trace(*gpu):
    rows = [dict(ph='M', name='process_name', pid=99,
                 args=dict(name='DEVICE<0> [TILE0] 0:5:0:0 #1')),
            dict(ph='X', cat='cpu_op', name='zeEventQueryStatus', ts=1000., dur=20., pid=7)]
    for item in gpu:
        rows.extend([dict(ph='s', cat='Flow_H2D_kernel', pid=7, ts=500., id=item['args']['id']), item])
    return rows


def summarize(rows, **options):
    args = dict(native_pid=7, begin_ns=1000000, end_ns=1020000,
                capture_begin_ns=900000, capture_end_ns=1100000)
    args.update(options)
    return window.summarize(rows, **args)


class WindowAccounting(unittest.TestCase):
    def test_overlapping_device_intervals_are_not_added(self):
        row = summarize(trace(event('kernel', 1000., 10., 1), event('kernel', 1005., 10., 2)))
        self.assertEqual(row['observed_gpu_interval_union_us'], 15.)
        self.assertEqual(row['unobserved_gpu_event_gap_us'], 5.)
        self.assertEqual(row['largest_observed_device_services'][0]['clipped_sum_us'], 20.)
        self.assertFalse(row['physical_saturation_qualified'])
        self.assertFalse(row['complete_device_coverage_qualified'])

    def test_crossing_copy_payload_is_not_prorated_or_assigned(self):
        rows = trace(event('kernel', 1002., 2.),
                     event('zeCommandListAppendMemoryCopy(H2D)[4096]', 990., 50., 2))
        row = summarize(rows)
        self.assertEqual(row['observed_copy_interval_union_us'], 20.)
        self.assertEqual(row['wholly_contained_requested_payload_bytes'], {})
        self.assertEqual(row['crossing_command_requested_bytes_not_assigned_to_window'], {'copy': 4096})
        self.assertEqual(row['boundary_crossing_counts'], {'copy': 1})

    def test_wholly_contained_copy_keeps_requested_bytes_separate(self):
        row = summarize(trace(event('kernel', 1000., 2.),
                        event('zeCommandListAppendMemoryCopy(D2H)[1024]', 1005., 3., 2)))
        self.assertEqual(row['wholly_contained_requested_payload_bytes'], {'copy': 1024})
        self.assertEqual(row['crossing_command_requested_bytes_not_assigned_to_window'], {})

    def test_capture_time_source_can_precede_replay_window(self):
        row = summarize(trace(event('replayed_graph_child', 1001., 3.)))
        self.assertEqual(row['operation_counts'], {'kernel': 1})
        self.assertTrue(row['included_native_flow_joins_unique'])

    def test_missing_or_ambiguous_source_is_refused(self):
        rows = trace(event('kernel', 1001., 3.))
        for bad in (rows[:2] + rows[3:], rows + [copy.deepcopy(rows[2])]):
            with self.assertRaisesRegex(ValueError, 'source'):
                summarize(bad)

    def test_unique_replay_ids_allowed_duplicate_ids_refused(self):
        summarize(trace(event('same_graph_child', 1001., 2., 1),
                        event('same_graph_child', 1005., 2., 2)))
        rows = trace(event('kernel', 1001., 2.))
        duplicate = copy.deepcopy(rows[-1]); duplicate['ts'] = 1008.
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            summarize(rows + [duplicate])

    def test_wrong_device_and_host_are_refused(self):
        rows = trace(event('kernel', 1001., 2.))
        wrong_device = copy.deepcopy(rows); wrong_device[0]['args']['name'] = 'DEVICE<1> [TILE0] 0:6:0:0 #1'
        wrong_host = copy.deepcopy(rows); wrong_host[1]['pid'] = 8
        for bad in (wrong_device, wrong_host):
            with self.assertRaises(ValueError):
                summarize(bad)

    def test_nonfinite_negative_and_boolean_times_are_refused(self):
        for value in (math.nan, math.inf, -1., True):
            rows = trace(event('kernel', 1001., 2.)); rows[-1]['dur'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize(rows)

    def test_zero_duration_kernel_does_not_qualify(self):
        with self.assertRaises(ValueError):
            summarize(trace(event('kernel', 1001., 0.)))

    def test_host_api_union_does_not_add_nested_waits(self):
        rows = trace(event('kernel', 1001., 2.))
        rows.append(dict(ph='X', cat='cpu_op', name='zeCommandQueueSynchronize',
                         ts=1004., dur=10., pid=7))
        row = summarize(rows)
        self.assertEqual(row['observed_host_api_interval_union_us'], 20.)
        self.assertEqual(sum(x['clipped_sum_us'] for x in row['largest_observed_host_apis']), 30.)

    def test_window_must_be_within_declared_collection(self):
        for options in (dict(begin_ns=0), dict(end_ns=999999),
                        dict(capture_begin_ns=1000001), dict(capture_end_ns=1019999)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                summarize(trace(event('kernel', 1001., 2.)), **options)

    def test_outside_work_is_counted_without_extending_service(self):
        row = summarize(trace(event('kernel', 1001., 2., 1), event('outside', 1090., 3., 2)))
        self.assertEqual(row['observed_gpu_interval_union_us'], 2.)
        self.assertEqual(row['outside_window_events'], {'gpu_op': 1})

    def test_copy_with_unknown_payload_cannot_report_bytes(self):
        with self.assertRaisesRegex(ValueError, 'payload'):
            summarize(trace(event('kernel', 1001., 2.),
                            event('zeCommandListAppendMemoryCopy', 1005., 2., 2)))

    def test_absent_gpu_or_host_does_not_become_zero_utilization(self):
        rows = trace(event('kernel', 1001., 2.))
        for bad in (rows[:2], [item for item in rows if item.get('cat') != 'cpu_op']):
            with self.assertRaises(ValueError):
                summarize(bad)


if __name__ == '__main__':
    unittest.main()
