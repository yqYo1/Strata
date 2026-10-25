"""Host-only negative controls for complete UR subscriber call records."""
from pathlib import Path
import copy
import json
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
from ur_call_subscriber import parse_calls, correlate_calls, metadata_census


def records():
    return [dict(kind='load', pid=71, raw_ns=1),
            dict(kind='subscribe', stream='ur.call', pid=71, raw_ns=2),
            dict(kind='begin', pid=71, tid=72, instance=100, function_id=8,
                 name='urPlatformGetApiVersion', raw_ns=3, status=None),
            dict(kind='end', pid=71, tid=72, instance=100, function_id=8,
                 name='urPlatformGetApiVersion', raw_ns=4, status=46),
            dict(kind='finish', stream='ur.call', pid=71, raw_ns=5)]


def parse(rows):
    return parse_calls(''.join(json.dumps(row) + '\n' for row in rows), 71)


class SubscriberRecordTests(unittest.TestCase):
    def test_records_rejected_call_and_actual_clock_identity_without_GPU_claim(self):
        value = parse(records())
        self.assertEqual(value['call_pairs'], 1)
        self.assertEqual(value['outcomes'], [dict(name='urPlatformGetApiVersion', status=46, calls=1)])
        self.assertFalse(value['device_work_qualified'])
        self.assertFalse(value['ordinary_runtime_teardown_qualified'])

    def test_missing_load_subscription_begin_or_end_fails(self):
        for index in range(4):
            rows = records()
            del rows[index]
            with self.subTest(index=index), self.assertRaises((RuntimeError, ValueError)):
                parse(rows)

    def test_duplicate_or_out_of_order_records_fail(self):
        for index in range(5):
            rows = records()
            rows.insert(index, copy.deepcopy(rows[index]))
            with self.subTest(index=index), self.assertRaises(RuntimeError):
                parse(rows)

    def test_wrong_owner_thread_function_clock_or_status_fails(self):
        for change in (dict(pid=73), dict(tid=73), dict(function_id=9), dict(instance=101),
                       dict(name='urOther'), dict(raw_ns=2), dict(raw_ns=float('nan')),
                       dict(status=None), dict(status=True), dict(status=1 << 32)):
            rows = records()
            rows[3].update(change)
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                parse(rows)

    def test_nested_calls_need_matching_per_thread_stack(self):
        rows = records()
        nested = dict(rows[2], instance=101, function_id=9, name='urEventRelease', raw_ns=3)
        rows[3:3] = [nested, dict(nested, kind='end', status=0)]
        self.assertEqual(parse(rows)['call_pairs'], 2)
        rows[3], rows[4], rows[5] = rows[3], rows[5], rows[4]
        with self.assertRaises(RuntimeError):
            parse(rows)

    def test_function_ID_name_contract_cannot_change_between_calls(self):
        rows = records()
        rows[4:4] = [dict(rows[2], instance=101, name='urEventRelease'),
                     dict(rows[3], instance=101, name='urEventRelease')]
        with self.assertRaisesRegex(RuntimeError, 'function identity'):
            parse(rows)

    def test_finish_may_be_absent_with_retained_runtime_reference(self):
        rows = records()[:-1]
        self.assertEqual(parse(rows)['call_pairs'], 1)
        rows.append(dict(rows[2], instance=100))
        with self.assertRaises(RuntimeError):
            parse(rows)


class DirectJoinTests(unittest.TestCase):
    def fixture(self):
        spans, events = [], []
        for index in range(2):
            ident, stamp = index + 1, 10 + index * 20
            spans.append(dict(instance=100 + index, tid=72, name='urEnqueueKernelLaunchWithArgsExp',
                function_id=8, begin_ns=(stamp - 1) * 1000, end_ns=(stamp + 2) * 1000, status=0))
            events.extend([
                dict(ph='X', cat='cpu_op', name='zeCommandListAppendLaunchKernelWithArguments',
                     pid=71, tid=72, ts=stamp, dur=1, id=0),
                dict(ph='s', cat=f'Flow_H2D_{ident}_71', pid=71, tid=72, ts=stamp, id=ident),
                dict(ph='t', cat=f'Flow_H2D_{ident}_71', pid=90, tid=91, ts=stamp + 5, id=ident),
                dict(ph='X', cat='gpu_op', name='main::kernel', pid=90, tid=91,
                     ts=stamp + 5, dur=1, args={'id': str(ident)}),
                dict(ph='t', cat=f'Flow_D2H_{ident}_71', pid=71, tid=72, ts=stamp + 7, id=ident)])
        return spans, events

    def join(self, spans, events):
        return correlate_calls(spans, events, 71, {'urEnqueueKernelLaunchWithArgsExp': 2})

    def test_valid_intervals_retain_actual_instance_clock_and_thread(self):
        value = parse_calls('\n'.join(json.dumps(r) for r in records()), 71, intervals=True)
        self.assertEqual(value['intervals'], [dict(instance=100, tid=72, function_id=8,
            name='urPlatformGetApiVersion', begin_ns=3, end_ns=4, status=46)])

    def test_shuffled_records_join_by_actual_thread_clock_not_ordinal(self):
        spans, events = self.fixture()
        result = self.join(list(reversed(spans)), list(reversed(events)))
        self.assertEqual(result['linked_UR_instances'], 2)
        self.assertTrue(result['independent_UR_to_PTI_links'])
        self.assertFalse(result['shape_lifetime_graph_qualified'])
        self.assertFalse(result['observer_cost_qualified'])

    def test_same_counts_do_not_hide_wrong_thread_clock_kind_or_status(self):
        for change in (dict(tid=73), dict(begin_ns=11000), dict(end_ns=9999),
                       dict(name='urEnqueueUSMMemcpy'), dict(status=1), dict(name='urEnqueueUnknown')):
            spans, events = self.fixture()
            spans[0].update(change)
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                self.join(spans, events)

    def test_overlapping_or_reused_UR_instances_fail(self):
        for change in (dict(begin_ns=11000), dict(instance=100)):
            spans, events = self.fixture()
            spans[1].update(change)
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                self.join(spans, events)

    def test_unlinked_UR_call_or_missing_device_execution_fails(self):
        for remove in ('UR', 'GPU'):
            spans, events = self.fixture()
            if remove == 'UR':
                spans.pop()
            else:
                del events[3]
            with self.subTest(remove=remove), self.assertRaises(RuntimeError):
                self.join(spans, events)

    def test_one_UR_interval_cannot_absorb_two_device_appends(self):
        spans, events = self.fixture()
        spans[0]['end_ns'] = 35000
        spans[1]['begin_ns'], spans[1]['end_ns'] = 36000, 38000
        with self.assertRaisesRegex(RuntimeError, 'duplicated device-to-UR'):
            self.join(spans, events)

    def test_broken_internal_flow_cannot_gain_independent_UR_qualification(self):
        spans, events = self.fixture()
        events[2]['tid'] = 92
        with self.assertRaisesRegex(RuntimeError, 'GPU flow endpoint'):
            self.join(spans, events)


class MetadataCaptureTests(unittest.TestCase):
    def fixture(self):
        return [dict(name='urEnqueueKernelLaunchWithArgsExp', status=0,
                     metadata=dict(queue=0x101, kernel=0x401, work_dim=1,
                                   **{'global': [257], 'offset': [], 'local': [],
                                      'wait_count': 0, 'waits': [], 'produced_event': 0})),
                dict(name='urEnqueueUSMMemcpy', status=0,
                     metadata=dict(queue=0x101, src=0x1000, dst=0x2000, bytes=1028,
                                   wait_count=0, waits=[], produced_event=0))]

    def test_complete_typed_counts_keep_lifetime_and_graph_gates_open(self):
        value = metadata_census(self.fixture())
        self.assertEqual(value['counts']['copy_bytes'], 1028)
        self.assertEqual(value['enqueue_queue_handles'], 1)
        self.assertFalse(value['lifetime_generations_qualified'])
        self.assertFalse(value['graph_children_qualified'])

    def test_lost_required_metadata_does_not_become_an_empty_default(self):
        for key in ('queue', 'kernel', 'work_dim', 'global', 'local', 'offset', 'wait_count', 'waits', 'produced_event'):
            spans = self.fixture();del spans[0]['metadata'][key]
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                metadata_census(spans)

    def test_invalid_type_range_geometry_and_wait_extent_fail(self):
        for change in (dict(queue=True), dict(kernel=-1), dict(work_dim=4), {'global': [0]},
                       {'global': [float('nan')]}, dict(local=[1, 2]), dict(waits=[0x201]),
                       dict(bytes=1 << 64)):
            spans = self.fixture();spans[0]['metadata'].update(change)
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                metadata_census(spans)

    def test_missing_transfer_address_or_extent_fails(self):
        for key in ('src', 'dst', 'bytes'):
            spans = self.fixture();del spans[1]['metadata'][key]
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                metadata_census(spans)


if __name__ == '__main__':
    unittest.main()
