"""Host-only negative controls for complete UR subscriber call records."""
from pathlib import Path
import copy
import json
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
from ur_call_subscriber import parse_calls, correlate_calls, metadata_census, lifetime_census


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


class ExternalLifetimeTests(unittest.TestCase):
    def fixture(self):
        return self.schedule([
            ('urQueueCreate', dict(context=0x100, created_queue=0x101)),
            ('urUSMDeviceAlloc', dict(context=0x100, allocation=0x1000, bytes=1028)),
            ('urEnqueueKernelLaunchWithArgsExp', dict(queue=0x101, waits=[], wait_count=0, produced_event=0x201)),
            ('urEventWait', dict(waits=[0x201], wait_count=1)),
            ('urEventRelease', dict(event=0x201)),
            ('urUSMFree', dict(context=0x100, allocation=0x1000)),
            ('urQueueRelease', dict(queue=0x101))])

    def schedule(self, events):
        return [dict(name=name, metadata=meta, status=0, instance=i, tid=72, begin_ns=i * 3, end_ns=i * 3 + 1)
                for i, (name, meta) in enumerate(events)]

    def serial(self, spans):
        for i, span in enumerate(spans):
            span.update(instance=i, begin_ns=i * 3, end_ns=i * 3 + 1)
        return spans

    def test_recycled_values_produce_distinct_closed_generations(self):
        spans = self.serial(self.fixture() + self.fixture())
        value = lifetime_census(spans)
        self.assertTrue(value['external_generations_closed'])
        for kind in ('queue', 'event', 'allocation'):
            self.assertEqual(value['counts'][kind + '_created'], 2)
            self.assertEqual(value['counts'][kind + '_reused_values'], 1)

    def test_live_duplicate_unknown_release_or_lost_retirement_fails(self):
        for corrupt in ('duplicate-create', 'duplicate-event', 'unknown-release', 'lost-free', 'lost-queue'):
            spans = self.fixture()
            if corrupt == 'duplicate-create':
                spans.insert(1, copy.deepcopy(spans[0]))
            elif corrupt == 'duplicate-event':
                spans.insert(3, copy.deepcopy(spans[2]))
            elif corrupt == 'unknown-release':
                spans[4]['metadata']['event'] = 0x202
            elif corrupt == 'lost-free':
                del spans[5]
            else:
                del spans[0]
            with self.subTest(corrupt=corrupt), self.assertRaises(RuntimeError):
                lifetime_census(self.serial(spans))

    def test_retain_requires_the_matching_final_release(self):
        spans = self.fixture()
        spans.insert(3, dict(spans[4], name='urEventRetain'))
        with self.assertRaisesRegex(RuntimeError, 'unretired'):
            lifetime_census(self.serial(spans))
        spans.insert(6, copy.deepcopy(spans[5]))
        self.assertTrue(lifetime_census(self.serial(spans))['external_generations_closed'])

    def test_rejected_call_cannot_create_or_retire_a_generation(self):
        spans = self.fixture()
        spans.insert(0, dict(spans[0], status=46))
        spans.insert(4, dict(spans[-1], status=46))
        value = lifetime_census(self.serial(spans))
        self.assertEqual(value['counts']['queue_created'], 1)
        self.assertEqual(value['counts']['rejected_without_transition'], 2)

    def graph_fixture(self):
        return self.schedule([
            ('urQueueCreate', dict(context=0x100, created_queue=0x101)),
            ('urCommandBufferCreateExp', dict(context=0x100, created_graph=0x301)),
            ('urCommandBufferAppendKernelLaunchExp', dict(graph=0x301)),
            ('urCommandBufferAppendKernelLaunchExp', dict(graph=0x301)),
            ('urCommandBufferFinalizeExp', dict(graph=0x301)),
            ('urEnqueueCommandBufferExp', dict(queue=0x101, graph=0x301)),
            ('urEnqueueCommandBufferExp', dict(queue=0x101, graph=0x301)),
            ('urCommandBufferReleaseExp', dict(graph=0x301)),
            ('urQueueRelease', dict(queue=0x101))])

    def test_two_graph_replays_expand_the_declared_children(self):
        value = lifetime_census(self.graph_fixture())
        self.assertEqual(value['counts']['graph_children'], 2)
        self.assertEqual(value['counts']['graph_replays'], 2)
        self.assertEqual(value['counts']['declared_replay_children'], 4)

    def test_wrong_context_unfinalized_and_changed_graph_are_rejected(self):
        for corrupt in ('context', 'finalize', 'late-append'):
            spans = self.graph_fixture()
            if corrupt == 'context':
                spans[1]['metadata']['context'] = 0x200
            elif corrupt == 'finalize':
                del spans[4]
            else:
                spans.insert(5, copy.deepcopy(spans[2]))
            with self.subTest(corrupt=corrupt), self.assertRaises(RuntimeError):
                lifetime_census(self.serial(spans))

    def test_concurrent_or_nested_calls_require_another_adapter(self):
        for corrupt in ('thread', 'overlap'):
            spans = self.fixture()
            spans[1].update(dict(tid=73) if corrupt == 'thread' else dict(begin_ns=0))
            with self.subTest(corrupt=corrupt), self.assertRaises(RuntimeError):
                lifetime_census(spans)


if __name__ == '__main__':
    unittest.main()
