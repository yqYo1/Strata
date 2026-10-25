"""Host-only adversarial controls for the upcoming native profiler fixture.

These exercise synthetic evidence, not an installed collector or GPU device.
The route plan is literal; the independent API side expands captured templates.
"""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
from work_ledger import CommandBuffers, DeviceEvent, Operation, Submission, Work, reconcile


def fixture():
    # Queues/event handles may repeat, but lifetime/execution identities may not.
    q, reused_q = 'queue0/g1', 'queue0/g2'
    add, multiply = Work('kernel', 'add', (257,)), Work('kernel', 'multiply', (257,))
    upload, download = Work('copy', 'H2D', bytes=1164), Work('copy', 'D2H', bytes=1164)
    fill, barrier = Work('fill', 'zero', bytes=1164), Work('barrier')
    plan = [Operation('upload', q, 'fixture', (upload,)),
            Operation('fill', q, 'fixture', (fill,)),
            Operation('direct', q, 'fixture', (add,)),
            Operation('barrier', q, 'fixture', (barrier,)),
            Operation('host', q, 'fixture', host_task=True),
            Operation('graph0', q, 'fixture', (add, multiply)),
            Operation('graph1', q, 'fixture', (add, multiply)),
            Operation('new-graph', reused_q, 'fixture', (multiply,)),
            Operation('download', reused_q, 'fixture', (download,)),
            Operation('invalid-admission', reused_q, 'fixture', (add,), result='rejected')]
    api, events = [], []

    def submit(ident, queue, work=(), *, host=False, result='accepted'):
        links = []
        for node in work:
            sequence = len(events)
            event_id = 'reused-native-event/execution-' + str(sequence)
            events.append(DeviceEvent(event_id, ident, queue, node, 10 + sequence, 11 + sequence))
            links.append(event_id)
        api.append(Submission(ident, queue, 'fixture', result, work, tuple(links), host))

    submit('upload', q, (Work('copy', 'H2D', bytes=1164),))
    submit('fill', q, (Work('fill', 'zero', bytes=1164),))
    submit('direct', q, (Work('kernel', 'add', (257,)),))
    submit('barrier', q, (Work('barrier'),))
    submit('host', q, host=True)
    capture = CommandBuffers()
    old = capture.create('recycled-buffer')
    capture.append(old, Work('kernel', 'add', (257,)))
    capture.append(old, Work('kernel', 'multiply', (257,)))
    capture.finalize(old)
    submit('graph0', q, capture.replay(old))
    submit('graph1', q, capture.replay(old))
    capture.destroy(old)
    new = capture.create('recycled-buffer')
    capture.append(new, Work('kernel', 'multiply', (257,)))
    capture.finalize(new)
    submit('new-graph', reused_q, capture.replay(new))
    capture.destroy(new)
    submit('download', reused_q, (Work('copy', 'D2H', bytes=1164),))
    submit('invalid-admission', reused_q, result='rejected')
    return dict(plan=plan, submissions=api, events=events, phase_bounds={'fixture': (0, 100)},
                drained=True, async_errors=())


class WorkLedgerTests(unittest.TestCase):
    def test_independent_plan_graph_replays_reuse_and_host_work_balance(self):
        self.assertEqual(reconcile(**fixture()), dict(planned_operations=10, accepted_operations=9,
            rejected_operations=1, host_tasks=1, device_counts=dict(kernel=6, copy=2, fill=1, barrier=1),
            requested_bytes=dict(copy=2328, fill=1164)))

    def test_missing_expected_api_or_device_record_is_detected(self):
        for ledger in ('plan', 'submissions', 'events'):
            data = fixture()
            del data[ledger][0]
            with self.subTest(ledger=ledger), self.assertRaises(ValueError):
                reconcile(**data)

    def test_duplicate_source_api_or_device_identity_is_detected(self):
        for ledger in ('plan', 'submissions', 'events'):
            data = fixture()
            data[ledger].append(data[ledger][0])
            with self.subTest(ledger=ledger), self.assertRaises(ValueError):
                reconcile(**data)

    def test_graph_template_capture_is_not_an_extra_execution(self):
        data = fixture()
        data['events'].append(DeviceEvent('capture-node', 'graph0', 'queue0/g1',
            Work('kernel', 'add', (257,)), 1, 2))
        with self.assertRaisesRegex(ValueError, 'unmapped device'):
            reconcile(**data)

    def test_api_dropping_one_graph_child_cannot_hide_a_replay_loss(self):
        data = fixture()
        row = data['submissions'][5]
        data['submissions'][5] = replace(row, work=row.work[:1], event_ids=row.event_ids[:1])
        with self.assertRaisesRegex(ValueError, 'expansion mismatch'):
            reconcile(**data)

    def test_balanced_but_wrong_origin_or_queue_generation_is_rejected(self):
        for field, value in (('origin', 'graph0'), ('queue', 'queue0/g2')):
            data = fixture()
            data['events'][0] = replace(data['events'][0], **{field: value})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'origin/queue/work'):
                reconcile(**data)

    def test_equal_total_bytes_do_not_hide_wrong_transfer_direction(self):
        data = fixture()
        data['events'][0] = replace(data['events'][0], work=Work('copy', 'D2H', bytes=1164))
        with self.assertRaisesRegex(ValueError, 'origin/queue/work'):
            reconcile(**data)

    def test_equal_kernel_count_does_not_hide_wrong_shape(self):
        data = fixture()
        data['events'][2] = replace(data['events'][2], work=Work('kernel', 'add', (256,)))
        with self.assertRaisesRegex(ValueError, 'origin/queue/work'):
            reconcile(**data)

    def test_one_device_event_cannot_satisfy_two_submissions(self):
        data = fixture()
        row = data['submissions'][6]
        data['submissions'][6] = replace(row, event_ids=data['submissions'][5].event_ids)
        with self.assertRaisesRegex(ValueError, 'duplicate device link'):
            reconcile(**data)

    def test_failed_submit_cannot_be_silently_dropped_or_counted_as_work(self):
        for change in (dict(result='accepted'), dict(work=(Work('barrier'),)),
                       dict(event_ids=('invented',)), dict(host_completed=True)):
            data = fixture()
            data['submissions'][-1] = replace(data['submissions'][-1], **change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                reconcile(**data)

    def test_host_task_requires_its_own_completion_without_device_work(self):
        data = fixture()
        data['submissions'][4] = replace(data['submissions'][4], host_completed=False)
        with self.assertRaisesRegex(ValueError, 'host task completion'):
            reconcile(**data)

    def test_async_error_or_uncertain_drain_invalidates_complete_counts(self):
        for change in (dict(drained=False), dict(async_errors=('mock async error',)), dict(drained=1)):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'uncertain drain'):
                reconcile(**dict(fixture(), **change))

    def test_zero_duration_kernel_bad_clock_or_out_of_phase_event_fails(self):
        for begin, end in ((10, 10), (20, 10), (-1, 1), (99, 101), (100, 100),
                           (float('nan'), 12), (True, 12)):
            data = fixture()
            data['events'][2] = replace(data['events'][2], begin_ns=begin, end_ns=end)
            with self.subTest(interval=(begin, end)), self.assertRaisesRegex(ValueError, 'device clock'):
                reconcile(**data)

    def test_zero_duration_barrier_is_allowed_but_not_outside_the_phase(self):
        data = fixture()
        data['events'][3] = replace(data['events'][3], begin_ns=12, end_ns=12)
        self.assertEqual(reconcile(**data)['device_counts']['barrier'], 1)

    def test_missing_or_overlapping_phase_bounds_are_rejected(self):
        for bounds in ({}, {'other': (0, 100)}, {'fixture': (0, 100), 'decode': (99, 101)},
                       {'fixture': (100, 0)}, {'fixture': (0, True)}):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                reconcile(**dict(fixture(), phase_bounds=bounds))

    def test_unknown_or_malformed_work_is_not_an_unclassified_bucket(self):
        for work in (Work('unknown'), Work('kernel', 'add', (0,)), Work('copy', 'H2D', bytes=-1),
                     Work('barrier', bytes=1), Work('fill', 'zero', bytes=True)):
            with self.subTest(work=work), self.assertRaises(ValueError):
                work.validate()

    def test_command_buffer_reuse_requires_a_new_generation(self):
        buffers = CommandBuffers()
        old = buffers.create('same-handle')
        buffers.append(old, Work('barrier'))
        buffers.finalize(old)
        buffers.destroy(old)
        new = buffers.create('same-handle')
        self.assertEqual((old, new), (('same-handle', 1), ('same-handle', 2)))
        for action in (buffers.replay, buffers.finalize, buffers.destroy):
            with self.subTest(action=action.__name__), self.assertRaisesRegex(ValueError, 'stale'):
                action(old)

    def test_failed_or_incomplete_graph_lifecycle_cannot_expand(self):
        buffers = CommandBuffers()
        graph = buffers.create('graph')
        for action in (lambda: buffers.create('graph'), lambda: buffers.replay(graph),
                       lambda: buffers.finalize(graph)):
            with self.assertRaises(ValueError):
                action()
        buffers.append(graph, Work('barrier'))
        buffers.finalize(graph)
        for action in (lambda: buffers.finalize(graph), lambda: buffers.append(graph, Work('barrier'))):
            with self.assertRaises(ValueError):
                action()


if __name__ == '__main__':
    unittest.main()
