"""Reject incomplete raw-count evidence before the native collector probe."""
from pathlib import Path
import copy
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
import work_ledger_fixture as probe


def call(name, args='', status='UR_RESULT_SUCCESS'):
    return '<--- ' + name + '(' + args + ') -> ' + status + ';\n'


class CensusProbeTests(unittest.TestCase):
    def test_actual_source_route_counts_are_independent_of_trace(self):
        self.assertEqual(probe.source_census(), dict(kernel=606, copy=53, fill=1, barrier=20,
                                                   copy_bytes=55844, fill_bytes=1028))

    def test_host_wait_does_not_manufacture_an_extra_device_barrier(self):
        trace = call('urEnqueueUSMMemcpy', '.size = 1028') + call('urEnqueueUSMFill', '.size = 1028')
        trace += call('urEnqueueKernelLaunch') + call('urEnqueueEventsWaitWithBarrier')
        trace += call('urEnqueueEventsWait')
        self.assertEqual(probe.api_census(trace)['counts'],
            dict(kernel=1, copy=1, fill=1, barrier=1, copy_bytes=1028, fill_bytes=1028))

    def test_unknown_failed_or_missing_api_work_fails(self):
        for text in ('', call('urEnqueueUnexpected'), call('urEnqueueUSMMemcpy'),
                     call('urEnqueueKernelLaunch', status='UR_RESULT_ERROR_INVALID_VALUE'),
                     call('urCommandBufferCreateExp')):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                probe.api_census(text)

    def timeline(self):
        events = [dict(ph='X', cat='cpu_op', name='zeCommandListAppendLaunchKernel', ts=10, dur=1)]
        for ident, name in enumerate(('main::kernel', 'zeCommandListAppendMemoryCopy(H2D)[1028]',
                                      'zeCommandListAppendBarrier')):
            events.append(dict(ph='X', cat='gpu_op', name=name, ts=20 + ident, dur=1, args={'id': str(ident)}))
            events.append(dict(ph='s', cat='Flow_H2D_0', id=str(ident), pid=71))
        return events

    def test_actual_device_ids_flows_bytes_and_clocks_are_required(self):
        events = self.timeline()
        self.assertEqual(probe.device_census(events, 71, 0, 100000)['counts'],
                         dict(kernel=1, copy=1, copy_bytes=1028, barrier=1))
        for corrupt in ('flow', 'id', 'clock', 'unknown', 'bytes', 'host'):
            rows = copy.deepcopy(events)
            if corrupt == 'flow':
                rows.pop()
            elif corrupt == 'id':
                rows[3]['args']['id'] = rows[1]['args']['id']
            elif corrupt == 'clock':
                rows[1]['dur'] = float('nan')
            elif corrupt == 'unknown':
                rows[1]['name'] = 'unknown'
            elif corrupt == 'bytes':
                rows[3]['name'] = 'zeCommandListAppendMemoryCopy'
            elif corrupt == 'host':
                rows.pop(0)
            with self.subTest(corrupt=corrupt), self.assertRaises(RuntimeError):
                probe.device_census(rows, 71, 0, 100000)

    def link_timeline(self):
        return [dict(ph='X', cat='cpu_op', name='zeCommandListAppendLaunchKernelWithArguments',
                     pid=71, tid=72, ts=10, dur=2, id=0),
                dict(ph='s', cat='Flow_H2D_1_71', pid=71, tid=72, ts=10, id=1),
                dict(ph='t', cat='Flow_H2D_1_71', pid=90, tid=91, ts=20, id=1),
                dict(ph='X', cat='gpu_op', name='main::kernel', pid=90, tid=91, ts=20, dur=3,
                     args={'id': '1'}),
                dict(ph='t', cat='Flow_D2H_1_71', pid=71, tid=72, ts=25, id=1)]

    def test_exact_collector_flow_links_do_not_claim_an_independent_UR_join(self):
        result = probe.trace_links(self.link_timeline(), 71)
        self.assertEqual(result['linked_device_operations'], 1)
        self.assertTrue(result['exact_collector_host_device_links'])
        self.assertFalse(result['independent_UR_API_join_qualified'])

    def test_same_counts_do_not_hide_wrong_flow_source_or_sink(self):
        for corrupt in ('timestamp', 'source-thread', 'sink-queue', 'sink-time', 'append-kind',
                        'duplicate-source', 'missing-completion'):
            events = self.link_timeline()
            if corrupt == 'timestamp':
                events[1]['ts'] = 11
            elif corrupt == 'source-thread':
                events[1]['tid'] = 73
            elif corrupt == 'sink-queue':
                events[2]['pid'] = 92
            elif corrupt == 'sink-time':
                events[2]['ts'] = 21
            elif corrupt == 'append-kind':
                events[0]['name'] = 'zeCommandListAppendMemoryCopy'
            elif corrupt == 'duplicate-source':
                events.append(copy.deepcopy(events[0]))
            else:
                events.pop()
            with self.subTest(corrupt=corrupt), self.assertRaises(RuntimeError):
                probe.trace_links(events, 71)


if __name__ == '__main__':
    unittest.main()
