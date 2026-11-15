"""Host controls for production-adapter collection; use the shared host runner."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
import profile_compat as profile


class AdapterWindow(unittest.TestCase):
    pid, device = 1234, 9000
    bounds = dict(capture_begin=1_000_000, begin=2_000_000, end=4_000_000, capture_end=5_000_000)

    def events(self, mode='active'):
        result = [dict(ph='M', name='process_name', pid=self.device,
                       args=dict(name='DEVICE<host>[Intel Arc B570] 0:5:0:0 #0'))]
        operations = []
        copy_name = 'zeCommandListAppendMemoryCopy(D2H)[1164]'
        if mode == 'active': operations.append(('zeCommandListAppendMemoryCopy(H2D)[1164]', 1500))
        for start in (2200, 2500, 2800, 4400):
            operations += [('StrataCompatWindowProduce', start),
                           ('StrataCompatWindowConsume', start + 20), (copy_name, start + 40)]
        for work_id, (name, start) in enumerate(operations, 1):
            result += [dict(ph='s', cat='Flow_H2D_0', pid=self.pid, id=work_id, ts=1100),
                dict(ph='X', cat='gpu_op', pid=self.device, name=name, ts=start, dur=10, args=dict(id=work_id)),
                dict(ph='X', cat='cpu_op', pid=self.pid, name='submit', ts=start, dur=7)]
        return result

    def parse(self, events, mode='active'):
        return profile.trace_evidence(events, self.pid, self.bounds, mode)

    def test_full_active_and_paused_construction_have_distinct_upload_counts(self):
        for mode, count in (('active', 5), ('paused-construction', 4)):
            with self.subTest(mode=mode):
                result = self.parse(self.events(mode), mode)
                self.assertEqual(result['whole_copies'], count)
                self.assertEqual(result['inner_kernels'], 6)
                self.assertFalse(result['physical_saturation_qualified'])
                self.assertFalse(result['lossless_pending_boundary_qualified'])

    def test_missing_post_copy_cannot_be_hidden_by_an_extra_initial_copy(self):
        events = self.events()
        post_copy = next(e for e in events if e.get('cat') == 'gpu_op' and
                         e['ts'] == 4440 and 'MemoryCopy' in e['name'])
        post_copy['ts'] = 1800
        with self.assertRaises(RuntimeError): self.parse(events)

    def test_replay_counts_alone_do_not_establish_both_kernel_names(self):
        events = self.events()
        next(e for e in events if e.get('name') == 'StrataCompatWindowProduce' and
             e.get('ts') == 2200)['name'] = 'StrataCompatWindowConsume'
        with self.assertRaises(RuntimeError): self.parse(events)

    def test_missing_recorded_replays_after_paused_construction_are_refused(self):
        events = [e for e in self.events('paused-construction') if e.get('cat') != 'gpu_op']
        with self.assertRaises(ValueError): self.parse(events, 'paused-construction')

    def test_crossing_and_ambiguous_instances_are_refused(self):
        for mutation in ('crossing', 'duplicate'):
            events = copy.deepcopy(self.events())
            kernel = next(e for e in events if e.get('name') == 'StrataCompatWindowProduce' and
                          e.get('ts') == 2200)
            if mutation == 'crossing': kernel['ts'] = 1999
            else: events.append(copy.deepcopy(kernel))
            with self.subTest(mutation=mutation), self.assertRaises((RuntimeError, ValueError)):
                self.parse(events)


if __name__ == '__main__':
    unittest.main()
