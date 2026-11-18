"""Host-only order, payload and GPU interval reader controls. Repository LICENSE."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent/'sycl'))
import fullfit_attention_overlap as experiment


def fixture():
    lines = ['fullfit_attention_overlap clock_pair_max_s=0.00000007']
    for round_n in range(5):
        for place in range(5):
            variant = experiment.ARMS[(round_n + place + 2) % 5]
            copies = 512 if variant != 'compute_only' else 0
            attention = int(variant != 'copy_only')
            lines.append(f'fullfit_attention_overlap process=1 round={round_n} order={place} arm={variant} '
                f'copies={copies} requested_bytes={copies*1971200} attention_calls={attention} '
                'milliseconds=50.0 raw_start=0 raw_end=0 payload=PASS guards=PASS output=PASS')
    lines.append('case cells=65536 queries=8192 cap=2051 pattern=gather samples=25 '
                 'full_bits=PASS guards=PASS inputs=PASS status=PASS')
    return '\n'.join(lines)


class MeasurementControls(unittest.TestCase):
    def test_order_payload_and_invalid_clean_samples(self):
        good = fixture()
        self.assertEqual(len(experiment.samples(good,'1')['samples']),25)
        for bad in (
            good.replace('order=0','order=4',1),
            good.replace('copies=512','copies=511',1),
            good.replace('milliseconds=50.0','milliseconds=9',1),
            good.replace('raw_end=0','raw_end=20000',1),
            good.replace(' guards=PASS',' guards=PASS guards=PASS',1),
        ):
            with self.assertRaises(RuntimeError):
                experiment.samples(bad, '1')

    def test_device_count_and_overlap(self):
        slab = 'zeCommandListAppendMemoryCopy(M2D)[1971200]'
        event = dict(cat='gpu_op', ph='X', ts=111, dur=7, name=slab)
        pack = dict(cat='gpu_op', ph='X', ts=114, dur=2, name='prefix_batch_impl pack')
        qsa = dict(cat='gpu_op', ph='X', ts=117, dur=15, name='prefix_batch_impl attention')
        point = dict(arm='deferred_wait', round=0, copies=1, requested_bytes=1971200, attention_calls=1,
                     raw_start=110000, raw_end=140000)
        trace = dict(traceEvents=[event, pack, qsa])
        result, = experiment.timeline(trace, [point], (100000, 150000))
        self.assertEqual((result['copy_union_ns'], result['qsa_union_ns'], result['overlap_ns']),
                         (7000, 17000, 3000))
        cases = ([dict(event, dur=30),pack,qsa], [event,pack],
                 [dict(event,name=slab.replace('M2D','D2D')),pack,qsa],
                 [event,pack,dict(qsa,name='foreign_kernel')])
        for entries in cases:
            with self.assertRaises(RuntimeError):
                experiment.timeline(dict(traceEvents=entries), [point], (100000,150000))

    def test_independent_decisions(self):
        def row(mode, index, times):
            timing = dict(samples=[dict(arm=arm,milliseconds=elapsed)
                for arm,elapsed in zip(experiment.ARMS,times)])
            return dict(mode=mode,pid=index,timing=timing,
                timeline=[dict(arm=arm,overlap_ns=10000 if arm=='deferred_wait' else 0)
                          for arm in experiment.ARMS])
        timings = (70,100,180,185,120)
        outcomes = [row('trace',10,timings)] + [row(str(i),i,timings) for i in range(3)]
        decision = experiment.assessment(outcomes)
        self.assertFalse(decision['ratios']['queue_choice']['all_three_faster'])
        self.assertTrue(decision['ratios']['wait_placement']['all_three_faster'])
        self.assertTrue(decision['eligible_for_separate_reader_ownership_test'])
        outcomes[2] = row('1',1,(70,100,180,185,186))
        self.assertFalse(experiment.assessment(outcomes)['eligible_for_separate_reader_ownership_test'])


if __name__ == '__main__':
    unittest.main()
