"""Host-only order, payload and GPU interval reader controls. Repository LICENSE."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent/'sycl'))
import fullfit_attention_overlap as experiment


def fixture():
    lines = ['fullfit_attention_overlap source=host_usm bytes=1009254400 alloc_ms=10.0 fill_ms=20.0 type=PASS',
             'fullfit_attention_overlap copy_api=sycl ordinal=0 index=0 mode=2 flags=2 priority=0 domain=PASS',
             'fullfit_attention_overlap clock_pair_max_s=0.00000007']
    for round_n in range(experiment.FULL_ROUNDS):
        for place in range(len(experiment.ARMS)):
            variant = experiment.ARMS[(round_n + place + 2) % len(experiment.ARMS)]
            copies = 512 if variant != 'compute_only' else 0
            attention = int(variant != 'copy_only')
            lines.append(f'fullfit_attention_overlap process=1 round={round_n} order={place} arm={variant} '
                f'copies={copies} requested_bytes={copies*1971200} attention_calls={attention} '
                'milliseconds=50.0 raw_start=0 raw_end=0 payload=PASS guards=PASS output=PASS')
    lines.append('case cells=65536 queries=8192 cap=2051 pattern=gather samples=20 '
                 'full_bits=PASS guards=PASS inputs=PASS status=PASS')
    return '\n'.join(lines)


class MeasurementControls(unittest.TestCase):
    def test_compiler_outputs_and_logs_are_disjoint(self):
        self.assertEqual(experiment.COMPILE_LOGS.parent,experiment.ROOT)
        for command in experiment.compile_commands():
            output = Path(command[-1])
            self.assertEqual(command[-2],'-o')
            self.assertEqual(output.parent,experiment.ROOT/'build')
            self.assertNotIn(experiment.COMPILE_LOGS,output.parents)

    def test_order_payload_and_invalid_clean_samples(self):
        good = fixture()
        self.assertEqual(len(experiment.samples(good,'clean-3')['samples']),20)
        for bad in (
            good.replace('order=0','order=3',1),
            good.replace('copies=512','copies=511',1),
            good.replace('milliseconds=50.0','milliseconds=9',1),
            good.replace('raw_end=0','raw_end=20000',1),
            good.replace(' guards=PASS',' guards=PASS guards=PASS',1),
            good.replace('source=host_usm','source=ordinary',1),
            good.replace('type=PASS','type=FAIL',1),
            good.replace('alloc_ms=10.0','alloc_ms=nan',1),
            good.replace('bytes=1009254400','bytes=1009254401',1),
            good.replace('copy_api=sycl','copy_api=native',1),
            good.replace('ordinal=0','ordinal=1',1),
            good.replace('flags=2','flags=6',1),
            good.replace('domain=PASS','domain=FAIL',1),
        ):
            with self.assertRaises(RuntimeError):
                experiment.samples(bad, 'clean-3')

    def test_device_count_and_overlap(self):
        slab = 'zeCommandListAppendMemoryCopy(H2D)[1971200]'
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
                 [dict(event,name=slab.replace('H2D','M2D')),pack,qsa],
                 [event,pack,dict(qsa,name='foreign_kernel')])
        for entries in cases:
            with self.assertRaises(RuntimeError):
                experiment.timeline(dict(traceEvents=entries), [point], (100000,150000))

    def test_copy_api_pairs_keep_waits_fixed_and_first_cycle_separate(self):
        def row(mode, index, times):
            timing = dict(samples=[dict(arm=arm,round=n,milliseconds=elapsed+1000*(n==0))
                for n in range(experiment.FULL_ROUNDS) for arm,elapsed in zip(experiment.ARMS,times)],
                queue=dict(api=experiment.CASES[mode][1]))
            return dict(mode=mode,pid=index,timing=timing,
                timeline=[dict(arm=arm,round=n,overlap_ns=10000 if arm=='deferred_wait' else 0,
                    copy_union_ns=20000 if arm!='compute_only' else 0,
                    qsa_union_ns=30000 if arm!='copy_only' else 0)
                          for n in range(experiment.FULL_ROUNDS) for arm in experiment.ARMS])
        timings = (70,100,185,120)
        native = (70,99,185,110)
        outcomes = [row('trace-'+api,10+i,timings) for i,api in enumerate(experiment.COPY_APIS)]
        outcomes += [row(f'clean-{i}',i,native if api=='native' else timings)
                     for i,api in enumerate(experiment.CLEAN_ORDER)]
        decision = experiment.assessment(outcomes)
        self.assertEqual(decision['paired_clean_indices'],
                         [dict(sycl=0,native=1),dict(sycl=3,native=2),dict(sycl=4,native=5)])
        self.assertTrue(decision['eligible_same_ordinal_native_gain'])
        self.assertEqual(decision['warmed_ratios']['deferred_wait']['paired'],[110/120]*3)
        self.assertEqual(decision['warmed_ratios']['compute_only']['paired'],[1.0]*3)
        self.assertEqual(decision['clean_first_cycle_ms'][0]['copy_only'],1100)
        self.assertEqual(decision['clean_process_warmed_medians_ms'][0]['copy_only'],100)
        outcomes[4] = row('clean-2',2,(70,99,185,121))
        self.assertFalse(experiment.assessment(outcomes)['eligible_same_ordinal_native_gain'])
        outcomes[4]['pid'] = outcomes[3]['pid']
        with self.assertRaises(RuntimeError):
            experiment.assessment(outcomes)


if __name__ == '__main__':
    unittest.main()
