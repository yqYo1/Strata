"""Synthetic join, clock-margin and selection checks."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import fullfit_attention_attribution as analysis


class CorrelationTests(unittest.TestCase):
    def test_exact_flow_identity_and_late_append(self):
        events = []
        for ident, kind, api_ts, gpu_ts, gpu_duration in (
            ('1','copy',11000,12000,6000),
            ('2','kernel',21000,22000,2000),
            ('3','kernel',23000,24000,2000),
        ):
            name = 'zeCommandListAppendMemoryCopy' if kind=='copy' else analysis.LAUNCH[0]
            device_name = name+'(M2D)[1971200]' if kind=='copy' else 'prefix_batch_impl'
            category = f'Flow_H2D_{ident}_4'
            events += [
                dict(cat='cpu_op',ph='X',tid=5,pid=4,ts=api_ts,dur=500,name=name),
                dict(cat='gpu_op',ph='X',tid=50,pid=40,ts=gpu_ts,dur=gpu_duration,
                    name=device_name,args=dict(id=ident)),
                dict(cat=category,ph='s',tid=5,pid=4,ts=api_ts,id=ident),
                dict(cat=category,ph='t',tid=50,pid=40,ts=gpu_ts,id=ident),
            ]
        points = [dict(arm='deferred_wait',round=0,raw_start=10_000_000,raw_end=30_000_000,
            copies=1,attention_calls=1)]
        device = [dict(arm='deferred_wait',round=0,overlap_ns=0)]
        value, = analysis.correlate(events,points,device,4)
        self.assertEqual(value['classification'],'native_append_begins_after_copy')
        self.assertEqual(value['first_qsa_append_enter_minus_copy_end_ns'],3_000_000)
        with self.assertRaises(RuntimeError):
            analysis.correlate(events[:-1],points,device,4)
        with self.assertRaises(RuntimeError):
            analysis.correlate(events+[events[-2]],points,device,4)
        with self.assertRaises(RuntimeError):
            analysis.correlate(events[:-1]+[dict(events[-1],pid=41)],points,device,4)

    def test_distinct_signed_relations(self):
        cases = [
            ((5,6,3,0),'native_append_begins_after_copy'),
            ((1,2,5,0),'native_append_returns_before_copy_end'),
            ((1,5,3,0),'native_append_spans_copy_tail'),
            ((2,3,3,0),'near_clock_margin'),
            ((2,3,3,2),'device_overlap_observed'),
        ]
        for values, expected in cases:
            with self.subTest(values=values):
                self.assertEqual(analysis.classify(*(x*1_000_000 for x in values)),expected)

    def test_warm_decision_requires_serial_sign_control(self):
        windows = [dict(arm=arm,round=n,classification=(
            'native_append_begins_after_copy' if arm=='serial_separate'
            else 'native_append_returns_before_copy_end'))
            for arm in analysis.ARMS for n in range(5)]
        self.assertEqual(analysis.decision(windows)['next'],'inspect_native_queue_and_command_list_scheduling')
        windows[2]['classification'] = 'near_clock_margin'
        self.assertEqual(analysis.decision(windows)['next'],'unresolved')


if __name__=='__main__':
    unittest.main()
