#!/usr/bin/env python3
"""Rejection tests for mechanism advancement, not an LLM performance oracle."""
import unittest
import refresh_transfer as probe


def cohort():
    cases = [dict(passed=True)]
    for process in range(3):
        samples = []
        for size in (1971200, 2662400):
            for loops in (16, 256):
                for repetition in range(-1, 5):
                    for arm in ('serial', 'pipeline'):
                        ratio = 0.8 if loops == 256 and arm == 'pipeline' else 1
                        samples.append(dict(bytes=str(size), loops=str(loops), repetition=str(repetition),
                            arm=arm, wall_s=str(.025 * ratio), device_span_s='.02', copy_s='.01',
                            overlap_s='.003' if arm == 'pipeline' else '0', experts='32', completed='32',
                            requested_uploads='32', observed_uploads='32', readers='64', cancelled='0'))
        cases.append(dict(passed=True, samples=samples, case=dict(argv=['--compare', str(process)]),
                          controls=[dict(samples='1000', p99_ns='200')] +
                          [dict(repetition=str(i),wall_s='.0001') for i in range(5)]))
    return cases


class Decision(unittest.TestCase):
    def test_balanced_eligible_cohort(self):
        r = probe.pipeline_decision(cohort())
        self.assertTrue(r['mechanism_screen_passed'])
        self.assertEqual(len(r['shapes']),12)
        self.assertFalse(r['inference_adopted'])

    def test_one_slow_process_cannot_hide_in_aggregate(self):
        c = cohort()
        for s in c[2]['samples']:
            if s['arm'] == 'pipeline' and s['loops'] == '256': s['wall_s'] = '.03'
        self.assertFalse(probe.pipeline_decision(c)['mechanism_screen_passed'])

    def test_short_batches_and_expensive_controls_are_ineligible(self):
        for mutation in ('batch','empty','timer'):
            c = cohort()
            if mutation == 'batch': c[1]['samples'][4]['device_span_s'] = '.0001'
            elif mutation == 'empty':
                for control in c[1]['controls'][1:]: control['wall_s'] = '.1'
            else: c[1]['controls'][0]['p99_ns'] = '10001'
            with self.subTest(mutation=mutation):
                self.assertFalse(probe.pipeline_decision(c)['timing_eligible'])

    def test_no_observed_overlap_cannot_advance(self):
        c = cohort()
        for s in c[1]['samples']: s['overlap_s'] = '0'
        self.assertFalse(probe.pipeline_decision(c)['mechanism_screen_passed'])

    def test_missing_process_or_reader_is_rejected(self):
        c = cohort()
        with self.assertRaises(RuntimeError): probe.pipeline_decision(c[:1])
        c[1]['samples'][0]['readers'] = '32'
        with self.assertRaises(RuntimeError): probe.pipeline_decision(c)


if __name__ == '__main__':
    unittest.main()
