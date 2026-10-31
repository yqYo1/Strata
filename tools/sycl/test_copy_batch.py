#!/usr/bin/env python3
"""Reject incomplete or slower cohorts before any production/model advance."""
import copy
import unittest
import refresh_transfer as probe


def cohort():
    cases = []
    for process in range(3):
        samples = []
        for size in (1971200, 2662400):
            for pattern in ('contiguous', 'runs4', 'gapped'):
                for loops in (1, 64):
                    for repetition in range(-1, 5):
                        for group in (1, 2, 4, 8):
                            calls = dict(contiguous=160//group, runs4=160//min(group,4), gapped=160)[pattern]
                            ratio = {1:1, 2:.85, 4:.8, 8:.9}[group] if pattern != 'gapped' else 1
                            samples.append(dict(group=str(group), bytes=str(size), pattern=pattern, loops=str(loops),
                                repetition=str(repetition), experts='160', completed='160', uploaded_experts='160',
                                readers='320', capacity='128', copy_calls=str(calls), copy_events=str(calls),
                                requested_upload_bytes=str(160*size), cancelled='0', compat='0',
                                admission_rejections='0', cpu_reference='0', verifier_negative_controls='0',
                                source_usm='unknown', gpu_verify='PASS', wall_s=str(.04*ratio),
                                device_span_s='.03'))
        cases.append(dict(passed=True, samples=samples,source_checks=[dict(bytes=str(size),pattern=pattern,full='PASS')
                          for size in (1971200,2662400) for pattern in ('contiguous','runs4','gapped')], case=dict(argv=['--compare', str(process)]),
                          controls=[dict(samples='1000', p99_ns='200')] +
                          [dict(repetition=str(i), wall_s='.0001') for i in range(5)]))
    return cases


class Decision(unittest.TestCase):
    def test_balanced_complete_cohort(self):
        r = probe.batch_decision(cohort())
        self.assertTrue(r['mechanism_screen_passed'])
        self.assertEqual(len(r['shapes']), 108)
        self.assertEqual(r['selected_group'], 4)
        self.assertFalse(r['inference_adopted'])

    def test_one_slow_process_cannot_hide_in_aggregate(self):
        c = cohort()
        for s in c[1]['samples']:
            if s['group'] != '1' and s['pattern'] == 'runs4': s['wall_s'] = '.05'
        self.assertFalse(probe.batch_decision(c)['mechanism_screen_passed'])

    def test_gapped_regression_is_not_paid_for_by_contiguous_gain(self):
        c = cohort()
        for s in c[2]['samples']:
            if s['group'] != '1' and s['pattern'] == 'gapped': s['wall_s'] = '.042'
        self.assertFalse(probe.batch_decision(c)['mechanism_screen_passed'])

    def test_ineligible_clocks_and_controls(self):
        for mutation in ('batch', 'empty', 'timer'):
            c = cohort()
            if mutation == 'batch': c[0]['samples'][4]['device_span_s'] = '.0001'
            elif mutation == 'empty':
                for control in c[0]['controls'][1:]: control['wall_s'] = '.1'
            else: c[0]['controls'][0]['p99_ns'] = '10001'
            with self.subTest(mutation=mutation):
                self.assertFalse(probe.batch_decision(c)['timing_eligible'])

    def test_copy_overfetch_or_missing_readers_is_rejected(self):
        for field, value in [('requested_upload_bytes', '999999999'), ('copy_calls', '159'),
                             ('readers', '160'), ('source_usm', 'host'), ('gpu_verify', 'FAIL')]:
            c = cohort(); c[0]['samples'][0][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError): probe.batch_decision(c)

    def test_first_use_cannot_replace_missing_subsequent_pair(self):
        c = cohort(); c[0]['samples'][-1] = copy.deepcopy(c[0]['samples'][1])
        with self.assertRaises(RuntimeError): probe.batch_decision(c)

    def test_one_bad_group_is_excluded_from_selection(self):
        c = cohort()
        for s in c[0]['samples']:
            if s['group'] == '4' and s['pattern'] == 'runs4': s['wall_s'] = '.05'
        r = probe.batch_decision(c)
        self.assertTrue(r['mechanism_screen_passed'])
        self.assertEqual(r['selected_group'], 2)
        self.assertFalse(next(g for g in r['groups'] if g['group'] == 4)['performance_passed'])

    def test_missing_final_immutable_source_check_is_rejected(self):
        c = cohort(); c[0]['source_checks'].pop()
        with self.assertRaises(RuntimeError): probe.batch_decision(c)

    def test_missing_or_duplicate_process_is_rejected(self):
        for c in (cohort()[:2], [cohort()[0], cohort()[0], cohort()[2]]):
            with self.assertRaises(RuntimeError): probe.batch_decision(c)


if __name__ == '__main__':
    unittest.main()
