#!/usr/bin/env python3
"""Host-only controls for cost eligibility,independent pairing and provenance.

Main-authored,repository LICENSE. No device/runtime initialization or native exec.
"""
import unittest
from profile_observer_cost import ORDER, summarize

HASHES = ('a' * 64, 'b' * 64)


def cohort():
    result = []
    for index, arm in enumerate(ORDER):
        samples = []
        for run in range(2):
            samples.append(dict(run=run, request_sha256=HASHES[run], actual_prompt_tokens=117489,
                actual_read_tokens=117489, actual_reused_tokens=0, generated_tokens=24,
                thinking_requested=True, prompt_ms=(200000 if run == 0 else 100000) * (1.01 if arm == 'on' else 1)))
        result.append(dict(index=index, arm=arm, passed=True, complete=True, ownership_closed=True, requests=samples))
    return result


class CostControls(unittest.TestCase):
    def test_disjoint_pairs_and_first_use(self):
        rows = cohort(); rows[0]['requests'][0]['prompt_ms'] = 9999999
        found = summarize(rows, HASHES)
        self.assertTrue(found['observer_cost_qualified'])
        self.assertAlmostEqual(found['geometric_mean'], 1.01)
        self.assertEqual(found['time_ratios_on_over_off'], [1.01] * 3)
        self.assertFalse(found['first_use_pooled'])

    def test_geometric_mean_gate(self):
        rows = cohort()
        for row in rows:
            if row['arm'] == 'on': row['requests'][1]['prompt_ms'] = 103000
        found = summarize(rows, HASHES)
        self.assertTrue(found['every_pair_gate']); self.assertFalse(found['observer_cost_qualified'])

    def test_one_slow_pair_cannot_hide_in_mean(self):
        rows = cohort()
        for index, value in ((1, 106000), (2, 98000), (5, 98000)):
            rows[index]['requests'][1]['prompt_ms'] = value
        found = summarize(rows, HASHES)
        self.assertTrue(found['geometric_mean_gate']); self.assertFalse(found['every_pair_gate'])
        self.assertFalse(found['observer_cost_qualified'])

    def test_incomplete_processes(self):
        for field in ('passed', 'complete', 'ownership_closed'):
            for value in (False, 'false'):
                with self.subTest(field=field, value=value):
                    rows = cohort(); rows[2][field] = value
                    with self.assertRaises(RuntimeError): summarize(rows, HASHES)

    def test_order_and_count(self):
        for rows in (cohort()[:-1], list(reversed(cohort()))):
            with self.assertRaises(RuntimeError): summarize(rows, HASHES)

    def test_invalid_clocks(self):
        for value in (True, 0, -1, 9999, float('nan'), float('inf')):
            with self.subTest(value=value):
                rows = cohort(); rows[0]['requests'][1]['prompt_ms'] = value
                with self.assertRaises(RuntimeError): summarize(rows, HASHES)

    def test_request_contract(self):
        sample = cohort()[0]['requests'][1]
        wrong = dict(actual_read_tokens=117488, actual_reused_tokens=1, generated_tokens=23,
            thinking_requested=False, request_sha256='c' * 64, run=0)
        for key, value in wrong.items():
            rows = cohort(); rows[0]['requests'][1] = dict(sample, **{key: value})
            with self.subTest(key=key):
                with self.assertRaises(RuntimeError): summarize(rows, HASHES)

    def test_missing_subsequent_request(self):
        rows = cohort(); rows[2]['requests'] = rows[2]['requests'][:1]
        with self.assertRaises(RuntimeError): summarize(rows, HASHES)


if __name__ == '__main__':
    unittest.main()
