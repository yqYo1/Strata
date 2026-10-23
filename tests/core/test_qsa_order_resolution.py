"""Refusal coverage for the mandatory downstream timing gate; repository LICENSE."""
import copy
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'tools' / 'sycl'))
import qualify_qsa_order_resolution as gate


def completed():
    return dict(stage='closed', passed=True, complete=True, ownership_closed=True,
                active=False, postchecks_complete=True, boot_unchanged=True, fresh_faults=[],
                sample_columns=['layer', 'round', 'order', 'arm', 'elapsed_ns', 'timer_pair_ns'],
                sample_calls=32, samples=[[]] + [
                    [[l, r, o, (p+r+o) % 3, 100000000, 20]
                     for l in range(12) for r in range(5) for o in range(3)] for p in range(3)])


class ResolutionTests(unittest.TestCase):
    def test_all_positive(self):
        self.assertEqual(gate.positive_resolution(completed()),
                         dict(gate=gate.GATE, passed=True, samples=540,
                              minimum_timer_pair_ns=20, maximum_timer_pair_ns=20))

    def test_zero_negative_boolean_and_excessive_overhead(self):
        for value in (0, -1, True, 100001):
            with self.subTest(value=value):
                record = completed(); record['samples'][3][-1][-1] = value
                with self.assertRaises(RuntimeError): gate.positive_resolution(record)

    def test_missing_or_duplicated_sample(self):
        for duplicate in (False, True):
            record = completed()
            if duplicate: record['samples'][2][-1] = copy.copy(record['samples'][2][0])
            else: record['samples'][2].pop()
            with self.assertRaises(RuntimeError): gate.positive_resolution(record)

    def test_rotated_order(self):
        record = completed(); record['samples'][1][0][3] = 1
        with self.assertRaises(RuntimeError): gate.positive_resolution(record)

    def test_unclosed_failed_or_faulted_cohort(self):
        for key, value in [('active', True), ('passed', False), ('complete', False),
                           ('ownership_closed', False), ('postchecks_complete', False),
                           ('boot_unchanged', False), ('fresh_faults', ['fault'])]:
            record = completed(); record[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError): gate.positive_resolution(record)


if __name__ == '__main__':
    unittest.main()
