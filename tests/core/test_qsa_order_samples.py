"""Main-authored cohort admission/conservation tests; repository LICENSE.

Run: python3 -m unittest discover -s tests/core -p test_qsa_order_samples.py -v
Only synthetic text is parsed. No model files or GPU/runtime are opened.
"""
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import measure_qsa_order_replay as measure


class CohortAdmission(unittest.TestCase):
    tail=dict(first=116976,cells=117488)

    @staticmethod
    def cases():
        return [f'case layer={l} first=116976 cells=117488 queries=512 cap=2051 cpu_order_ns=123 '
            'output_bits=PASS perturbation=PASS guards=PASS immutable=PASS' for l in range(12)]

    @staticmethod
    def rows():
        return [f'sample layer={l} process=0 arm={(r+o)%3} round={r} order={o} calls=32 '
            f'elapsed_ns={[320000000,304000000,300000000][(r+o)%3]} timer_pair_ns=25 eligible=1 checks=PASS'
            for l in range(12) for r in range(5) for o in range(3)]

    def text(self,rows,cases=None):
        return '\n'.join((self.cases() if cases is None else cases)+rows+[
            'fixture=PASS model=UNRUN physical_counters=UNMEASURED cpu_order=EXCLUDED'])

    def test_individual_samples_and_gate(self):
        samples,summary=measure.output(self.text(self.rows()),1,self.tail)
        self.assertEqual(len(samples),180);self.assertEqual(summary['passing_methods'],[1,2])
        self.assertAlmostEqual(summary['ratios'][1],.95);self.assertAlmostEqual(summary['ratios'][2],.9375)

    def test_missing_duplicate_or_rotated_sample_is_rejected(self):
        rows=self.rows()
        for bad in (rows[:-1],rows+[rows[0]],rows[:-1]+[rows[0]],
            [rows[0].replace('arm=0','arm=2')]+rows[1:]):
            with self.assertRaises(RuntimeError):measure.output(self.text(bad),1,self.tail)

    def test_declared_eligible_does_not_override_duration(self):
        rows=self.rows();rows[0]=rows[0].replace('elapsed_ns=320000000','elapsed_ns=99999999')
        with self.assertRaisesRegex(RuntimeError,'ineligible timing'):measure.output(self.text(rows),1,self.tail)

    def test_duplicate_field_or_negative_timer_is_rejected(self):
        rows=self.rows()
        for changed in (rows[0]+' calls=32',rows[0].replace('timer_pair_ns=25','timer_pair_ns=-1')):
            with self.assertRaises(RuntimeError):measure.output(self.text([changed]+rows[1:]),1,self.tail)

    def test_wrong_layer_extent_or_missing_success_is_rejected(self):
        cases=self.cases();cases[-1]=cases[-1].replace('first=116976','first=116975')
        with self.assertRaisesRegex(RuntimeError,'replay checks'):measure.output(self.text(self.rows(),cases),1,self.tail)
        with self.assertRaises(RuntimeError):measure.output(self.text(self.rows()).replace('fixture=PASS','fixture=FAIL'),1,self.tail)

    def test_validation_cannot_supply_speed_samples(self):
        self.assertEqual(measure.output(self.text([]),0,self.tail),([],None))
        with self.assertRaises(RuntimeError):measure.output(self.text(self.rows()),0,self.tail)

    def test_slow_method_does_not_pass_from_successful_execution(self):
        rows=[r.replace('elapsed_ns=304000000','elapsed_ns=400000000') for r in self.rows()]
        _,summary=measure.output(self.text(rows),1,self.tail);self.assertEqual(summary['passing_methods'],[2])


if __name__=='__main__':unittest.main()
