"""Admission and timing eligibility checks; no GPU/model/build execution."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/sycl'))
import cpu_roofline as calibration


class TimingEligibility(unittest.TestCase):
    def row(self):
        return dict(pair=[18,20],workers=5,closed=True,exit_code=0,rows=[
            dict(kind='sample',mode='hot',nt=3,attempt=0,rep=i,iterations=10,
                 requested_bytes=1000000000,wall_s=1.2,cpu_s=6.) for i in range(5)])

    def test_individual_samples_and_rate(self):
        row=self.row();cell=calibration.summarize([row])[0]
        self.assertEqual(cell['samples'],row['rows'])
        self.assertAlmostEqual(cell['requested_GB_s_median'],1/1.2)

    def test_failure_or_missing_repetition_is_ineligible(self):
        for field,value in [('closed',False),('exit_code',1)]:
            row=self.row();row[field]=value
            with self.assertRaises(RuntimeError):calibration.summarize([row])
        row=self.row();row['rows'].pop()
        with self.assertRaises(RuntimeError):calibration.summarize([row])

    def test_under_floor_not_silently_accepted(self):
        row=self.row();row['rows'][2]['wall_s']=.999
        with self.assertRaises(RuntimeError):calibration.summarize([row])
        extended=copy.deepcopy(row['rows'])
        for r in extended:r.update(attempt=1,wall_s=2.4,iterations=20)
        row['rows']+=extended
        self.assertEqual(len(calibration.summarize([row])[0]['samples']),10)

    def test_duplicate_repetition_is_ineligible(self):
        row=self.row();row['rows'][-1]['rep']=3
        with self.assertRaises(RuntimeError):calibration.summarize([row])


if __name__=='__main__':unittest.main()
