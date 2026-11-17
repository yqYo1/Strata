"""Refusal controls for incomplete, rerouted and ineligible QSA evidence."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import qsa_prefix_calibration as probe


def fixture(process=0):
    lines=[]
    for cells,queries,cap,pattern in probe.CASES:
        for r in range(6):
            for order in range(2):
                arm=(process+r+order)%2
                lines.append(f'sample process={process} cells={cells} queries={queries} cap={cap} pattern={pattern} round={r} order={order} variant={arm} implementation={"cached_subgroup" if arm else "ordered_exp"} milliseconds={11 if arm else 12} compared={queries*24*256} exact_bits=PASS guards=PASS status=PASS')
        lines.append(f'case cells={cells} queries={queries} cap={cap} pattern={pattern} samples=12 full_bits=PASS guards=PASS inputs=PASS status=PASS')
    lines.append('fixture_status=PASS model_quality=UNRUN model_PP=UNRUN physical262144_model=UNRUN faults=EXTERNAL_REQUIRED')
    return '\n'.join(lines)+'\n'


class Evidence(unittest.TestCase):
    def test_complete(self):
        for i in range(3):
            points=probe.parse(fixture(i),i)
            self.assertEqual([len(p['samples']) for p in points],[12]*3)
    def test_missing_or_duplicate_sample(self):
        text=fixture();line=text.splitlines()[0]
        for corrupt in (text.replace(line+'\n','',1),text+line+'\n'):
            with self.assertRaises(RuntimeError):probe.parse(corrupt,0)
    def test_wrong_arm_or_order(self):
        for a,b in [('implementation=ordered_exp','implementation=cached_subgroup'),('order=0','order=1'),('process=0','process=2')]:
            with self.assertRaises(RuntimeError):probe.parse(fixture().replace(a,b,1),0)
    def test_wrong_output_or_incomplete_case(self):
        for a,b in [('exact_bits=PASS','exact_bits=FAIL'),('guards=PASS','guards=FAIL'),('inputs=PASS','inputs=FAIL'),('compared=50331648','compared=1')]:
            with self.assertRaises(RuntimeError):probe.parse(fixture().replace(a,b,1),0)
    def test_short_or_invalid_timing(self):
        for t in ('0.01','nan','inf','-1'):
            with self.assertRaises(RuntimeError):probe.parse(fixture().replace('milliseconds=12','milliseconds='+t,1),0)
    def test_new_case_cannot_enter(self):
        with self.assertRaises(RuntimeError):probe.parse(fixture().replace('cap=2051','cap=2048'),0)
    def test_one_slower_process_blocks_advancement(self):
        processes=[dict(index=i,passed=True,ownership_closed=True,fresh_faults=[],points=probe.parse(fixture(i),i)) for i in range(3)]
        self.assertTrue(probe.decide(processes)['eligible_for_new_128K_plan'])
        altered=copy.deepcopy(processes);altered[1]['points'][1]['candidate_over_reference']=1.01
        self.assertFalse(probe.decide(altered)['eligible_for_new_128K_plan'])
        with self.assertRaises(RuntimeError):probe.decide(processes[:2])


if __name__=='__main__':unittest.main()
