"""Host-only admission controls for planner evidence; repository LICENSE."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kv_planner_experiment as controller

MARKER = ('planner_benchmark=PASS samples=18 timing=sum_of_isolated_graph_launch_and_wait '
          'reset_and_oracle=excluded physical_bus_bytes=unknown')


def rows(arm='scalar', factor=1.):
    result = []
    for nq in (2, 4):
        for missing in (0, 513*nq//4, 513*nq):
            for rep in range(3):
                elapsed = 819200000.*factor
                result.append(dict(phase='planner_sample', nq=nq, missing=missing, rep=rep,
                    unique_pages=513*nq, cooperative=arm == 'cooperative', oracle_exact=True,
                    graph_nodes=2, calls=8192, elapsed_ns=elapsed, ns_per_call=elapsed/8192,
                    clock_pair_ns=20., timer_fraction=5*20.*8192/elapsed,
                    calibration_ns=800000000., warmup_ns=512000000.,
                    requested_host_payload_bytes_per_call=missing*4224))
    return result


def text(values):
    return '\n'.join([*(json.dumps(r) for r in values), MARKER])


class PlannerEvidence(unittest.TestCase):
    def test_complete_each_arm(self):
        for arm in ('scalar', 'cooperative'):
            self.assertEqual(controller.parse_samples(text(rows(arm)), arm), rows(arm))

    def test_missing_duplicate_or_unfinished_matrix(self):
        base = rows()
        for case in (base[:-1], base[:-1]+base[:1]):
            with self.assertRaises(RuntimeError): controller.parse_samples(text(case), 'scalar')
        with self.assertRaises(RuntimeError): controller.parse_samples(text(base).replace(MARKER,''), 'scalar')

    def test_actual_arm_and_oracle(self):
        for key, value in (('cooperative',True), ('oracle_exact',False), ('cooperative',0)):
            case = rows(); case[0][key] = value
            with self.assertRaises(RuntimeError): controller.parse_samples(text(case), 'scalar')

    def test_all_duration_floors(self):
        for key, value in (('elapsed_ns',249999999.),('calibration_ns',749999999.),('warmup_ns',499999999.)):
            case=rows(); case[0][key]=value
            case[0]['ns_per_call']=case[0]['elapsed_ns']/case[0]['calls']
            case[0]['timer_fraction']=5*case[0]['clock_pair_ns']*case[0]['calls']/case[0]['elapsed_ns']
            with self.assertRaises(RuntimeError): controller.parse_samples(text(case), 'scalar')

    def test_clock_cost_and_arithmetic(self):
        case=rows();case[0]['clock_pair_ns']=1000.
        case[0]['timer_fraction']=5*1000.*8192/case[0]['elapsed_ns']
        with self.assertRaises(RuntimeError): controller.parse_samples(text(case), 'scalar')
        for key in ('ns_per_call','timer_fraction'):
            case=rows();case[0][key]*=1.1
            with self.assertRaises(RuntimeError): controller.parse_samples(text(case), 'scalar')

    def test_nonfinite_and_geometry_or_type_corruption(self):
        for key,value in (('elapsed_ns',float('nan')),('graph_nodes',1),('unique_pages',0),
                          ('requested_host_payload_bytes_per_call',True),('calls',63),('nq',2.)):
            case=rows();case[0][key]=value
            with self.assertRaises(RuntimeError): controller.parse_samples(text(case), 'scalar')

    def test_three_pairs_and_all_miss_cells_required(self):
        cohort=[dict(arm=arm,samples=rows(arm,.97 if arm=='cooperative' else 1.)) for arm in controller.ARMS]
        # One slower pair defeats eligibility even though its geometric mean is faster.
        for r in cohort[1]['samples']:
            if r['nq']==2 and r['missing']==513*2//4: r['ns_per_call']=102000.
        decision=controller.decision(cohort)
        self.assertEqual(decision['eligible_widths'],[4])
        cell=next(c for c in decision['cells'] if c['nq']==2 and c['missing']==256)
        self.assertLess(cell['geometric_mean_ratio'],1.)
        self.assertFalse(cell['reproducible_gain'])
        self.assertFalse(decision['production_adopted'])

    def test_incomplete_or_reordered_processes_rejected(self):
        cohort=[dict(arm=arm,samples=rows(arm)) for arm in controller.ARMS]
        with self.assertRaises(RuntimeError): controller.decision(cohort[:-1])
        bad=copy.deepcopy(cohort);bad[0]['arm']='cooperative'
        with self.assertRaises(RuntimeError): controller.decision(bad)


if __name__ == '__main__': unittest.main()
