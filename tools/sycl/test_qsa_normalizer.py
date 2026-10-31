#!/usr/bin/env python3
"""Host-only refusal checks for the prospective normalizer evidence gate."""
import copy
import unittest
import qsa_normalizer as probe


def comparison():
    result=[]
    for process in range(3):
        rows=[]
        for scenario in range(4):
            for rep in range(-1,3):
                for arm in range(2):
                    scale=.97 if arm else 1
                    rows.append(dict(process=str(process),scenario=str(scenario),cells=str((32768,117488)[scenario//2]),
                        gapped=str(scenario%2),queries='8192',cap='2051',page='4',batch='8',repetition=str(rep),arm=str(arm),
                        wall_s=str(.2*scale),pack_s='.01',attention_s=str(.1*scale),output='PASS'))
        result.append(dict(kind='compare',argv=['--compare',str(process)],passed=True,samples=rows,
            case_checks=[dict(scenario=str(i),inputs='PASS',guards='PASS') for i in range(4)],
            controls=[dict(repetition=str(i),wall_s='.00001') for i in range(-1,5)],
            timers=[dict(samples='1000',p99_ns='100')]))
    return result


def small_numerics():
    return dict(NUMERICAL=[dict(cells='17',cap='17',queries='1',page='4',kind='0',masked='0',
        inputs='PASS',guards='PASS',repeatability='PASS',output_budget_ratio='.01')],
        STATE=[dict(PV_bits='PASS',probability_bits='PASS',maximum_alpha_bits='PASS',accumulator_bits='PASS',guards='PASS',coordinates='4',normalizer_max_abs_delta='0.000001')],
        ORACLE=[dict(PASS='1',compared='12288',rtol='.02',atol='.02',maximum_budget_ratio='.02')])


class EvidenceGate(unittest.TestCase):
    def test_qualified_independent_cohort(self):
        d=probe.decision(comparison())
        self.assertTrue(d['mechanism_screen_passed']);self.assertEqual(len(d['process_scenario']),12)

    def test_bad_process_not_hidden_by_average(self):
        data=comparison()
        for s in data[2]['samples']:
            if s['arm']=='1':s['wall_s']='.202'
        self.assertFalse(probe.decision(data)['mechanism_screen_passed'])

    def test_duplicate_process_refused(self):
        data=comparison();data[2]=copy.deepcopy(data[1])
        with self.assertRaises(RuntimeError):probe.decision(data)

    def test_missing_pair_refused(self):
        data=comparison();data[0]['samples'].pop()
        with self.assertRaises(RuntimeError):probe.decision(data)

    def test_wrong_actual_shape_refused(self):
        data=comparison();data[0]['samples'][0]['cells']='16384'
        with self.assertRaises(RuntimeError):probe.decision(data)

    def test_missing_input_closure_refused(self):
        data=comparison();data[1]['case_checks'][2]['inputs']='FAIL'
        with self.assertRaises(RuntimeError):probe.decision(data)

    def test_clock_cost_refused(self):
        data=comparison();data[0]['controls'][2]['wall_s']='.2';data[0]['controls'][3]['wall_s']='.2';data[0]['controls'][4]['wall_s']='.2'
        with self.assertRaises(RuntimeError):probe.decision(data)

    def test_numerical_shape_and_state(self):
        data=small_numerics();probe.numerical_check(data,True)
        data['STATE'][0]['PV_bits']='FAIL'
        with self.assertRaises(RuntimeError):probe.numerical_check(data,True)

    def test_numerical_nonfinite_and_tolerance_refused(self):
        for key,value in [('output_budget_ratio','nan'),('output_budget_ratio','1.1')]:
            data=small_numerics();data['NUMERICAL'][0][key]=value
            with self.assertRaises(RuntimeError):probe.numerical_check(data,True)
        data=small_numerics();data['ORACLE'][0]['atol']='.03'
        with self.assertRaises(RuntimeError):probe.numerical_check(data,True)


if __name__=='__main__':unittest.main()
