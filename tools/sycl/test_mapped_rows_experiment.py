"""CPU-only admission controls; no device/runtime initialization."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mapped_rows_experiment as trial


def corpus(arm='scalar', ns=10000.):
    return [dict(type='sample', rows=r, width=2560, hits=h, producer_update=u,
        subgroup_dispatch=arm == 'subgroup', rep=i, calls=32768,
        elapsed_ns=ns * 32768, ns_per_call=ns, clock_pair_ns=20.,
        timer_fraction=20./ns, calibration_ns=400000000.)
        for r in (10, 20, 30, 40) for h in (0, r//4, r)
        for u in (False, True) for i in range(3)]


def encode(data):
    return '\n'.join(json.dumps(x) for x in data)


class Admission(unittest.TestCase):
    def graph_log(self):
        names = {'urCommandBufferCreateExp':18, 'urCommandBufferAppendKernelLaunchWithArgsExp':18,
                 'urCommandBufferFinalizeExp':18, 'urEnqueueCommandBufferExp':54, 'urCommandBufferReleaseExp':18}
        return '\n'.join(' ---> '+k+'\n <--- '+k+'(.arg = 0) -> UR_RESULT_SUCCESS;'
                         for k,n in names.items() for _ in range(n))

    def test_actual_command_graph_api(self):
        self.assertTrue(trial.graph_trace(self.graph_log())['complete_success_pairs'])

    def test_missing_graph_enqueue_or_append_refused(self):
        text = self.graph_log()
        for name in ('urEnqueueCommandBufferExp', 'urCommandBufferAppendKernelLaunchWithArgsExp'):
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                trial.graph_trace(text.replace(' ---> '+name, ' ---> Missing', 1))

    def test_graph_error_refused(self):
        with self.assertRaises(RuntimeError):
            trial.graph_trace(self.graph_log().replace('UR_RESULT_SUCCESS', 'UR_RESULT_ERROR_INVALID_VALUE', 1))

    def test_completion_is_consumed_before_eof(self):
        from unittest.mock import Mock
        io = Mock(); io.line.return_value = 'fixture=PASS checked_teardown=PASS'
        trial.complete_retirement(io)
        self.assertEqual([x[0] for x in io.method_calls], ['send', 'line', 'exit'])

    def test_bad_completion_refuses_qualification(self):
        from unittest.mock import Mock
        io = Mock(); io.line.return_value = 'fixture=FAIL'
        with self.assertRaises(RuntimeError): trial.complete_retirement(io)
        io.exit.assert_not_called()

    def test_exact_matrix(self):
        for arm in ('scalar', 'subgroup'):
            data = corpus(arm)
            self.assertEqual(trial.samples(encode(data), arm), data)

    def test_missing_duplicate_changed_dispatch(self):
        data = corpus()
        for bad in (data[:-1], data[:-1] + [data[0]], corpus('subgroup')):
            with self.assertRaises(RuntimeError): trial.samples(encode(bad), 'scalar')

    def test_duration_and_timer_must_pass(self):
        for k, v in (('elapsed_ns', 249999999.), ('calibration_ns', 299999999.),
                     ('timer_fraction', .010001), ('calls', 32767),
                     ('ns_per_call', float('nan')), ('clock_pair_ns', -1.)):
            data = corpus(); data[0][k] = v
            with self.subTest(field=k), self.assertRaises(RuntimeError):
                trial.samples(encode(data), 'scalar')

    def test_single_slow_pair_remains_failed(self):
        cases = [dict(mode=a, samples=corpus(a, 9000. if a == 'subgroup' else 10000.))
                 for a in trial.ORDER]
        self.assertEqual(trial.decision(cases)['qualified_primary_rows'], [10,20,30,40])
        changed = copy.deepcopy(cases)
        for s in changed[1]['samples']:
            if s['rows'] == 20: s['ns_per_call'] = 11000.
        decision = trial.decision(changed)
        self.assertEqual(decision['qualified_primary_rows'], [10,30,40])
        self.assertFalse(decision['production_adopted'])
        self.assertFalse(decision['model_speed_qualified'])

    def test_incomplete_or_reordered_cohort_refused(self):
        cases = [dict(mode=a, samples=corpus(a)) for a in trial.ORDER]
        for bad in (cases[:-1], list(reversed(cases))):
            with self.assertRaises(RuntimeError): trial.decision(bad)


if __name__ == '__main__': unittest.main()
