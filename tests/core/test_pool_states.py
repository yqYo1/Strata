"""Reject incomplete attribution and phase/accounting errors; no native execution."""
import copy
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
import pool_states as subject


def evidence(case=(20, 0, 0, 0), mode='measure', observed=1):
    down, mix, phase, gap = case
    jobs, entries, ones = (24, 36, 16) if mix else (28, 35, 22)
    steps = 168 // jobs
    samples = [] if mode == 'check' else [dict(
        kind='sample', rep=i, passes=1, batch_calls=steps,
        wall_s=1.2, cpu_s=5., GU_s=.7, quant_s=.02, Down_s=.3,
        row_ops=steps*640*entries, NT1_rows=steps*640*ones,
        pair_rows=steps*640*ones if phase else 0, min_thread_rows=640)
        for i in range(5)]
    calls = (2 + len(samples))*steps
    threads = []
    if observed:
        for slot in range(6):
            states = [[0, 0, 1, 100, 50],
                      [phase+1, 3, 3*calls, 300, 280],
                      [phase+1, 4, 3*calls, 250, 200]]
            if slot < 5:
                states += [[0, 1, 1, 200, 190]]
                if gap:
                    states += [[0, 2, 1, 500, 10]]
            else:
                states += [[phase+1, 5, 4*calls, 100, 90],
                           [phase+1, 6, 2*calls, 100, 90],
                           [phase+1, 7, calls, 100, 90]]
            threads.append(dict(kind='thread', slot=slot, tid=100+slot, core=slot,
                                life_wall_ns=sum(s[3] for s in states),
                                life_cpu_ns=sum(s[4] for s in states),
                                voluntary_switches=1, involuntary_switches=0,
                                states=states))
    return threads + samples + [dict(
        kind='closed', mode=mode, observed=bool(observed), down=down, mixture=mix,
        phase=phase, gap_ms=gap, calls_including_warmups=calls,
        clock_probe_s=1e-7, warm_pass_s=.01, bitwise=True, guards=True,
        inputs_unchanged=True, passed=True)]


class PoolStates(unittest.TestCase):
    def parse(self, rows, case=(20, 0, 0, 0), observed=1, mode='measure'):
        return subject.parse(rows, case, observed, mode, list(range(6)))

    def test_both_phases_shapes_and_idle_cases_conserve(self):
        for case in subject.CASES:
            for mode in ('check', 'measure'):
                for observed in (0, 1):
                    with self.subTest(case=case, mode=mode, observed=observed):
                        result = self.parse(evidence(case, mode, observed), case, observed, mode)
                        self.assertEqual(len(result['samples']), 0 if mode == 'check' else 5)

    def test_missing_duplicate_or_unaccounted_thread_refused(self):
        for mutation in ('missing', 'duplicate', 'wall', 'cpu', 'tid', 'core', 'negative'):
            rows = evidence()
            if mutation == 'missing': rows.pop(0)
            elif mutation == 'duplicate': rows[1] = copy.deepcopy(rows[0])
            elif mutation == 'wall': rows[0]['life_wall_ns'] += 1
            elif mutation == 'cpu': rows[0]['life_cpu_ns'] += 1
            elif mutation == 'tid': rows[1]['tid'] = rows[0]['tid']
            elif mutation == 'core': rows[1]['core'] = rows[0]['core']
            else: rows[0]['voluntary_switches'] = -1
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                self.parse(rows)

    def test_wrong_work_phase_and_assigned_idle_refused(self):
        for state, value in ((0, 8), (1, 2), (3, 1)):
            rows = evidence()
            # Unexpected work kind, GU tagged Decode during Prefill, or assigned idle.
            if state == 0: rows[0]['states'][1][1] = value
            else: rows[0]['states'][state][0] = value
            with self.subTest(state=state), self.assertRaises(RuntimeError):
                self.parse(rows)

    def test_missing_task_or_duplicate_state_refused(self):
        rows = evidence(); rows[0]['states'][1][2] -= 1
        with self.assertRaises(RuntimeError): self.parse(rows)
        rows = evidence(); rows[0]['states'].append(copy.deepcopy(rows[0]['states'][0]))
        rows[0]['life_wall_ns'] += 100; rows[0]['life_cpu_ns'] += 50
        with self.assertRaises(RuntimeError): self.parse(rows)

    def test_long_gap_requires_actual_cv_entry(self):
        case = (20, 0, 0, 40); rows = evidence(case)
        for row in rows[:5]:
            state = row['states'].pop()
            row['life_wall_ns'] -= state[3]; row['life_cpu_ns'] -= state[4]
        with self.assertRaises(RuntimeError): self.parse(rows, case)

    def test_timer_dispatch_and_complete_samples_required(self):
        for field, value in (('wall_s', .999), ('cpu_s', float('nan')),
                             ('GU_s', 2.), ('rep', 1), ('pair_rows', 1),
                             ('row_ops', 1), ('min_thread_rows', 0)):
            rows = evidence(); rows[6][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError): self.parse(rows)
        rows = evidence(); rows[-1]['clock_probe_s'] = 1.
        with self.assertRaises(RuntimeError): self.parse(rows)

    def test_raw_cannot_supply_observer_or_relabel_numeric_failure(self):
        rows = evidence(observed=0); rows.insert(0, evidence()[0])
        with self.assertRaises(RuntimeError): self.parse(rows, observed=0)
        rows = evidence(); rows[-1]['guards'] = False
        with self.assertRaises(RuntimeError): self.parse(rows)

    def test_overhead_is_independent_per_phase_shape_and_gap(self):
        checks = [dict(case=list(case), observed=arm,
                       samples=[dict(wall_s=1.01 if arm else 1., batch_calls=1)]*5)
                  for case in subject.CASES for arm in subject.ORDER]
        checks[2]['samples'] = [dict(wall_s=1.11, batch_calls=1)]*5
        result = subject.reduce(checks)
        self.assertFalse(result[0]['overhead_gate_passed'])
        self.assertTrue(all(r['overhead_gate_passed'] for r in result[1:]))
        self.assertEqual([r['compute_overhead_case'] for r in result], [True, False]*4)
        with self.assertRaises(RuntimeError): subject.reduce(checks[:-1])


if __name__ == '__main__': unittest.main()
