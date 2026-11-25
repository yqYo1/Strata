"""Independent small oracle checks and canonical packet/answer validation."""

import itertools
import json
import re
import unittest

from dflash2_scheduling_fixture import JOBS, WINDOWS, Option, build, solve, validate_answer


class SchedulingFixtureTest(unittest.TestCase):
    def test_solve_matches_exhaustive_rectangular_assignment(self):
        allowed = lambda risk, disruption: Option(True, risk, disruption)
        options = [
            [allowed(8, 1), allowed(3, 8), allowed(3, 8), Option(False, 0, 0)],
            [allowed(3, 0), allowed(2, 9), allowed(2, 9), allowed(12, 0)],
            [allowed(4, 2), allowed(8, 0), allowed(8, 0), allowed(5, 1)],
        ]
        feasible = []
        for slots in itertools.permutations(range(4), 3):
            selected = [row[slot] for row, slot in zip(options, slots, strict=True)]
            if all(item.allowed for item in selected):
                feasible.append((sum(item.risk for item in selected),
                                 sum(item.disruption for item in selected), slots))
        expected = min(feasible)
        chosen = solve(options)
        self.assertEqual((chosen['risk'], chosen['disruption'],
                          tuple(int(item['window'][1:]) for item in chosen['assignments'])), expected)

    def test_solving_an_infeasible_packet_fails(self):
        with self.assertRaisesRegex(ValueError, 'no complete feasible assignment'):
            solve([[Option(True, 1, 1), Option(False, 0, 0)]] * 2)

    def test_canonical_serialization_and_oracle(self):
        fixture = build()
        self.assertEqual(fixture, build())
        rows = re.findall(r'^change (J\d+), window (W\d+), approved ([01]), risk (\d+), disruption (\d+)$',
                          fixture['messages'][0]['content'], re.M)
        packet = {(job, window): (int(ok), int(risk), int(disruption))
                  for job, window, ok, risk, disruption in rows}
        self.assertEqual(len(rows), fixture['records'])
        self.assertEqual(len(packet), JOBS * WINDOWS)
        expected = fixture['expected']
        assignments = expected['assignments']
        self.assertEqual([item['job'] for item in assignments], [f'J{i:02}' for i in range(JOBS)])
        self.assertEqual(len({item['window'] for item in assignments}), JOBS)
        for item in assignments:
            self.assertEqual(packet[item['job'], item['window']], (1, item['risk'], item['disruption']))
        for field in ('risk', 'disruption'):
            self.assertEqual(expected[field], sum(item[field] for item in assignments))
        fence = '\x60' * 3
        answer = f'Handover.\n{fence}json\n{json.dumps(expected)}\n{fence}\n'
        validate_answer(answer, expected)
        with self.assertRaisesRegex(ValueError, 'differs'):
            validate_answer(answer.replace(f'"risk": {expected["risk"]}', '"risk": false', 1), expected)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_answer(answer.replace('{"risk":', '{"risk": 0, "risk":', 1), expected)


if __name__ == '__main__':
    unittest.main()
