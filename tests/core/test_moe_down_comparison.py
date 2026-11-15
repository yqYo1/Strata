"""Host-only controls for the frozen model-comparison advancement decision."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import compare_moe_down as comparison


def cohort(row64=98.8):
    times = dict(original=100., row128=99., row64=row64)
    return [dict(kernel=kernel, subsequent_median_ms=times[kernel],
        passed=True, complete=True, ownership_closed=True, fresh_faults=[], clean_speed=True,
        native_identity=dict(pid=100+i, start_ticks=1000+i)) for i, kernel in enumerate(comparison.ORDER)]


class AdvancementTests(unittest.TestCase):
    def test_nearby_faster_row64_does_not_replace_qualified_row128(self):
        self.assertEqual(comparison.decision(cohort())['eligible_policy_for_broader_qualification'], 'row128')

    def test_material_repeated_row64_improvement_advances_only_to_more_qualification(self):
        result = comparison.decision(cohort(98.))
        self.assertEqual(result['eligible_policy_for_broader_qualification'], 'row64')
        self.assertFalse(result['inference_adopted'])
        self.assertFalse(result['sustained_decode_qualified'])
        self.assertFalse(result['physical262144_qualified'])

    def test_one_slow_block_prevents_advancement_despite_a_favorable_pooled_gain(self):
        rows = cohort(90.)
        for row in rows[:3]:
            if row['kernel'] in ('row128', 'row64'):
                row['subsequent_median_ms'] = 101.
        self.assertIsNone(comparison.decision(rows)['eligible_policy_for_broader_qualification'])

    def test_partial_failed_observed_duplicate_and_invalid_clock_cohorts_are_rejected(self):
        cases = [cohort()[:-1]]
        for field, value in (('passed', False), ('complete', False), ('ownership_closed', False),
                             ('fresh_faults', ['fault']), ('clean_speed', False),
                             ('subsequent_median_ms', float('nan')), ('subsequent_median_ms', 0.)):
            rows = cohort(); rows[-1][field] = value; cases.append(rows)
        rows = cohort(); rows[-1]['native_identity'] = copy.deepcopy(rows[0]['native_identity']); cases.append(rows)
        for rows in cases:
            with self.subTest(last=rows[-1]):
                with self.assertRaises(RuntimeError):
                    comparison.decision(rows)


if __name__ == '__main__':
    unittest.main()
