#!/usr/bin/env python3
"""Host-only admission controls for one request's source-emitted MoE plan."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
import bench_engine as subject


PAIRS = ((0, 8193), (8193, 3))
FOOTER = ('strata MoE accumulator window: windows=96 dequants=960 jobs=1440 '
          'source_H2D_bytes=15736320 GPU_resident_windows_only=1 operator_max=8192 maximum_rows=8193')


def windows():
    # Ten distinct experts per token. An 8193-row expert needs two jobs;
    # the final three-row window needs one. The remaining 502 are unused.
    return [dict(version=1, layer=layer, first_row=first, rows=rows,
                 jobs=20 if rows == 8193 else 10,
                 job_rows_histogram=[[1, 10], [8192, 10]] if rows == 8193 else [[3, 10]],
                 jobs_per_expert_histogram=[[0, 502], [2 if rows == 8193 else 1, 10]])
            for layer in range(48) for first, rows in PAIRS]


def render(rows, footer=FOOTER):
    return '\n'.join('strata MoE planned job census: ' + json.dumps(row) for row in rows) + '\n' + footer


class PlannedMoeWorkTest(unittest.TestCase):
    def test_plan_without_optional_resource_observers(self):
        result = subject.extract_planned_moe_work(render(windows()), PAIRS)
        self.assertEqual(len(result['planned_moe_windows']), 96)
        self.assertEqual(sum(w['jobs'] for w in result['planned_moe_windows']), 1440)
        self.assertEqual(sum(w['rows'] * 10 for w in result['planned_moe_windows']), 48 * 8196 * 10)
        self.assertNotIn('device_events', result)

    def test_configured_window_capacity_can_exceed_actual_rows(self):
        footer = FOOTER.replace('maximum_rows=8193', 'maximum_rows=81920')
        self.assertEqual(len(subject.extract_planned_moe_work(render(windows(), footer), PAIRS)
                             ['planned_moe_windows']), 96)

    def test_missing_repeated_reordered_or_changed_window(self):
        original = windows()
        for case in (original[:-1], original + [original[-1]],
                     [original[1], original[0], *original[2:]]):
            with self.subTest(case=len(case)), self.assertRaises((RuntimeError, ValueError)):
                subject.extract_planned_moe_work(render(case), PAIRS)
        changed = copy.deepcopy(original)
        changed[1]['first_row'] -= 1
        with self.assertRaises((RuntimeError, ValueError)):
            subject.extract_planned_moe_work(render(changed), PAIRS)

    def test_invalid_or_nonconserving_histogram(self):
        for name, histogram in (
            ('job_rows_histogram', [[1, 5], [1, 5], [8192, 10]]),
            ('job_rows_histogram', [[8192, 10], [1, 10]]),
            ('job_rows_histogram', [[0, 10], [8192, 10]]),
            ('job_rows_histogram', [[1, True], [8192, 10]]),
            ('job_rows_histogram', [[1, -1], [8192, 10]]),
            ('job_rows_histogram', [[1, 9], [8192, 10]]),
            ('jobs_per_expert_histogram', [[0, 501], [2, 10]]),
            ('jobs_per_expert_histogram', [[0, 502], [11, 10]])):
            rows = windows()
            rows[0][name] = histogram
            with self.subTest(name=name, histogram=histogram), self.assertRaises((RuntimeError, ValueError)):
                subject.extract_planned_moe_work(render(rows), PAIRS)

    def test_invalid_window_scalar(self):
        for key, value in (('version', 2), ('layer', False), ('rows', True),
                           ('first_row', False), ('jobs', True)):
            rows = windows()
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises((RuntimeError, ValueError)):
                subject.extract_planned_moe_work(render(rows), PAIRS)

    def test_missing_duplicate_or_inconsistent_retirement(self):
        for value in ('', FOOTER + '\n' + FOOTER, FOOTER + ' jobs=1',
                      FOOTER.replace('1440', '1441'), FOOTER.replace('windows=96', 'windows=95'),
                      FOOTER.replace('15736320', '15736321'),
                      FOOTER.replace('operator_max=8192', 'operator_max=8193')):
            with self.subTest(footer=value), self.assertRaises((RuntimeError, ValueError)):
                subject.extract_planned_moe_work(render(windows(), value), PAIRS)


if __name__ == '__main__':
    unittest.main()
