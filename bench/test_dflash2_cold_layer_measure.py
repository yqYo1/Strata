import unittest
from unittest.mock import patch

import dflash2_cold_layer_measure as layer


class ColdLayerAccounting(unittest.TestCase):
    def setUp(self):
        vals = dict(wall_s=3, cpu_s=1, unclassified_wall_s=.5, unclassified_cpu_s=.25,
                    minor_faults=2, major_faults=0, peak_unclassified_wall_s=.3, peak_layer=1)
        tokens = ['version=1', 'available=1', 'layers=2']
        tokens.extend(f'{s}_{k}={v}' for s in layer.SEGMENTS for k, v in vals.items())
        self.line = ' '.join(tokens)
        self.enclosing = dict(wall_s=9.1, cpu_s=3.1, unclassified_wall_s=1.6,
                              unclassified_cpu_s=.8, minor_faults=6, major_faults=0)

    def test_reads_segments_with_unmeasured_boundary_remainder(self):
        data = layer.layer_accounting([self.line], self.enclosing, 2, enabled=True)
        self.assertEqual(tuple(data), layer.SEGMENTS)
        self.assertEqual(data['run']['peak_layer'], 1)
        self.assertEqual(sum(s['minor_faults'] for s in data.values()), 6)

    def test_requires_exact_record_cardinality_and_extent(self):
        self.assertIsNone(layer.layer_accounting([], {}, 0, enabled=False))
        for lines, n, enabled in (([self.line], 2, False), ([], 2, True),
                                  ([self.line, self.line], 2, True), ([self.line], 3, True)):
            with self.subTest(n=n, enabled=enabled, lines=len(lines)), self.assertRaises(RuntimeError):
                layer.layer_accounting(lines, self.enclosing, n, enabled=enabled)

    def test_rejects_invalid_or_escaping_subtotals(self):
        for old, new in (('run_peak_layer=1', 'run_peak_layer=2'),
                         ('retire_minor_faults=2', 'retire_minor_faults=3'),
                         ('run_wall_s=3', 'run_wall_s=4'),
                         ('run_unclassified_cpu_s=0.25', 'run_unclassified_cpu_s=nan')):
            with self.subTest(field=old), self.assertRaises(RuntimeError):
                layer.layer_accounting([self.line.replace(old, new)], self.enclosing, 2, enabled=True)

    def test_arms_change_only_the_detail_switch(self):
        with patch.object(layer.coarse, 'environment', return_value={layer.wait.SELECTOR: '1'}):
            control = layer.environment({}, {'arm': 'coarse'})
            detailed = layer.environment({}, {'arm': 'layers'})
        self.assertEqual(detailed, dict(control, **{layer.SELECTOR: '1'}))
        self.assertEqual(control[layer.SELECTOR], '0')
        self.assertEqual(control[layer.coarse.SELECTOR], '1')
