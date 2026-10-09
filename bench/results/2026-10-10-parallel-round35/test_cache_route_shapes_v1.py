"""Independent scalar job-list oracle; no model/controller imports."""
import copy
import hashlib
from pathlib import Path
import unittest
from cache_route_shapes_v1 import formats_from_metadata, reconcile_shapes


def reference():
    formats = [(18, 20), (22, 42)] + [(21, 20)] * 46
    residents = {(47, 7)}
    # Explicit evaluated expert jobs. Counts are built from these token lists,
    # rather than from the NT reconstruction formula under test.
    jobs = [(0, 0, (1,)), (0, 0, (2, 3)), (1, 8, (2, 3)),
            (47, 7, (1,)), (47, 7, (2,)), (47, 7, (3, 4))]
    pairs = {}
    histogram = {}
    for layer, expert, tokens in jobs:
        p = pairs.setdefault((layer, expert), dict(layer=layer, expert=expert,
            entries=0, callback_pairs=0, hits=0, refused=0, offloaded=0))
        p['entries'] += len(tokens)
        p['callback_pairs'] += 1
        p['hits' if (layer, expert) in residents else 'refused'] += len(tokens)
        if (layer, expert) not in residents:
            for phase, typ, rows in (('GU', formats[layer][0], 640), ('Down', formats[layer][1], 2560)):
                c = histogram.setdefault((phase, typ, len(tokens)), dict(phase=phase,
                    type=typ, nt=len(tokens), experts=0, tokens=0, output_rows=0))
                c['experts'] += 1; c['tokens'] += len(tokens); c['output_rows'] += rows
    parsed = dict(passed=True, begin={'complete': 1}, end={'complete': 1, 'offloaded': 0},
                  pairs=list(pairs.values()))
    return parsed, residents, formats, dict(passed=True, cells=list(histogram.values()))


class ShapeFixtures(unittest.TestCase):
    def test_explicit_jobs_mixed_nt_formats_and_cached_pair(self):
        result = reconcile_shapes(*reference())
        self.assertTrue(result['passed'])
        self.assertEqual(result['pair_shapes'], [(0,0,3,2,1,1), (1,8,2,1,0,1), (47,7,4,3,2,1)])
        self.assertEqual(sum(c['experts'] for c in result['cells'] if c['phase']=='GU'), 3)
        self.assertEqual(sum(c['tokens'] for c in result['cells'] if c['phase']=='Down'), 5)

    def test_mislabelled_format_preserving_global_totals_rejected(self):
        data = reference()
        for cell in data[3]['cells']:
            if cell['phase'] == 'GU' and cell['type'] == 18:
                cell['type'] = 23
        with self.assertRaisesRegex(ValueError, 'per-format/NT histogram differs'):
            reconcile_shapes(*data)

    def test_wrong_nt_preserving_expert_total_rejected(self):
        data = reference(); data[3]['cells'][0]['nt'] = 2
        with self.assertRaisesRegex(ValueError, 'duplicate actual histogram cell'):
            reconcile_shapes(*data)

    def test_all_hit_and_all_miss(self):
        data = reference()
        data[0]['pairs'] = [data[0]['pairs'][-1]]; data[3]['cells'] = []
        self.assertEqual(reconcile_shapes(*data)['cells'], [])
        data = reference(); data[0]['pairs'].pop()
        self.assertTrue(reconcile_shapes(*data)['passed'])

    def test_impossible_nt_or_zero_jobs_rejected(self):
        for n, j in [(0,0), (3,1), (1,2), (True,1), ((1<<64),1)]:
            data = reference(); data[0]['pairs'][0].update(entries=n, callback_pairs=j)
            with self.subTest(n=n,j=j), self.assertRaisesRegex(ValueError, 'source-proved NT1/NT2'):
                reconcile_shapes(*data)

    def test_incomplete_or_changed_classification_rejected(self):
        data = reference(); data[0]['begin']['complete'] = 0
        with self.assertRaisesRegex(ValueError, 'incomplete/adaptive route scope'): reconcile_shapes(*data)
        data = reference(); data[0]['pairs'][0]['hits'] = 1
        with self.assertRaisesRegex(ValueError, 'nonresident pair classification'): reconcile_shapes(*data)
        data = reference(); data[0]['pairs'][-1]['refused'] = 1
        with self.assertRaisesRegex(ValueError, 'resident pair classification'): reconcile_shapes(*data)

    def test_rows_and_missing_extra_cells_rejected(self):
        for change in ('rows', 'missing', 'extra'):
            data = reference()
            if change == 'rows': data[3]['cells'][0]['output_rows'] += 1
            elif change == 'missing': data[3]['cells'].pop()
            else:
                extra = copy.deepcopy(data[3]['cells'][0]); extra['type'] = 21
                data[3]['cells'].append(extra)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'per-format/NT histogram differs'):
                reconcile_shapes(*data)

    def test_actual_small_metadata_identity_geometry(self):
        data = Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s/native_experts.txt').read_bytes()
        expected = 'd9ac2dfa3ee63c55c9c6a6db26f72da0cec0ee41733f007e5cbbbba71617aa5d'
        formats = formats_from_metadata(data, expected)
        self.assertEqual(len(formats), 48)
        self.assertEqual(formats[:2], [(18,20), (22,42)])
        self.assertEqual(formats[-1], (23,20))
        with self.assertRaisesRegex(ValueError, 'metadata hash mismatch'):
            formats_from_metadata(data+b'\n', expected)
        malformed = data.replace(b'2176000', b'2176001', 1)
        with self.assertRaisesRegex(ValueError, 'logical layout/shape'):
            formats_from_metadata(malformed, hashlib.sha256(malformed).hexdigest())


if __name__ == '__main__':
    unittest.main()
