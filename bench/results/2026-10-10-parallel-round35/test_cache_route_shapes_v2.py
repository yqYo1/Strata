"""SOURCE ONLY v2: independent job-list plus literal mutation oracles.

Immutable shape module and v1 fixture are untouched. Root runs all tests.
"""
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


def phase_and_nt_totals(cells):
    """Test-owned observed sum, compared to independently declared literals."""
    phase={}; nt={}; keys=[]
    for cell in cells:
        key=(cell['phase'],cell['type'],cell['nt']);keys.append(key)
        for table,at in ((phase,cell['phase']),(nt,(cell['phase'],cell['nt']))):
            values=table.setdefault(at,[0,0,0])
            for i,name in enumerate(('experts','tokens','output_rows')): values[i]+=cell[name]
    return phase,nt,keys


def literal_cell(phase,typ,nt,experts,tokens,rows):
    return dict(phase=phase,type=typ,nt=nt,experts=experts,tokens=tokens,output_rows=rows)


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

    def assert_literal_reference_aggregates(self,histogram):
        phase,nt,keys=phase_and_nt_totals(histogram['cells'])
        self.assertEqual(len(keys),len(set(keys)), 'mutation must not reach duplicate-key rejection')
        self.assertEqual(phase,{'GU':[3,5,1920],'Down':[3,5,7680]})
        self.assertEqual(nt,{('GU',1):[1,1,640],('GU',2):[2,4,1280],
                             ('Down',1):[1,1,2560],('Down',2):[2,4,5120]})

    def test_unique_key_nt_redistribution_preserves_all_aggregate_totals(self):
        data=reference()
        self.assert_literal_reference_aggregates(data[3])
        # Independently literal cells: GU18's NT1+NT2 become two NT2 jobs;
        # GU22's one NT2 becomes one NT1. Neither phase nor per-NT global
        # experts/tokens/rows changes. There is no duplicate key to reject first.
        data[3]['cells']=[
            literal_cell('GU',18,2,2,4,1280),
            literal_cell('GU',22,1,1,1,640),
            literal_cell('Down',20,1,1,1,2560),
            literal_cell('Down',20,2,1,2,2560),
            literal_cell('Down',42,2,1,2,2560),
        ]
        self.assert_literal_reference_aggregates(data[3])
        self.assertEqual({(c['phase'],c['type'],c['nt']):(c['experts'],c['tokens'],c['output_rows'])
                          for c in data[3]['cells']}, {
            ('GU',18,2):(2,4,1280),('GU',22,1):(1,1,640),
            ('Down',20,1):(1,1,2560),('Down',20,2):(1,2,2560),('Down',42,2):(1,2,2560)})
        with self.assertRaisesRegex(ValueError,'^per-format/NT histogram differs from pair-derived jobs$'):
            reconcile_shapes(*data)

    def test_unique_down20_down42_swap_preserves_all_aggregate_totals(self):
        data=reference()
        self.assert_literal_reference_aggregates(data[3])
        # Down rows/job are identical for20/42. Swap their format assignments
        # using literals while GU remains exactly its original job-list oracle.
        data[3]['cells']=[
            literal_cell('GU',18,1,1,1,640),
            literal_cell('GU',18,2,1,2,640),
            literal_cell('GU',22,2,1,2,640),
            literal_cell('Down',42,1,1,1,2560),
            literal_cell('Down',42,2,1,2,2560),
            literal_cell('Down',20,2,1,2,2560),
        ]
        self.assert_literal_reference_aggregates(data[3])
        self.assertEqual({(c['type'],c['nt']):(c['experts'],c['tokens'],c['output_rows'])
                          for c in data[3]['cells'] if c['phase']=='Down'}, {
            (42,1):(1,1,2560),(42,2):(1,2,2560),(20,2):(1,2,2560)})
        with self.assertRaisesRegex(ValueError,'^per-format/NT histogram differs from pair-derived jobs$'):
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
