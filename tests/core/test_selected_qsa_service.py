"""Reject incomplete or misclassified closed QSA evidence; no GPU use."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import selected_qsa_service as q


class SourceGeometryTests(unittest.TestCase):
    def fixture(self):
        rows = []
        cases = [('attention',16,128,[(16384,168),(5600,12)]),
            ('pack',32,64,[(n,12) for n in [*range(16384,229377,16384),234976]])]
        for family,simd,local,shapes in cases:
            for group,calls in shapes:
                rows.append(dict(name=q.BASE+q.SUFFIXES[family]+f'[SIMD{simd} {{{group}; 1; 1}} {{{local}; 1; 1}}]',
                    calls=calls,total_ns=calls*100,min_ns=90,max_ns=110))
        return dict(rows=rows)

    def test_two_kv_heads_are_inside_each_launch_grid(self):
        result = q.reconcile(self.fixture(),117488)
        self.assertEqual(result['attention']['calls'],180)
        self.assertEqual(result['pack']['calls'],180)
        self.assertEqual(result['attention']['query_KV_head_workgroups'],117488*12*2)
        self.assertEqual(result['attention']['query_rows_across_12_layers'],117488*12)
        self.assertEqual(result['attention']['geometries'],2)
        self.assertEqual(result['pack']['geometries'],15)

    def test_unrelated_operators_do_not_change_selected_family_service(self):
        table = self.fixture();expected=q.reconcile(table,117488)
        table['rows'].append(dict(name='unrelated DenseKernel',calls=999,total_ns=999999))
        self.assertEqual(q.reconcile(table,117488),expected)

    def test_missing_duplicate_and_count_preserving_redistribution_fail(self):
        for variant in ('missing','duplicate','redistribute','count'):
            table=self.fixture()
            if variant=='missing':table['rows'].pop()
            if variant=='duplicate':table['rows'].append(dict(table['rows'][0]))
            if variant=='redistribute':table['rows'][0]['calls']-=1;table['rows'][1]['calls']+=1
            if variant=='count':table['rows'][0]['calls']*=2
            with self.subTest(variant=variant),self.assertRaises(ValueError):q.reconcile(table,117488)

    def test_specialization_lambda_simd_local_and_grid_fail_closed(self):
        for before,after in [('true, true, true, false>','true, true, true, true>'),
            ('handler&)#1','handler&)#2'),('SIMD16','SIMD32'),('{128; 1; 1}','{64; 1; 1}'),
            ('{16384; 1; 1}','{8192; 1; 1}')]:
            table=self.fixture();table['rows'][0]['name']=table['rows'][0]['name'].replace(before,after)
            with self.subTest(before=before),self.assertRaises(ValueError):q.reconcile(table,117488)

    def test_row_extent_and_physical_boundary_are_source_arithmetic_only(self):
        for rows in (True,0,8192,262145):
            with self.subTest(rows=rows),self.assertRaises(ValueError):q.geometry(rows)
        g=q.geometry(262144)
        self.assertEqual(g['attention'],{16384:384})
        self.assertEqual(max(g['pack']),524288)
        self.assertEqual(sum(g['pack'].values()),384)


if __name__=='__main__':unittest.main()
