"""Production-pool evidence must reject missing work and component regressions."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import cpu_iq2 as subject


def matrix(ratios):
    rows=[]
    for p in range(3):
        for arm in range(2):
            for down in (20,42):
                for mix in range(2):
                    n,entries,ones=(24,36,16) if mix else (28,35,22)
                    for access in range(2):
                        calls=2*(168//n if access else 1)
                        for rep in range(6):
                            wall=.5*(ratios[down] if arm else 1)
                            rows.append(dict(process=p,arm=arm,down=down,mixture=mix,access=access,rep=rep,
                                passes=2,batch_calls=calls,wall_s=wall,cpu_s=wall*5,
                                GU_s=wall*.6,quant_s=wall*.05,Down_s=wall*.3,
                                row_ops=calls*640*entries,NT1_rows=calls*640*ones,
                                paired_rows=calls*640*ones if arm else 0,min_thread_rows=640,
                                timer_s=1e-8,warm_pass_s=.01))
    return rows


class Evidence(unittest.TestCase):
    def test_formats_are_selected_independently(self):
        # Their combined ratio would pass0.97, but Down42 must still be rejected.
        decisions=subject.reduce_samples(matrix({20:.90,42:1.04}))
        self.assertEqual([d['performance_gate_passed']for d in decisions],[True,False])
        self.assertAlmostEqual(decisions[0]['geomean'],.9)

    def test_complete_unique_matrix_required(self):
        rows=matrix({20:.9,42:.9})
        for bad in (rows[:-1],rows[:-1]+[copy.deepcopy(rows[0])]):
            with self.assertRaises(RuntimeError):subject.reduce_samples(bad)

    def test_dispatch_and_all_threads_required(self):
        for field,value in (('min_thread_rows',0),('paired_rows',1),('row_ops',1),('NT1_rows',1)):
            rows=matrix({20:.9,42:.9});rows[0][field]=value
            with self.subTest(field=field),self.assertRaises(RuntimeError):subject.reduce_samples(rows)

    def test_floor_and_nested_clock_gate(self):
        for field,value in (('wall_s',.24),('timer_s',.001),('GU_s',1)):
            rows=matrix({20:.9,42:.9});rows[0][field]=value
            with self.subTest(field=field),self.assertRaises(RuntimeError):subject.reduce_samples(rows)

    def test_bad_cell_cannot_hide_in_geomean(self):
        rows=matrix({20:.9,42:.9})
        for s in rows:
            if (s['process'],s['arm'],s['down'],s['mixture'],s['access'])==(0,1,20,0,0):
                s['wall_s']=.5*1.03
        self.assertFalse(subject.reduce_samples(rows)[0]['performance_gate_passed'])

    def test_bad_independent_pair_cannot_hide(self):
        rows=matrix({20:.9,42:.9})
        for s in rows:
            if (s['process'],s['arm'],s['down'])==(0,1,20):s['wall_s']=.5*1.001
        self.assertFalse(subject.reduce_samples(rows)[0]['performance_gate_passed'])


if __name__=='__main__':unittest.main()
