"""Exercise the prospective barrier timing admission; no native/GPU execution."""
import copy
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import check_qsa_barrier as c


def observations():
    result=[]
    for process in range(3):
        rows=[]
        for queries in (512,6491):
            for rnd in range(6):
                for offset in range(3):
                    arm=(process+rnd+offset)%3;ns=196000000 if arm==2 else 200000000
                    repeat=32 if queries==512 else 2
                    rows.append(dict(kind='sample',queries=queries,process=process,round=rnd,
                        arm=arm,repeats=repeat,batch_ns=ns,attention_device_ns=ns-1000000 if arm else None,
                        event_calls=repeat if arm else 0,timer_floor_ns=10))
        result.append(dict(samples=c.parse_samples(rows,process)))
    return result


class TimingAdmission(unittest.TestCase):
    def setUp(self):self.data=observations()

    def refused(self,rows):
        with self.assertRaises(RuntimeError):c.parse_samples(rows,0)

    def test_complete_grid(self):
        self.assertTrue(c.performance(self.data)['advance_to_model'])

    def test_missing_sample(self):self.refused(self.data[0]['samples'][:-1])

    def test_subfloor_batch(self):
        rows=copy.deepcopy(self.data[0]['samples']);rows[0]['batch_ns']=99999999
        self.refused(rows)

    def test_invented_original_event(self):
        rows=copy.deepcopy(self.data[0]['samples']);rows[0]['attention_device_ns']=1
        self.refused(rows)

    def test_changed_order(self):
        rows=copy.deepcopy(self.data[0]['samples']);rows[0],rows[1]=rows[1],rows[0]
        self.refused(rows)

    def test_one_process_service_regression(self):
        for row in self.data[2]['samples']:
            if row['arm']==2:row['attention_device_ns']=199000000
        self.assertFalse(c.performance(self.data)['advance_to_model'])

    def test_namespace_control_drift(self):
        for row in self.data[1]['samples']:
            if row['arm']==1:row['batch_ns']=204000000
        self.assertFalse(c.performance(self.data)['advance_to_model'])


if __name__=='__main__':unittest.main()
