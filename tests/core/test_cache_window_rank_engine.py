"""Require both workload gates; collection success cannot silently adopt a profile."""
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import cache_window_rank_engine as engine


def pair():
    rows=[]
    for arm in (0,1):
        rows.append(dict(passed=True,complete=True,
            requests=[dict(request_sha256=str(i)*64,read_tokens=33214+i,decode_ms=1000 if arm==0 else 900,
                generated_tokens=1000) for i in (1,2)],
            work=[dict(logical_job_blob_bytes=1000 if arm==0 else 900,rows=100,
                cpu_entries=900 if arm==0 else 800,vram_entries=100 if arm==0 else 200) for _ in range(2)]))
    return rows


class WindowRankDecisionTests(unittest.TestCase):
    def test_both_workloads_required_without_adoption(self):
        result=engine.decide(pair())
        self.assertTrue(result['advance_to_new_clean_plan'])
        self.assertFalse(result['clean_speed_qualified']);self.assertFalse(result['adoption_qualified'])

    def test_any_workload_or_matching_failure_prevents_advancement(self):
        for phase in range(2):
            for field,value in [('decode_ms',1001),('read_tokens',100),('request_sha256','f'*64)]:
                rows=pair();rows[1]['requests'][phase][field]=value
                with self.subTest(phase=phase,field=field):self.assertFalse(engine.decide(rows)['advance_to_new_clean_plan'])
            rows=pair();rows[1]['work'][phase]['logical_job_blob_bytes']=951
            self.assertFalse(engine.decide(rows)['advance_to_new_clean_plan'])
            rows=pair();rows[1]['work'][phase].update(cpu_entries=900,vram_entries=100)
            self.assertFalse(engine.decide(rows)['advance_to_new_clean_plan'])

    def test_bytes_normalize_actual_verifier_rows_and_latency_actual_generated_tokens(self):
        rows=pair();rows[1]['work'][0].update(logical_job_blob_bytes=900,rows=90)
        self.assertFalse(engine.decide(rows)['advance_to_new_clean_plan'])
        rows=pair();rows[1]['requests'][0]['generated_tokens']=800
        self.assertFalse(engine.decide(rows)['advance_to_new_clean_plan'])

    def test_incomplete_or_failed_collection_cannot_be_a_successful_pair(self):
        for key in ('passed','complete'):
            rows=pair();rows[0][key]=False
            with self.assertRaises(RuntimeError):engine.decide(rows)
        with self.assertRaises(RuntimeError):engine.decide(pair()[:1])


if __name__=='__main__':unittest.main()
