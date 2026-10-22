#!/usr/bin/env python3
"""Refuse incomplete or mislabeled current-work calibration evidence; no GPU."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import cpu_decode_mix as mix


class EvidenceTests(unittest.TestCase):
    def fixture(self):
        row=dict(blob_bytes=2<<20,observed_jobs_NT=[1,1,1,1])
        valid=dict(kind='validation',distinct_experts=66,NT_mask=30,pool_serial_bits_equal=True,
            inactive_outputs_untouched=True,canaries=True)
        lines=[dict(kind='work',workers=5,host_works=True,group=6,batches=100,
            unique_stream_experts=66,blob_bytes=2<<20,jobs_NT=[150]*4,phase='decode'),
            valid,dict(kind='placement',workers=5,host_cpu=0,worker_cpus=[1,2,3,4,5],actual_affinity_verified=True)]
        for stream in (False,True):
            lines.append(dict(kind='timer',stream=stream,query_s=1e-8,fastest_pass_s=.2))
            lines.extend(dict(kind='sample',stream=stream,rep=i,passes=7,wall_s=1.4,cpu_s=7,
                gu_s=.9,quant_s=.1,down_s=.3) for i in range(5))
            lines.append(copy.deepcopy(valid))
        lines.append(dict(kind='closed',passed=True))
        return row,lines

    def test_known_histograms_and_integer_conservation(self):
        self.assertEqual(mix.expected_histogram([1,1,1,1],6),[150]*4)
        self.assertEqual(mix.expected_histogram([0,0,0,7],28),[0,0,0,2800])
        self.assertEqual(mix.expected_histogram([1,1,1,0],28),[934,933,933,0])

    def test_full_evidence_and_refusals(self):
        for variant in ('valid','phase','histogram','affinity','numerical','lost_sample','extra',
                        'short','zero_passes','nonfinite','overcount','timer','closure'):
            row,lines=self.fixture()
            if variant=='phase':lines[0]['phase']='prefill'
            if variant=='histogram':lines[0]['jobs_NT']=[151,149,150,150]
            if variant=='affinity':lines[2]['actual_affinity_verified']=False
            if variant=='numerical':lines[1]['pool_serial_bits_equal']=False
            if variant=='lost_sample':lines.pop(4)
            if variant=='extra':lines.insert(0,dict(kind='unknown'))
            if variant=='short':lines[4]['wall_s']=.9
            if variant=='zero_passes':lines[4]['passes']=0
            if variant=='nonfinite':lines[4]['cpu_s']=float('nan')
            if variant=='overcount':lines[4]['gu_s']=2
            if variant=='timer':lines[3]['query_s']=.1
            if variant=='closure':lines[-1]['passed']=False
            with self.subTest(variant=variant),patch.object(mix.base,'machine',return_value={'physical_cpus':list(range(6))}):
                if variant=='valid':self.assertEqual(len(mix.parse(lines,row,5,6,'measure')['samples']),10)
                else:
                    with self.assertRaises(RuntimeError):mix.parse(lines,row,5,6,'measure')


if __name__=='__main__':unittest.main()
