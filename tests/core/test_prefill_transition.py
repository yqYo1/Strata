#!/usr/bin/env python3
"""Reject incomplete, double-counted or malformed transition attribution."""
import copy
from pathlib import Path
import sys
import unittest
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import prefill_transition as t
class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.header='strata prefill transition: v=1 scope=dense_restore available=1 complete=1 wall_s=1.0 thread_cpu_s=0.2 wait_wall_s=0.7 wait_cpu_s=0.1 other_wall_s=0.2 other_cpu_s=0.05 open_parts=0 wait_errors=0 parts_disjoint=1 waits_nested=1 units=charged_wrapper_intervals logical_bytes_not_bus_traffic=1'
        self.part='strata prefill transition part: v=1 scope=dense_restore kind=submit calls=1 items=300 logical_bytes=2116951040 throws=0 wall_s=0.8 thread_cpu_s=0.15 wait_wall_s=0.7 wait_cpu_s=0.1 outside_wait_wall_s=0.1 outside_wait_cpu_s=0.05'
    def test_disjoint_parts_keep_nested_waits_and_payload_distinct(self):
        rows=t.profiles(self.header+'\n'+self.part)
        self.assertEqual(rows[0]['parts']['submit']['items'],300)
        self.assertEqual(rows[0]['wall_s'],1.0)
        self.assertEqual(rows[0]['other_wall_s'],0.2)
        self.assertEqual(rows[0]['parts']['submit']['logical_bytes'],2116951040)
        self.assertEqual(t.profiles('unrelated log'),[])
    def test_refuse_incomplete_errors_and_nonfinite_clocks(self):
        for a,b in [('complete=1','complete=0'),('available=1','available=0'),('open_parts=0','open_parts=1'),('wait_errors=0','wait_errors=1'),('wall_s=1.0','wall_s=nan'),('cpu_s=0.2','cpu_s=-1')]:
            with self.subTest(field=a),self.assertRaises((RuntimeError,ValueError)):
                t.profiles(self.header.replace(a,b)+'\n'+self.part)
    def test_refuse_missing_duplicate_or_orphan_parts(self):
        for text in [self.header,self.part,self.header+'\n'+self.part+'\n'+self.part,self.header+' v=1\n'+self.part,self.header+'\n'+self.part.replace('scope=dense_restore','scope=other')]:
            with self.subTest(text=text),self.assertRaises(RuntimeError):t.profiles(text)
    def test_refuse_overlapping_parts_bad_waits_or_error_counter(self):
        for a,b in [('other_wall_s=0.2','other_wall_s=0.3'),('wait_wall_s=0.7','wait_wall_s=0.9'),('outside_wait_wall_s=0.1','outside_wait_wall_s=0.2'),('throws=0','throws=1'),('items=300','items=-1')]:
            text=(self.header+'\n'+self.part).replace(a,b)
            with self.subTest(field=a),self.assertRaises(RuntimeError):t.profiles(text)
if __name__=='__main__':unittest.main()
