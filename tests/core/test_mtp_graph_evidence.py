#!/usr/bin/env python3
"""Refusal gates for graph support; no device access."""
import sys
from pathlib import Path
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
from check_mtp_observer import evidence

CONTROL = ('graph envelope: profile=0 numerical=PASS guards=PASS aliases=PASS '
    'checked_teardown=PASS support=NOT_ENABLED arithmetic_service=UNMEASURED hardware_saturation=UNMEASURED\n')
PROFILE = (
    'graph sample: replay=0 status=0 fields=7 submit=90 start=100 end=200 timer_resolution=1 query_ns=10 envelope_qualified=1 queue_delay_qualified=1\n'
    'graph sample: replay=1 status=0 fields=7 submit=250 start=300 end=400 timer_resolution=1 query_ns=10 envelope_qualified=1 queue_delay_qualified=1\n'
    'graph envelope: profile=1 numerical=PASS guards=PASS aliases=PASS checked_teardown=PASS support=SUPPORTED arithmetic_service=UNMEASURED hardware_saturation=UNMEASURED\n')


class SupportEvidenceTests(unittest.TestCase):
    def test_both_ordinary_and_profiled_contracts(self):
        self.assertTrue(evidence(CONTROL,0,0)['passed'])
        self.assertEqual(len(evidence(PROFILE,1,0)['samples']),2)

    def test_unknown_never_becomes_a_pass(self):
        row=evidence('graph envelope: profile=1 support=UNKNOWN execution=NOT_RUN status=801\n',1,77)
        self.assertFalse(row['passed'])
        with self.assertRaises(RuntimeError): evidence(PROFILE,1,77)

    def test_missing_replay_or_failed_guard_rejects_support(self):
        for value in (PROFILE.split('\n',1)[1],PROFILE.replace('guards=PASS','guards=FAIL'),PROFILE+PROFILE):
            with self.subTest(value=value),self.assertRaises(RuntimeError): evidence(value,1,0)

    def test_bad_partial_slow_or_stale_clocks_are_rejected(self):
        for old,new in (('fields=7','fields=3'),('end=200','end=100'),('query_ns=10','query_ns=1000000000'),
                ('start=300','start=150'),('status=0','status=1'),('envelope_qualified=1','envelope_qualified=0'),
                ('replay=1','replay=0'),('timer_resolution=1','timer_resolution=1000')):
            with self.subTest(old=old),self.assertRaises(RuntimeError): evidence(PROFILE.replace(old,new,1),1,0)

    def test_submit_inversion_only_invalidates_queue_delay(self):
        value=PROFILE.replace('submit=90','submit=150').replace('queue_delay_qualified=1','queue_delay_qualified=0',1)
        self.assertTrue(evidence(value,1,0)['passed'])
        with self.assertRaises(RuntimeError): evidence(value.replace('queue_delay_qualified=0','queue_delay_qualified=1'),1,0)

    def test_ordinary_graph_cannot_claim_profiled_samples(self):
        with self.assertRaises(RuntimeError): evidence(PROFILE.split('\n',1)[0]+'\n'+CONTROL,0,0)


if __name__ == '__main__': unittest.main()
