#!/usr/bin/env python3
"""Reject partial MTP host accounting before any real model interpretation."""
import sys
from pathlib import Path
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
from mtp_model_observer import observation


def host_record():
    header = ('strata MTP_OBSERVER v=1 request=5 profile=0 request_complete=1 sample_complete=1 '
        'accounting_valid=1 skip=8 limit=64 seen=100 windows=64 completed=64 failed=0 replays=192 rows=256 T_mask=16 '
        'wall_s=1.000000000 thread_cpu_s=0.500000000 other_wall_s=0.940000000 other_cpu_s=0.494000000 '
        'valid_graph_clocks=0 valid_queue_clocks=0 graph_envelope_ns=0 round_clocks=0 round_envelope_ns=0 '
        'chain_clocks=0 chain_envelope_ns=0 raw_retained=0 raw_omitted=0 scope=draft_entry_through_return '
        'clocks=HOST_DISJOINT_DEVICE_NONADDITIVE arithmetic_service=UNKNOWN hardware_saturation=UNKNOWN observer_charged=1\n')
    rows = []
    for kind,calls in (('round',64),('chain',128)):
        for phase in ('capture','submit','drain','query'):
            n = 0 if phase == 'query' else calls
            rows.append(f'strata MTP_OBSERVER_PHASE v=1 request=5 kind={kind} phase={phase} calls={n} throws=0 '
                + ('wall_s=0.010000000 thread_cpu_s=0.001000000' if n else 'wall_s=0.000000000 thread_cpu_s=0.000000000'))
    return header+'\n'.join(rows)+'\n'


class AccountingTests(unittest.TestCase):
    def test_complete_host_record_and_disabled_route(self):
        self.assertEqual(observation(host_record(),'host')['summary']['replays'],192)
        self.assertIsNone(observation('ordinary request\n','0'))
        with self.assertRaises(RuntimeError): observation(host_record(),'0')

    def test_incomplete_or_wrong_selected_work_cannot_qualify(self):
        for a,b in (('sample_complete=1','sample_complete=0'),('request=5','request=6'),
                ('failed=0','failed=1'),('T_mask=16','T_mask=4'),('windows=64','windows=63'),
                ('calls=64','calls=63'),('replays=192','replays=191'),('throws=0','throws=1')):
            with self.subTest(field=a),self.assertRaises(RuntimeError): observation(host_record().replace(a,b,1),'host')

    def test_duplicate_missing_and_nonconserving_regions_are_rejected(self):
        text = host_record()
        for bad in (text+text,text.rsplit('\n',2)[0]+'\n',text.replace('other_wall_s=0.940000000','other_wall_s=0.900000000'),
                text.replace('other_cpu_s=0.494000000','other_cpu_s=nan'),text.replace('v=1 request=5','v=1 v=1 request=5',1)):
            with self.subTest(value=bad),self.assertRaises((RuntimeError,ValueError,KeyError)): observation(bad,'host')


if __name__ == '__main__': unittest.main()
