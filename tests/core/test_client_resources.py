#!/usr/bin/env python3
"""Synthetic proc/fdinfo contracts only; no real process, GPU or profiler."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'bench/sycl'))
import client_resources as r

PCI = '0000:05:00.0'


def text(busy=100, total=1000, capacity=None, client=7):
    return ('drm-driver:\txe\ndrm-pdev:\t'+PCI+'\ndrm-client-id:\t'+str(client)+
        '\ndrm-cycles-ccs:\t'+str(busy)+'\ndrm-total-cycles-ccs:\t'+str(total)+
        '\ndrm-resident-vram0:\t3 MiB\ndrm-total-gtt:\t17 KiB\n'+
        ('' if capacity is None else 'drm-engine-capacity-ccs:\t'+str(capacity)+'\n'))


def sample(busy=100, total=1000, capacity=None, stamp=1):
    return dict(target=dict(pid=101, start_ticks=50, sid=99),
        begin_monotonic_ns=stamp, end_monotonic_ns=stamp+1,
        clients=r.deduplicate_clients([(5,text(busy,total,capacity))],PCI))


class ClientResourceTests(unittest.TestCase):
    def test_units_capacity_and_unsupported_counters(self):
        row=r.parse_fdinfo(text())
        self.assertEqual(row['engines']['ccs'],dict(busy=100,total=1000,capacity=1))
        self.assertEqual(row['memory_bytes']['drm-resident-vram0'],3*1024**2)
        self.assertEqual(row['memory_bytes']['drm-total-gtt'],17*1024)
        for broken in (text(capacity=0),text(busy=-1),text()+'drm-client-id: 8\n',
                       text().replace('drm-total-cycles-ccs','other-key'),
                       text().replace('3 MiB','3 MB')):
            with self.subTest(broken=broken),self.assertRaises((ValueError,KeyError)):
                r.parse_fdinfo(broken)

    def test_duplicated_descriptors_do_not_double_count_or_take_later_counter(self):
        rows=r.deduplicate_clients([(5,text()),(8,text(110,1010)),(9,text(client=8))],PCI)
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['descriptors'],[5,8])
        self.assertEqual(rows[0]['engines']['ccs']['busy'],100)
        with self.assertRaises(ValueError):
            r.deduplicate_clients([(5,text()),(8,text(capacity=2))],PCI)
        self.assertEqual(r.deduplicate_clients([(5,text())],'0000:06:00.0'),[])

    def test_capacity_normalization_and_raw_deltas(self):
        intervals=r.CycleIntervals()
        self.assertFalse(intervals.update(sample(capacity=2))[0]['eligible'])
        row=intervals.update(sample(350,1500,capacity=2,stamp=10))[0]
        self.assertTrue(row['eligible'])
        self.assertEqual(row['busy_cycles'],250)
        self.assertEqual(row['total_cycles'],500)
        self.assertEqual(row['busy_fraction'],.25)

    def test_rollback_keeps_original_high_water_and_recovery_span(self):
        intervals=r.CycleIntervals();intervals.update(sample())
        row=intervals.update(sample(90,1100,stamp=10))[0]
        self.assertFalse(row['eligible']);self.assertNotIn('busy_fraction',row)
        row=intervals.update(sample(120,1200,stamp=20))[0]
        self.assertTrue(row['eligible']);self.assertEqual(row['busy_fraction'],.1)
        self.assertEqual(row['span_begin']['begin_monotonic_ns'],1)
        self.assertEqual(row['recovered_samples'],1)

    def test_over_capacity_capacity_change_and_absent_client_are_not_idle(self):
        intervals=r.CycleIntervals();intervals.update(sample())
        row=intervals.update(sample(300,1100,stamp=10))[0]
        self.assertFalse(row['eligible']);self.assertNotIn('busy_fraction',row)
        row=intervals.update(sample(400,1200,capacity=2,stamp=20))[0]
        self.assertFalse(row['eligible']);self.assertEqual(row['reason'],'capacity_changed')
        absent=sample();absent['clients']=[]
        self.assertEqual(intervals.update(absent),[])
        self.assertEqual(intervals.update(sample(500,1400))[0]['reason'],'baseline')

    def test_delayed_busy_accounting_recovers_over_whole_valid_envelope(self):
        intervals=r.CycleIntervals();intervals.update(sample())
        for stamp,busy,total in ((10,300,1100),(20,350,1200)):
            row=intervals.update(sample(busy,total,stamp=stamp))[0]
            self.assertFalse(row['eligible']);self.assertNotIn('busy_fraction',row)
        row=intervals.update(sample(400,1400,stamp=30))[0]
        self.assertTrue(row['eligible']);self.assertEqual(row['busy_fraction'],.75)
        self.assertEqual(row['busy_cycles'],300)
        self.assertEqual(row['span_begin']['begin_monotonic_ns'],1)
        self.assertEqual(row['recovered_samples'],2)

    def test_interval_rejects_changed_process_and_duplicate_client_engine(self):
        intervals=r.CycleIntervals();first=sample();intervals.update(first)
        first['target']['start_ticks']+=1
        with self.assertRaises(ValueError): intervals.update(first)
        changed=sample();changed['clients']*=2
        with self.assertRaises(ValueError): r.CycleIntervals().update(changed)
        with self.assertRaises(ValueError): intervals.update(sample(120,1200,stamp=0))

    def test_proc_identity_is_rechecked_after_descriptor_and_thread_reads(self):
        target=dict(pid=101,start_ticks=50,sid=99)
        fields=['0']*37
        fields[0]='R';fields[3]='99';fields[11]='17';fields[12]='9';fields[19]='50';fields[36]='3'
        stat='101 (name with (parentheses)) '+' '.join(fields)+'\n'
        with tempfile.TemporaryDirectory() as directory:
            proc=Path(directory);root=proc/'101';(root/'fdinfo').mkdir(parents=True)
            (root/'task/101').mkdir(parents=True)
            (root/'stat').write_text(stat);(root/'task/101/stat').write_text(stat)
            (root/'fdinfo/5').write_text(text());(root/'fdinfo/8').write_text(text())
            row=r.snapshot(target,PCI,proc)
            self.assertEqual(row['process']['cpu_ticks'],26)
            self.assertEqual(row['threads'][0]['processor'],3)
            self.assertEqual(len(row['clients']),1)
            original=r.process_stat
            calls=0
            def reused(path):
                nonlocal calls
                result=original(path)
                if path==root/'stat':
                    calls+=1
                    if calls==2: result['start_ticks']+=1
                return result
            with patch.object(r,'process_stat',side_effect=reused),self.assertRaises(ValueError):
                r.snapshot(target,PCI,proc)
            fields[0]='Z';(root/'stat').write_text('101 (zombie) '+' '.join(fields))
            with self.assertRaises(r.TargetExited):r.snapshot(target,PCI,proc)


if __name__=='__main__':
    unittest.main()
