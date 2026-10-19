#!/usr/bin/env python3
"""Offline rejection checks for actual cache-loan admission evidence."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import cache_borrow as c


def log():
    result=[]
    for chunk,first,bytes_,need in ((8192,160,2300000000,2200000000),(256,360,1900000000,1800000000)):
        lent=1280-first
        result += [f'strata loan audit: phase=layout chunk={chunk} first={first} slots=1280 bytes={bytes_} need={need} lent={lent} mask_match=1',
            'strata loan audit: phase=lend table_entries=24576 table_match=1 free_bytes=100000000 total_bytes=11000000000',
            f'strata loan audit: phase=payload rows=1280 bytes=2639104000 lent={lent} same_slots=1 payload_match=1',
            'strata loan audit: phase=refill table_entries=24576 table_match=1 free_bytes=100000000 total_bytes=11000000000']
    result += ['strata mtp: prompt expert phase='+p+' free_bytes=100000000 total_bytes=11000000000'
               for p in ('before-detach','after-detach','after-prompt-before-restore','after-restore')]
    return '\n'.join(result)+'\n'


class LoanEvidence(unittest.TestCase):
    def test_complete(self):
        result=c.loan_evidence(log())
        self.assertEqual([x['chunk'] for x in result['layouts']],[8192,256])
        self.assertEqual(result['payloads'][0]['rows'],1280)

    def test_missing_duplicate_or_malformed(self):
        for value in (log().split('\n',1)[1],log()+log(),log()+'strata loan audit: malformed\n'):
            with self.subTest(value=value),self.assertRaises(RuntimeError):c.loan_evidence(value)

    def test_bad_snapshot_or_tail(self):
        for old,new in (('table_match=1','table_match=0'),('same_slots=1','same_slots=0'),
                        ('payload_match=1','payload_match=0'),('need=2200000000','need=2400000000'),
                        ('lent=1120','lent=1119'),('rows=1280','rows=1279'),
                        ('table_entries=24576','table_entries=12288'),
                        ('free_bytes=100000000 ','free_bytes=12000000000 ')):
            with self.subTest(old=old),self.assertRaises(RuntimeError):c.loan_evidence(log().replace(old,new,1))

    def test_missing_relayout_or_mtp_restore(self):
        for value in (log().replace('chunk=256','chunk=8192'),
                      log().replace('first=360','first=160'),
                      log().replace('phase=after-restore','phase=wrong')):
            with self.subTest(value=value),self.assertRaises(RuntimeError):c.loan_evidence(value)


class RefusalClosure(unittest.TestCase):
    def test_explicit_refusal_exits_without_signals_or_second_request(self):
        script="import sys;print('READY 131072 stop',flush=True);assert sys.stdin.readline().startswith('GEN 16 ');print('ERR injected configuration refusal',flush=True);sys.exit(1)"
        record=dict(small_requests=[],model_route_executed=False)
        with tempfile.TemporaryDirectory(prefix='strata-loan-protocol-') as directory:
            owner=c.supervision.Owner(Path(directory)/'owned')
            entry,_,_=owner.run('refusal',[sys.executable,'-c',script],
                dict(PATH='/usr/bin:/bin',PYTHONDONTWRITEBYTECODE='1'),c.P,
                interaction=lambda io,e:c.interact_requests(io,e,record,[[1,2,3],[4,5]]),
                wall=10,cpu=5,rss_cap=128<<20,total_cap=1<<20,text_cap=64<<10,file_cap=64<<10)
            self.assertTrue(c.supervision.closed(entry),entry)
            self.assertEqual(entry['exit_code'],1)
            self.assertEqual(entry['cleanup'],[])
            self.assertIsNone(owner.active)
            self.assertEqual(record['native_error'],'ERR injected configuration refusal')
            self.assertEqual(record['small_requests'],[])
            self.assertFalse(record['model_route_executed'])


if __name__=='__main__':unittest.main()
