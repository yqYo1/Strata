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


def owner_log():
    result=[log().rstrip(),
        'strata cache loan: phase=masked generation=1 chunk=8192 first=160 lent=1120 decode_ready=0',
        'strata prefill cache binding: generation=1 parent_match=1 execution_distinct=1']
    for layer in range(48):
        result += [f'strata MoE mixed owners: phase=entry layer={layer} first=0 rows=8193 '
            'primary=0x10000 primary_bytes=41948160 region=0x30000000 region_bytes=1700000000 '
            'tail=(nil) tail_offset=1700000000 tail_bytes=0 residual=0xa0000000 '
            'residual_bytes=167792640 bo=0x96000000 queue=0x1234 direct_owners=0 previous_attention=checked_drain',
            f'strata MoE cache loan: generation=1 layout_generation=2 layer={layer} first=0 rows=8193 '
            'used=2200000000 capacity=2300000000',
            f'strata MoE mixed owners: phase=retired layer={layer} first=0 rows=8193 '
            'route_status=0 shared_HC=checked_drain callbacks=after_return']
    result += ['strata cache loan: phase=ready generation=1 chunk=8192 first=160 lent=1120 decode_ready=1',
        'strata cache loan: phase=masked generation=2 chunk=256 first=360 lent=920 decode_ready=0',
        'strata cache loan: phase=ready generation=2 chunk=256 first=360 lent=920 decode_ready=1']
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


class CheckedOwners(unittest.TestCase):
    def check(self,value):
        return c.checked_owner_evidence(value,c.loan_evidence(value))

    def test_complete(self):
        result=self.check(owner_log())
        self.assertEqual(result['generations'],[1,2])
        self.assertEqual(result['borrowed_ffn_entries'],48)
        self.assertEqual(result['checked_retirements'],48)
        self.assertEqual(result['decode_admission_transitions'],[0,1,0,1])
        self.assertEqual(result['execution_cache_substitutions_checked'],1)

    def test_missing_or_false_execution_cache_substitution(self):
        line='strata prefill cache binding: generation=1 parent_match=1 execution_distinct=1\n'
        for value in (owner_log().replace(line,''),owner_log()+line,
                owner_log().replace('parent_match=1','parent_match=0',1),
                owner_log().replace('execution_distinct=1','execution_distinct=0',1),
                owner_log().replace(line,line.replace('generation=1','generation=2'))):
            with self.subTest(value=value),self.assertRaises(RuntimeError):self.check(value)

    def test_missing_duplicate_or_malformed(self):
        value=owner_log()
        missing='strata cache loan: phase=ready generation=2 chunk=256 first=360 lent=920 decode_ready=1\n'
        for bad in (value.replace(missing,''),value+missing,value+'strata MoE cache loan: malformed\n'):
            with self.subTest(bad=bad),self.assertRaises(RuntimeError):self.check(bad)

    def test_stale_generation_or_wrong_admission(self):
        for old,new in (('generation=2 chunk=256','generation=1 chunk=256'),
                ('layout_generation=2 layer=1 ','layout_generation=3 layer=1 '),
                ('decode_ready=1','decode_ready=0'),
                ('used=2200000000','used=2400000000'),
                ('capacity=2300000000','capacity=2200000000')):
            with self.subTest(old=old),self.assertRaises(RuntimeError):self.check(owner_log().replace(old,new,1))

    def test_owned_or_incomplete_reordered_windows(self):
        value=owner_log()
        lines=value.splitlines()
        a=next(i for i,line in enumerate(lines) if 'phase=retired layer=0 ' in line)
        lines[a],lines[a+1]=lines[a+1],lines[a]
        for bad in ('\n'.join(lines)+'\n',value.replace('direct_owners=0','direct_owners=1',1),
                value.replace('phase=retired layer=47','phase=retired layer=46',1),
                value.replace('primary_bytes=41948160','primary_bytes=41948159',1)):
            with self.subTest(bad=bad),self.assertRaises(RuntimeError):self.check(bad)


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
