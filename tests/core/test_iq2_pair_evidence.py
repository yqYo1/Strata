"""Actual IQ2 dispatch evidence must reject misbound or incomplete counters."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import generation_engine as subject


def snapshot(multiplier, arm):
    return [dict(tid=100+i,cpu=i,pinned=1,calls=2*multiplier,row_ops=640*multiplier,
        nt1_rows=320*multiplier,pair_rows=320*multiplier*arm)for i in range(6)]


def render(request,before,after):
    lines=[]
    for stage,rows in [('begin',before),('end',after)]:
        lines.append(f'strata IQ2_GU_AUDIT stage={stage} request={request} enabled=1 overflow=0 threads={len(rows)}')
        for slot,row in enumerate(rows):
            body=' '.join(f'{k}={row[k]}'for k in ('tid','calls','row_ops','nt1_rows','pair_rows','cpu','pinned'))
            lines.append(f'strata IQ2_GU_THREAD stage={stage} request={request} slot={slot} '+body)
    lines.append('strata serve: prompt 33214 tokens = 0 reused + 33214 read, 3611 generated, drafts accepted 2578 of 3102, done')
    return '\n'.join(lines)+'\n'


class Evidence(unittest.TestCase):
    def setUp(self):
        self.sample=dict(prompt_tokens=33214,reused_tokens=0,read_tokens=33214,generated_tokens=3611)

    def parse(self,text,arm=1,samples=None):
        return subject.iq2_pair_evidence(text,samples or [self.sample],arm,list(range(6)))

    def test_new_threads_and_both_arms(self):
        for arm in (0,1):
            rows=self.parse(render(1,[],snapshot(1,arm)),arm)
            self.assertEqual(rows[0]['row_ops'],3840)
            self.assertEqual(rows[0]['nt1_rows'],1920)
            self.assertEqual(rows[0]['pair_rows'],1920*arm)
            self.assertEqual(rows[0]['active_threads'],6)

    def test_cumulative_threads_are_not_request_counters(self):
        first,second=snapshot(2,1),snapshot(5,1)
        rows=self.parse(render(1,[],first)+render(2,first,second),samples=[self.sample]*2)
        self.assertEqual([r['row_ops']for r in rows],[7680,11520])

    def test_refuse_missing_extra_malformed_and_wrong_request(self):
        text=render(1,[],snapshot(1,1))
        for bad in [text.replace('request=1','request=2'),text.replace('overflow=0','overflow=1'),
                    text.replace('enabled=1','enabled=0'),text.replace('threads=6','threads=65'),
                    '\n'.join(text.splitlines()[:-2]+text.splitlines()[-1:]),
                    text+text.splitlines()[0]+'\n',text.replace('slot=1','slot=0'),
                    text.replace('IQ2_GU_THREAD','IQ2_GU_UNKNOWN')]:
            with self.subTest(bad=bad[:80]),self.assertRaises(RuntimeError):self.parse(bad)

    def test_refuse_counter_regression_and_identity_change(self):
        for key,value in [('calls',0),('row_ops',0),('nt1_rows',0),('pair_rows',0),
                          ('tid',999),('cpu',9),('pinned',0)]:
            before=snapshot(1,1);after=snapshot(2,1);after[0][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.parse(render(1,before,after))

    def test_refuse_inactive_unpinned_duplicate_and_wrong_dispatch(self):
        for key,value in [('tid',101),('cpu',1),('pinned',0),('pair_rows',0),('pair_rows',641),
                          ('nt1_rows',641),('calls',0),('row_ops',(1<<64))]:
            after=snapshot(1,1);after[0][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(RuntimeError):self.parse(render(1,[],after))
        with self.assertRaises(RuntimeError):self.parse(render(1,[],snapshot(1,1)),arm=0)
        after=snapshot(1,1);after[-1]=dict(after[-1],calls=0,row_ops=0,nt1_rows=0,pair_rows=0)
        with self.assertRaises(RuntimeError):self.parse(render(1,[],after))

    def test_refuse_aggregate_overflow(self):
        after=snapshot(1,1)
        for row in after:row['calls']=(1<<64)-1
        with self.assertRaises(RuntimeError):self.parse(render(1,[],after))

    def test_bind_actual_input_reuse_and_output(self):
        text=render(1,[],snapshot(1,1))
        for key in ['prompt_tokens','reused_tokens','read_tokens','generated_tokens']:
            sample=copy.deepcopy(self.sample);sample[key]+=1
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.parse(text,samples=[sample])


if __name__=='__main__':unittest.main()
