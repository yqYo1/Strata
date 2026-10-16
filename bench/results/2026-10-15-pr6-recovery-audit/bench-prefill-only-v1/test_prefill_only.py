"""Proposed host-only checks. NOT RUN by author. Main runs after native retirement.
Reconstruct in memory from pinned root + source.diff; no product copies or writes.
"""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'tools'))

def proposed(name):
    pins=json.loads((HERE/'pins.json').read_text())[name]
    original=(ROOT/name).read_text()
    if hashlib.sha256(original.encode()).hexdigest()!=pins['base_sha256']:
        raise RuntimeError('root base differs; reconstruct fixture from pinned base')
    lines=original.splitlines(True);delta=(HERE/'source.diff').read_text().splitlines(True)
    idx=delta.index('+++ b/'+name+'\n')+1;pos=0;result=[]
    while idx<len(delta) and not delta[idx].startswith('--- '):
        m=re.match(r'@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@',delta[idx]);idx+=1
        if not m:raise RuntimeError('bad patch hunk')
        start=max(0,int(m[1])-1);result+=lines[pos:start];pos=start
        while idx<len(delta) and not delta[idx].startswith(('@@ ','--- ')):
            line=delta[idx];idx+=1
            if line[0] in ' -':
                if lines[pos]!=line[1:]:raise RuntimeError('patch context mismatch')
                pos+=1
            if line[0] in ' +':result.append(line[1:])
    result+=lines[pos:];text=''.join(result)
    if hashlib.sha256(text.encode()).hexdigest()!=pins['proposed_sha256']:raise RuntimeError('output hash mismatch')
    return text

bench=types.ModuleType('prefill_proposed');bench.__file__=str(ROOT/'tools/strata_bench.py')
exec(compile(proposed('tools/strata_bench.py'),bench.__file__,'exec'),bench.__dict__)

def raw(n=70000,read=70000,reuse=0):
    return dict(first=.1,total=2.,text='',reasoning_text='thinking',err=None,done_received=True,
                finish_reason='length',request_sha256='fixture',usage=dict(prompt_tokens=n,completion_tokens=24,
                prompt_tokens_details=dict(cached_tokens=reuse)),timings=dict(prompt_n=read,cache_n=reuse,prompt_per_second=600.))

class PrefillOnlyTests(unittest.TestCase):
    def suite(self,**kw):
        return bench.Suite('http://unused','',262144,3,False,log=lambda *a:None,workload_id='pair',**kw)

    def test_default_preserves_suite_order(self):
        s=self.suite();calls=[]
        with patch.object(s,'warm_up',side_effect=lambda:calls.append('warm')),patch.object(s,'run_chat',side_effect=lambda:calls.append('chat')),patch.object(s,'run_prompt',side_effect=lambda n,label:calls.append(label)),patch.object(s,'run_agent',side_effect=lambda:calls.append('agent')):
            s.run()
        self.assertEqual(calls,['warm','chat','4K','16K','agent'])
        self.assertFalse(s.think);self.assertFalse(s.fail_fast_engine_error)

    def test_fixed_body_freshness_and_scope(self):
        s=self.suite(prefill_only=True);other=self.suite(prefill_only=True)
        other.tokens_per_word=9;other.overhead=9000
        different=self.suite(prefill_only=True);different.nonce='other'
        self.assertNotEqual(s.prefill_prompt(0),different.prefill_prompt(0))
        self.assertEqual([s.prefill_prompt(i) for i in range(3)],[other.prefill_prompt(i) for i in range(3)])
        bodies=[]
        def fake(base,key,messages,cap,think,**kw):
            body=bench.chat_body(messages,cap,think);bodies.append(json.dumps(body));self.assertEqual(cap,24)
            self.assertTrue(body['chat_template_kwargs']['enable_thinking'])
            response=raw();response['request_sha256']=hashlib.sha256(json.dumps(body).encode()).hexdigest();return response
        with patch.object(bench,'chat',side_effect=fake):out=s.run()
        self.assertEqual(len(bodies),4);self.assertEqual(len(set(bodies)),4)
        self.assertEqual(out['prefill_only']['eligible_runs'],3)
        self.assertEqual(len(out['prompts']),1);self.assertFalse(out['chat']);self.assertFalse(out['agent']);self.assertFalse(out['clients'])
        self.assertEqual([x['phase'] for x in out['request_samples']],['warm-up/chat']+['prefill-only']*3)

    def test_actual_bounds_missing_read_reuse_and_output_fail(self):
        n=raw(32089,32089)
        self.assertFalse(bench.prefill_eligibility(n,bench.digest(n),32768,262144)['eligible'])
        for r in (raw(32089,32089),raw(70000,65535),raw(70000,70001),raw(reuse=1)):
            self.assertFalse(bench.prefill_eligibility(r,bench.digest(r),65536,262144)['eligible'])
        for which in ('read','reuse','generated'):
            r=raw()
            if which=='read':del r['timings']['prompt_n']
            if which=='reuse':del r['timings']['cache_n'];del r['usage']['prompt_tokens_details']
            if which=='generated':r['usage']['completion_tokens']=25
            check=bench.prefill_eligibility(r,bench.digest(r),65536,262144)
            self.assertFalse(check['eligible'])
            if which=='read':self.assertIsNone(check['read_tokens'])
        s=self.suite(prefill_only=True)
        with patch.object(bench,'chat',side_effect=[raw(),raw(65000,65000)]) as http:
            with self.assertRaises(bench.EngineRequestError):s.run()
        self.assertEqual(http.call_count,2);self.assertFalse(s.out['prefill_only']['complete'])
        self.assertEqual(len(s.out['prefill_only']['requests']),1)
        s=self.suite(prefill_only=True);s.ctx=65536
        with patch.object(bench,'chat') as http:
            with self.assertRaises(bench.EngineRequestError):s.run()
            http.assert_not_called()

    def test_arguments_rejected_before_server(self):
        bad=[['--prefill-tokens','32768'],['--prefill-only','--clients','2'],['--prefill-only','--long'],['--prefill-only','--long-generation'],['--prefill-only','--long-generation-only'],['--prefill-only','--prefill-tokens','32089'],['--prefill-only','--runs','0']]
        with patch.object(bench,'is_strata') as http,patch.object(bench,'start_server') as start:
            for argv in bad:
                with self.subTest(argv=argv),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as caught:bench.main(argv)
                self.assertEqual(caught.exception.code,2)
            http.assert_not_called();start.assert_not_called()

    def test_cli_report_journal_success_failure_interrupt(self):
        for suffix in ('success','failure','interrupt'):
            with self.subTest(suffix=suffix),tempfile.TemporaryDirectory() as td:
                replies=[raw(),raw(),raw(),raw()]
                if suffix=='failure':replies[2]=raw(reuse=1)
                if suffix=='interrupt':replies[2]=KeyboardInterrupt()
                with patch.object(bench,'is_strata',return_value=True),patch.object(bench,'collect_machine',return_value={'context':262144,'model':'host fixture'}),patch.object(bench,'chat',side_effect=replies),patch.object(bench.Suite,'agent_files',side_effect=AssertionError('unused agent read')):
                    if suffix=='interrupt':
                        with self.assertRaises(KeyboardInterrupt):bench.main(['--url','http://unused','--out',td,'--prefill-only','--workload-id','pair','--quiet'])
                    else:self.assertEqual(bench.main(['--url','http://unused','--out',td,'--prefill-only','--workload-id','pair','--quiet']),0 if suffix=='success' else 1)
                d=next(Path(td).iterdir());res=json.loads((d/'results.json').read_text());report=(d/'README.md').read_text()
                self.assertEqual(res['params']['mode'],'prefill-only');self.assertTrue(res['params']['think'])
                self.assertIn('65536',report);self.assertEqual(res['complete'],suffix=='success')
                self.assertTrue(bench.verify_request_journal(d)['valid'])
                state=json.loads((d/'journal-state.json').read_text());self.assertEqual(state['complete'],suffix=='success')
                self.assertEqual(len(res['results']['prefill_only']['requests']),3 if suffix=='success' else 2 if suffix=='failure' else 1)
                self.assertIn('reasoning_observed',res['results']['prefill_only']['requests'][0])
                bench.summary_metrics({'prompts':{}});bench.results_table({})

if __name__=='__main__':unittest.main()
