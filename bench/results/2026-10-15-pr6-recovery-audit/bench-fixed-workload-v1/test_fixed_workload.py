#!/usr/bin/env python3
"""Host-only source tests, NOT_RUN by author; no real HTTP, server or model."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
sys.path.insert(0, str(PROJECT/'tools'))
spec = importlib.util.spec_from_file_location('proposed_fixed_workload', HERE/'proposed/tools/strata_bench.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
# Proposal is stored outside the repo tools path; match its eventual installed ROOT.
bench.ROOT = PROJECT


class FixedWorkloadTests(unittest.TestCase):
    def suite(self, name):
        return bench.Suite('http://unused.invalid', '', 262144, 1, True,
                           log=lambda *a: None, workload_id=name)

    def bodies(self, name):
        rows = []
        def fake_chat(base, key, messages, max_tokens, think=False, **kwargs):
            payload = bench.chat_body(messages,max_tokens,think,kwargs.get('tools'),kwargs.get('request_overrides'))
            encoded = json.dumps(payload).encode()
            rows.append(encoded)
            count = sum(len(m['content'].split()) for m in messages)+10 if max_tokens==1 else 40000
            return dict(text='x'*600, reasoning_text='', total=2., first=.1,
                        done_received=True,finish_reason='stop',err=None,
                        usage=dict(prompt_tokens=count,completion_tokens=1100,prompt_tokens_details=dict(cached_tokens=0)),
                        timings=dict(predicted_per_second=550.,predicted_ms=2000.,draft_n=512,draft_n_accepted=256),
                        request_sha256=hashlib.sha256(encoded).hexdigest())
        suite=self.suite(name)
        with patch.object(bench,'chat',side_effect=fake_chat):
            suite.calibrate()
            suite.run_prompt(32768,'32K')
            suite.run_long_generation()
        return suite,rows

    def test_fixed_tags_and_actual_calibration_prompt_longgen_bodies(self):
        a,one=self.bodies('paired_20261015-v1')
        b,two=self.bodies('paired_20261015-v1')
        self.assertEqual(a.tag('cal',200),b.tag('cal',200))
        self.assertEqual(len(one),5)  # two calibration,32K,two long-generation bodies
        self.assertEqual(one,two)
        self.assertEqual(a.tokens_per_word,b.tokens_per_word)
        self.assertEqual(a.overhead,b.overhead)
        self.assertEqual(a.workload_identity(),dict(workload_id='paired_20261015-v1',nonce='paired_20261015-v1',nonce_policy='fixed-id'))

    def test_different_id_and_constructor_validation(self):
        a,b=self.suite('pair_A'),self.suite('pair_B')
        self.assertNotEqual(a.tag(0,'p'),b.tag(0,'p'))
        self.assertNotEqual(a.prompt_for(32768,0),b.prompt_for(32768,0))
        self.assertNotEqual(a.workload_identity(),b.workload_identity())
        for bad in ('','x'*65,'space id','a/b','a\nb','日本語'):
            with self.subTest(bad=bad),self.assertRaises(bench.argparse.ArgumentTypeError):self.suite(bad)
        for good in ('a','Z_9-','x'*64):self.assertEqual(self.suite(good).nonce,good)

    def test_legacy_positional_constructor_and_default_timestamp(self):
        before=int(time.time())
        old=bench.Suite('unused','',262144,3,False,lambda *a:None,0,False,False,False,False,None,False)
        after=int(time.time())
        self.assertTrue(before<=int(old.nonce)<=after)
        self.assertEqual(old.workload_identity(),dict(workload_id=None,nonce=old.nonce,nonce_policy='timestamp-seconds'))
        self.assertTrue(old.tag('old').endswith('-'+old.nonce))

    def test_invalid_cli_rejected_before_any_http_or_engine(self):
        with patch.object(bench,'is_strata') as http,patch.object(bench,'start_server') as engine:
            for bad in ('','x'*65,'bad value','a/b','日本語','a\nb'):
                with self.subTest(bad=bad),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as caught:
                    bench.main(['--url','http://unused.invalid','--workload-id',bad])
                self.assertEqual(caught.exception.code,2)
            http.assert_not_called();engine.assert_not_called()

    def cli_identity(self, extra):
        saved=[]
        def writer(res,parent,red,log_lines,directory=None):saved.append(copy.deepcopy(res));return directory
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(bench,'is_strata',return_value=True), \
             patch.object(bench,'start_server',side_effect=AssertionError('no engine')), \
             patch.object(bench,'collect_machine',return_value=dict(context=262144,model='test',quant='test',gpu='host',parallel_slots=1)), \
             patch.object(bench.Suite,'run',lambda self:self.out), \
             patch.object(bench.Suite,'agent_files',return_value=[]), \
             patch.object(bench,'write_results',side_effect=writer), \
             patch.object(bench,'short_table',return_value=''):
            self.assertEqual(bench.main(['--url','http://unused.invalid','--out',tmp,'--quiet',*extra]),0)
            journals=list(Path(tmp).glob('*/requests.jsonl'))
            self.assertEqual(len(journals),1)
            identity=json.loads(journals[0].read_text().splitlines()[0])
            self.assertEqual(identity['kind'],'identity')
            self.assertTrue(bench.verify_request_journal(journals[0].parent)['complete'])
        return saved[0],identity

    def test_cli_params_comparison_and_journal_record_actual_nonce(self):
        for extra in ([],['--workload-id','pair_fixed-v1']):
            with self.subTest(extra=extra):
                result,identity=self.cli_identity(extra)
                record=result['params']['workload']
                self.assertEqual(record,identity['workload'])
                self.assertEqual(record,result['comparison_identity']['workload'])
                self.assertEqual(record['nonce_policy'],'fixed-id' if extra else 'timestamp-seconds')
                if extra:
                    self.assertEqual(record['nonce'],'pair_fixed-v1')
                    self.assertIn('--workload-id pair_fixed-v1',result['params']['command'])
                else:self.assertTrue(record['nonce'].isdigit())

    def test_comparison_rejects_different_or_missing_actual_nonce(self):
        base=dict(results={},machine={},complete=True,comparison_identity=dict(workload=self.suite('same').workload_identity()))
        other=copy.deepcopy(base)
        with patch.object(bench,'summary_metrics',return_value={'metric':{'median':10.}}):
            self.assertNotIn('not comparable',bench.compare_table(base,other))
            other['comparison_identity']['workload']=self.suite('different').workload_identity()
            self.assertIn('not comparable',bench.compare_table(base,other))
            old=copy.deepcopy(base);old['comparison_identity']={'client':'legacy'}
            self.assertIn('missing actual nonce/workload policy',bench.compare_table(old,old))


if __name__=='__main__':unittest.main()
