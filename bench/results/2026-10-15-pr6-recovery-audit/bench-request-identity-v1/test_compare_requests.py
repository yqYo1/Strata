#!/usr/bin/env python3
"""Proposed host tests using real RequestJournal; no native/GPU receipts."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import compare_requests as utility
from strata_bench import RequestJournal


def requests():
    keys=[('warm-up/chat',None,None),('calibration',200,None),('calibration',800,None),
          ('warm-up/prompt',None,None),('chat',0,0),('prompt/4K',0,None),
          ('prompt/16K',0,None),('prompt/32K',0,None)]
    keys += [('agent',0,t) for t in range(6)]
    keys += [('long-generation/regular-code',0,None),('long-generation/varied-prose',0,None)]
    return [dict(phase=p,run=r,turn=t,max_tokens=24,thinking=True,
                 request_sha256=hashlib.sha256(f'actual-body-{i}'.encode()).hexdigest())
            for i,(p,r,t) in enumerate(keys)]


class RequestIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def fixture(self,name,rows=None,identity=None,pending=False):
        root=self.root/name;root.mkdir()
        journal=RequestJournal(root,lambda text:text)
        journal.append('identity','suite',source=identity or {'client':'old'},nonce='ignored-identity-'+name)
        for row in requests() if rows is None else rows:
            values=dict(row);phase=values.pop('phase');run=values.pop('run');turn=values.pop('turn')
            index=journal.append('start',phase,run,turn,**values)
            journal.append('result',phase,run,turn,request_index=index,sample={'not_a_quality_claim':True})
        if pending:
            journal.append('start','pending',0,None,max_tokens=1,thinking=False,request_sha256='0'*64)
        else:journal.state(complete=True,terminal='complete',qualified=False)
        return root

    def test_all_requests_match_despite_different_identity_metadata(self):
        a=self.fixture('a');b=self.fixture('b',identity={'client':'new','workload':{'nonce':'fixed'}})
        report=utility.compare(a/'requests.jsonl',b)
        self.assertTrue(report['inputs_match']);self.assertEqual(len(report['samples']),len(requests()))
        self.assertTrue(report['left']['verification']['complete'])
        self.assertEqual({x['left']['phase'] for x in report['samples']},{x['phase'] for x in requests()})
        for row in report['samples']:
            self.assertEqual(row['left']['request_sha256'],row['right']['request_sha256'])
            self.assertFalse(row['differences'])

    def test_tampered_chain(self):
        a=self.fixture('a');b=self.fixture('b')
        path=b/'requests.jsonl';rows=path.read_text().splitlines();row=json.loads(rows[1]);row['max_tokens']=25
        rows[1]=json.dumps(row);path.write_text('\n'.join(rows)+'\n')
        report=utility.compare(a,b)
        self.assertFalse(report['inputs_match']);self.assertFalse(report['right']['verification']['valid'])

    def test_incomplete_and_pending(self):
        a=self.fixture('a');b=self.fixture('b',pending=True)
        report=utility.compare(a,b)
        self.assertFalse(report['inputs_match']);self.assertTrue(report['right']['verification']['pending'])

    def test_missing_extra_reordered_and_duplicate_starts(self):
        original=requests();a=self.fixture('a')
        variants={'missing':original[:-1],'extra':original+[dict(original[-1],phase='extra')],
                  'reordered':[original[1],original[0],*original[2:]],'duplicate':[original[0],*original]}
        for name,rows in variants.items():
            with self.subTest(name=name):
                report=utility.compare(a,self.fixture(name,rows))
                self.assertFalse(report['inputs_match'])
                self.assertTrue(report['samples'])
                if name=='duplicate':self.assertTrue(report['right']['duplicate_request_keys'])

    def test_changed_request_parameters_and_body_hash(self):
        a=self.fixture('a')
        for key,value in [('max_tokens',25),('thinking',False),('request_sha256','f'*64)]:
            rows=copy.deepcopy(requests());rows[7][key]=value
            with self.subTest(key=key):
                report=utility.compare(a,self.fixture(key,rows))
                self.assertFalse(report['inputs_match']);self.assertIn(key,report['samples'][7]['differences'])
        rows=copy.deepcopy(requests());del rows[7]['max_tokens']
        self.assertFalse(utility.compare(a,self.fixture('missing-param',rows))['inputs_match'])

    def test_budgets_and_exclusive_report(self):
        a=self.fixture('a');b=self.fixture('b')
        output=self.root/'report.json'
        with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(utility.main([str(a),str(b),str(output)]),0)
        before=output.read_bytes()
        with self.assertRaises(FileExistsError):utility.main([str(a),str(b),str(output)])
        self.assertEqual(output.read_bytes(),before)
        huge=self.root/'huge';huge.mkdir();(huge/'requests.jsonl').write_bytes(b'x'*(utility.MAX_BYTES+1))
        self.assertFalse(utility.compare(a,huge)['inputs_match'])
        crowded=self.root/'crowded';crowded.mkdir()
        (crowded/'requests.jsonl').write_bytes(b'{}\n'*(utility.MAX_ROWS+1))
        (crowded/'journal-state.json').write_text('{}')
        self.assertFalse(utility.compare(a,crowded)['inputs_match'])


if __name__=='__main__':unittest.main()
