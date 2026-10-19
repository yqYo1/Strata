#!/usr/bin/env python3
"""Original-source and relocated-binary admission; no real Git/GPU execution."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import resource_engine as r


class SourceProofTests(unittest.TestCase):
    def test_closed_diagnostic_cannot_restart_after_raw_root_retirement(self):
        import copy
        previous=dict(mode='client-resources',diagnostic_version=1,complete=True,passed=True,
            ownership_closed=True,active=False,fresh_faults=[],resource_admission_qualified=True,
            artifact_retirement=dict(complete=True))
        for variant in ('new_capture_exists','incomplete','failed','active','open_owner','fault',
                        'unadmitted','raw_still_needed'):
            with self.subTest(variant=variant):
                changed=copy.deepcopy(previous)
                if variant=='new_capture_exists':changed['diagnostic_version']=2
                if variant=='incomplete':changed['complete']=False
                if variant=='failed':changed['passed']=False
                if variant=='active':changed['active']=True
                if variant=='open_owner':changed['ownership_closed']=False
                if variant=='fault':changed['fresh_faults']=['GPU fault']
                if variant=='unadmitted':changed['resource_admission_qualified']=False
                if variant=='raw_still_needed':changed['artifact_retirement']['complete']=False
                with patch.object(r.generation,'committed',return_value=changed):
                    with self.assertRaisesRegex(RuntimeError,'no automatic retry'):r.prepare()

    def test_result_clocks_and_work_must_equal_the_verified_journal(self):
        journal=[dict(workload='regular-code',request_sha256='a'*64,read_tokens=33000,
            generated_tokens=1500,ok=True,client_interval=dict(request_begin_monotonic_ns=10,
                first_output_monotonic_ns=20,request_end_monotonic_ns=30))]
        import copy
        official=[dict(copy.deepcopy(journal[0]),controller_eligible=True,reasoning_tokens=None)]
        r.require_journal_binding(official,journal)
        for variant in ('clock','input','output','identity','success','missing'):
            with self.subTest(variant=variant):
                changed=copy.deepcopy(official)
                if variant=='clock':changed[0]['client_interval']['first_output_monotonic_ns']+=1
                if variant=='input':changed[0]['read_tokens']+=1
                if variant=='output':changed[0]['generated_tokens']+=1
                if variant=='identity':changed[0]['request_sha256']='b'*64
                if variant=='success':changed[0]['ok']=False
                if variant=='missing':changed=[]
                with self.assertRaises(RuntimeError):r.require_journal_binding(changed,journal)

    def test_relocation_requires_original_committed_reference_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);native=root/'retained-strata';native.write_bytes(b'engine')
            binary=dict(path=str(native),bytes=6,sha256=hashlib.sha256(b'engine').hexdigest())
            source=b'original source';recipe=b'original recipe'
            full=dict(source_commit='native-revision',binary=dict(binary,path='/original/build/strata'),
                project=dict(root='/project',revision='native-revision',source_sha256={'src/test.cpp':hashlib.sha256(source).hexdigest()}),
                dependencies=[],recipe_reference=dict(path='recipe.json',sha256=hashlib.sha256(recipe).hexdigest()))
            receipt=root/'source-build.json';receipt.write_text(json.dumps(full))
            receipt_pin=dict(path=str(receipt),sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
            reference=json.dumps(dict(retained_validation_reference=dict(source_commit=full['source_commit'],
                binary=binary,source_build_record=receipt_pin))).encode()
            record=dict(source_build_record=receipt_pin,build_source_commit=full['source_commit'],binary=binary,
                fallback_reference=dict(commit='relocation-revision',path='build.json',sha256=hashlib.sha256(reference).hexdigest()))
            def git(*args):
                return reference if args[-1]=='relocation-revision:build.json' else recipe
            with patch.object(r.generation,'git',side_effect=git),patch.object(r,'blobs',return_value=[('src/test.cpp',source)]):
                self.assertEqual(r.source_proof(record),1)
                bad=dict(record,fallback_reference=dict(record['fallback_reference'],sha256='0'*64))
                with self.assertRaises(RuntimeError):r.source_proof(bad)
                bad=dict(record,build_source_commit='other-source')
                with self.assertRaises(RuntimeError):r.source_proof(bad)
                native.write_bytes(b'changed engine')
                with self.assertRaises(RuntimeError):r.source_proof(record)
            with patch.object(r.generation,'git',side_effect=git),patch.object(r,'blobs',return_value=[('src/test.cpp',b'wrong source')]):
                with self.assertRaises(RuntimeError):r.source_proof(record)


if __name__=='__main__':unittest.main()
