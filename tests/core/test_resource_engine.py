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
    def test_new_diagnostic_requires_fresh_root_and_retired_previous_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(r,'ROOT',root),patch.object(r,'modules',return_value=(None,None,None)):
                with self.assertRaisesRegex(RuntimeError,'single-use'):r.prepare()
            for changed in (dict(complete=False,passed=True,ownership_closed=True,active=False),
                            dict(complete=True,passed=True,ownership_closed=True,active=True),
                            dict(complete=True,passed=True,ownership_closed=True,active=False)):
                with self.subTest(changed=changed),patch.object(r,'ROOT',root/'fresh'), \
                     patch.object(r,'modules',return_value=(None,None,None)), \
                     patch.object(r.generation,'committed',return_value=changed),patch.object(r,'qualification') as q:
                    with self.assertRaisesRegex(RuntimeError,'preserve/close'):r.prepare()
                    q.assert_not_called()

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


class CurrentQualificationTests(unittest.TestCase):
    def fixture(self):
        from types import SimpleNamespace
        from test_generation_engine import sample
        binary=dict(path='/current',sha256='a'*64)
        full=dict(source_commit='native',binary=binary)
        requests=[dict(sample(),workload='regular-code'),dict(sample(),workload='varied-prose',order=1,acceptance_rate=.39,drafts_accepted=390)]
        route=dict(actual_MoE_mode=10,restored_dense_layers=48,cache_payload_and_map_match=True,private_bank=dict(actual_private_layers=48))
        row=dict(arm=1,passed=True,complete=True,ownership_closed=True,active=False,native_exit_code=0,
            fault_probe_successful=True,fresh_faults=[],boot_unchanged=True,build_source_commit='native',binary_sha256=binary['sha256'],
            requests=requests,routes=[route.copy(),route.copy()])
        closed=dict(complete=True,passed=True,ownership_closed=True,active=False,stage='closed',processes=[row.copy(),row],
            plans=[{},dict(binary=binary)])
        return full,closed,SimpleNamespace(RESULT=Path('/reference'))

    def test_only_closed_same_build_current_generation_can_be_profiled(self):
        import copy
        from types import SimpleNamespace
        full,closed,comparison=self.fixture()
        bank=SimpleNamespace(qualification=lambda:(full,{'source':'proof'}))
        for variant in ('valid','not_closed','old_arm_fault','source','binary','owner','short','CPU_route','weak_code'):
            value=copy.deepcopy(closed);current=value['processes'][1]
            if variant=='not_closed':value['complete']=False
            if variant=='old_arm_fault':value['processes'][0]['fresh_faults']=['fault']
            if variant=='source':current['build_source_commit']='other'
            if variant=='binary':current['binary_sha256']='b'*64
            if variant=='owner':current['ownership_closed']=False
            if variant=='short':current['requests'][0]['read_tokens']=10
            if variant=='CPU_route':current['routes'][0]['actual_MoE_mode']=2
            if variant=='weak_code':current['requests'][0].update(acceptance_rate=.4,drafts_accepted=400)
            with self.subTest(variant=variant),patch.object(r,'modules',return_value=(None,bank,comparison)), \
                 patch.object(r.generation,'committed',return_value=value),patch.object(r.generation.common,'digest',return_value='digest'):
                if variant=='valid':self.assertEqual(r.qualification()[0],full)
                else:
                    with self.assertRaises(RuntimeError):r.qualification()


class CurrentCollectionPathTests(unittest.TestCase):
    def test_resource_collection_calls_the_real_declared_collector_closure_reader(self):
        import copy
        from types import SimpleNamespace
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'bench'/'run').mkdir(parents=True);result_path=root/'result.json'
                reader=root/'reader.py';reader.write_text('# passive test reader\n')
                collection=dict(source=dict(path=str(reader),sha256=r.generation.common.digest(reader)),seconds=900,interval=1)
                binary=dict(path='/not-executed',sha256='a'*64);config={'args':['--spec','4']}
                record=dict(stage='prepared',complete=False,passed=False,ownership_closed=True,active=False,
                    binary=binary,config=config,environment={},resource_collection=collection,native_validation_probe=None,budgets={},command=[])
                (root/'diagnostic-plan.json').write_text(json.dumps(record));(root/'plan.json').write_text(json.dumps({'resource_collection':collection}))
                boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                start=dict(binary=binary,args=['--serve','--spec','4'],boot_id=boot,native_validation_probe=None,
                    native=dict(pid=44,start_ticks=101,sid=42),wrapper=dict(pid=43,start_ticks=100,sid=42))
                end=dict(copy.deepcopy(start),exit_code=0,resource_collector=dict(pid=45,start_ticks=102,sid=42),resource_collector_exit_code=1 if fail else 0)
                for name,data in [('engine-start.json',start),('engine-exit.json',end)]: (root/name).write_text(json.dumps(data))
                (root/'engine.stderr').write_text('')
                (root/'client-resources.jsonl').write_text('{}\n{}\n')
                (root/'bench'/'run'/'results.json').write_text(json.dumps({'complete':True,'results':{'failures':[]}}))
                (root/'bench'/'run'/'requests.jsonl').write_text('{}\n')
                row=dict(request_sha256='b'*64,client_interval={})
                comparison=SimpleNamespace(official=lambda *a,**k:([row],{'valid':True},[]),native_routes=lambda *a,**k:[{'mode':10}])
                class Owner:
                    active=None
                    commands=[]
                    def __init__(self,*a):pass
                    def run(self,*a,**k):return {'exit_code':0,'pid':42},None,None
                with patch.object(r,'ROOT',root),patch.object(r,'RESULT',result_path), \
                     patch.object(r,'modules',return_value=(None,None,comparison)),patch.object(r,'current',return_value={'processes':[{}, {'requests':[row]}]}), \
                     patch.object(r.generation,'committed',return_value=record),patch.object(r.supervision,'Owner',Owner), \
                     patch.object(r.supervision,'closed',return_value=True),patch.object(r.subprocess,'check_output',side_effect=['-- cursor: x\n','']), \
                     patch.object(r.generation,'decode_timing_evidence',return_value=[{'matched':True}]), \
                     patch.object(r.client_resources,'summarize_capture',return_value={'coverage':True}), \
                     patch.object(r,'collection_read_spans',return_value={}),patch.object(r.guards,'driver_guard'):
                    self.assertEqual(r.run(),1 if fail else 0)
                saved=json.loads(result_path.read_text())
                self.assertTrue(saved['ownership_closed']);self.assertEqual(saved['stage'],'closed')
                self.assertEqual(saved['passed'],not fail)
                if fail:self.assertIn('collector',saved['error'])
                else:self.assertTrue(saved['resources']['coverage'])


if __name__=='__main__':unittest.main()
