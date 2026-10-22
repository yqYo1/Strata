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
            for changed in (dict(stage='closed',complete=False,passed=False,ownership_closed=False,active=False),
                            dict(stage='closed',complete=True,passed=True,ownership_closed=True,active=True),
                            dict(stage='closed',complete=True,passed=True,ownership_closed=True,active=False)):
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
        request=dict(sample(),workload='varied-prose',acceptance_rate=.39,drafts_accepted=390)
        route=dict(actual_MoE_mode=10,restored_dense_layers=48,cache_payload_and_map_match=True,
            private_bank=dict(actual_private_layers=48))
        row=dict(passed=True,complete=True,ownership_closed=True,active=False,native_exit_code=0,
            fault_probe_successful=True,fresh_faults=[],boot_unchanged=True,requests=[request],routes=[route],
            observation=dict(summary=dict(accounting_valid=1)))
        closed=dict(stage='closed',binary=binary,processes=[row],
            process_plans=[dict(root='/retired',inputs={'/retired/expert-profile.bin':'profile'})])
        return full,closed,SimpleNamespace(RESULT=Path('/reference'),
            original_setup=lambda:dict(reference={'commit':'original','path':'canonical.json'}))

    def test_only_closed_same_build_current_route_can_be_profiled(self):
        import copy
        import query_profile as q
        full,closed,restored=self.fixture()
        for variant in ('valid','not_closed','fault','binary','owner','short','CPU_route','accounting'):
            value=copy.deepcopy(closed);row=value['processes'][0]
            if variant=='not_closed':value['stage']='running'
            if variant=='fault':row['fresh_faults']=['fault']
            if variant=='binary':value['binary']['sha256']='b'*64
            if variant=='owner':row['ownership_closed']=False
            if variant=='short':row['requests'][0]['read_tokens']=10
            if variant=='CPU_route':row['routes'][0]['actual_MoE_mode']=2
            if variant=='accounting':row['observation']['summary']['accounting_valid']=0
            with self.subTest(variant=variant),patch.object(r,'modules',return_value=(None,restored,None)), \
                 patch.object(q,'current_build',return_value=(full,9853)), \
                 patch.object(r.generation,'committed',return_value=value), \
                 patch.object(r.generation,'git',return_value=json.dumps({'processes':[{'requests':[{'request_sha256':'a'*64}]}]}).encode()), \
                 patch.object(r.generation.common,'digest',return_value='digest'):
                if variant=='valid':self.assertEqual(r.qualification()[0],full)
                else:
                    with self.assertRaises(RuntimeError):r.qualification()


class RegisteredThreadTests(unittest.TestCase):
    def fixture(self):
        text='\n'.join('strata IQ2_GU_THREAD stage='+stage+' request=5 slot=0 tid=44 calls='+str(count)+
            ' row_ops='+str(count)+' nt1_rows='+str(count)+' pair_rows='+str(count)+' cpu=2 pinned=1'
            for stage,count in [('begin',10),('end',30)])
        samples=[dict(begin_monotonic_ns=i*100,end_monotonic_ns=i*100+10,ticks_per_second=100,
            threads=[dict(tid=44,start_ticks=5,cpu_ticks=i*10)]) for i in range(8)]
        interval=dict(request_begin_monotonic_ns=0,first_output_monotonic_ns=350,request_end_monotonic_ns=800)
        audit=dict(active_threads=1,pinned_threads=1,cpus=[2],calls=20,row_ops=20,nt1_rows=20,pair_rows=20)
        return text,samples,interval,audit

    def test_registered_work_and_scheduled_ticks_are_separate_bound_evidence(self):
        text,samples,interval,audit=self.fixture()
        result=r.observed_iq2_threads(text,samples,interval,audit)['threads'][0]
        self.assertEqual(result['work']['row_ops'],20)
        for phase in ('before_first_output','after_first_output'):
            self.assertEqual(result[phase],dict(contained_intervals=3,cpu_seconds=.3))

    def test_reject_missing_reused_regressed_and_unbound_threads(self):
        for variant in ('missing','reuse','boundary_reuse','ticks','work','cpu','pin','coverage'):
            text,samples,interval,audit=self.fixture()
            if variant=='missing':samples[2]['threads']=[]
            if variant=='reuse':samples[2]['threads'][0]['start_ticks']=6
            if variant=='boundary_reuse':
                for row in samples[4:]:row['threads'][0]['start_ticks']=6
            if variant=='ticks':samples[2]['threads'][0]['cpu_ticks']=0
            if variant=='work':audit['row_ops']+=1
            if variant=='cpu':audit['cpus']=[3]
            if variant=='pin':audit['pinned_threads']=0
            if variant=='coverage':interval['first_output_monotonic_ns']=110
            with self.subTest(variant=variant),self.assertRaises(RuntimeError):
                r.observed_iq2_threads(text,samples,interval,audit)


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
                row=dict(request_sha256='b'*64,client_interval={},acceptance_rate=.8)
                (root/'bench'/'run'/'requests.jsonl').write_text(json.dumps({'kind':'result','sample':row})+'\n')
                restored=SimpleNamespace(recipe=SimpleNamespace(work_evidence=lambda *a:([{}],[{}],[{'matched':True}]),compact_work=lambda w:w))
                comparison=SimpleNamespace(official=lambda *a,**k:([row],{'valid':True},[]),native_routes=lambda *a,**k:[{'mode':10}])
                class Owner:
                    active=None
                    commands=[]
                    def __init__(self,*a):pass
                    def run(self,*a,**k):
                        record.update(native_identity=start['native'],effective_environment_verified=True)
                        return {'exit_code':0,'pid':42},None,None
                with patch.object(r,'ROOT',root),patch.object(r,'RESULT',result_path), \
                     patch.object(r,'modules',return_value=(None,restored,comparison)),patch.object(r,'current',return_value={'request_sha256':'b'*64}), \
                     patch.object(r.generation,'committed',return_value=record),patch.object(r.supervision,'Owner',Owner), \
                     patch.object(r.supervision,'closed',return_value=True),patch.object(r.subprocess,'check_output',side_effect=['-- cursor: x\n','']), \
                     patch.object(r.generation,'journal_evidence',return_value=({},[row])), \
                     patch.object(r,'observed_iq2_threads',return_value={'threads':[]}), \
                     patch.object(r.client_resources,'summarize_capture',return_value={'coverage':True}), \
                     patch.object(r,'collection_read_spans',return_value={}),patch.object(r.guards,'driver_guard'):
                    self.assertEqual(r.run(),1 if fail else 0)
                saved=json.loads(result_path.read_text())
                self.assertTrue(saved['ownership_closed']);self.assertEqual(saved['stage'],'closed')
                self.assertEqual(saved['passed'],not fail)
                if fail:self.assertIn('collector',saved['error'])
                else:self.assertTrue(saved['resources']['coverage'])


class HistoricalNativeInputTests(unittest.TestCase):
    def test_retention_controller_requires_the_reviewed_original_and_current_pair(self):
        import query_profile as q
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);name='tools/sycl/build_engine.py'
            path=root/name;path.parent.mkdir(parents=True);path.write_bytes(b'reviewed retention controller')
            original=hashlib.sha256(b'original controller').hexdigest()
            actual=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(q,'P',root),patch.object(q,'RETENTION_CONTROLLER_UPDATES',{name:(original,actual)}):
                for variant in ('valid','wrong_original','uncommitted','unreviewed'):
                    path.write_bytes(b'changed recipe' if variant=='unreviewed' else b'reviewed retention controller')
                    committed=b'previous' if variant=='uncommitted' else path.read_bytes()
                    expected='0'*64 if variant=='wrong_original' else original
                    with self.subTest(variant=variant),patch.object(q.gen,'git',return_value=committed):
                        self.assertEqual(q.worktree_input_matches(dict(root=str(root)),name,expected),variant=='valid')

    def test_only_named_committed_host_gate_updates_can_reuse_native_inputs(self):
        import query_profile as q
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=dict(root=str(root))
            original=hashlib.sha256(b'original').hexdigest()
            for name in sorted(q.HOST_GATE_TESTS):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b'new committed Python gate')
                with self.subTest(name=name),patch.object(q,'P',root):
                    with patch.object(q.gen,'git',return_value=path.read_bytes()):
                        self.assertTrue(q.worktree_input_matches(source,name,original))
                        full=dict(project=dict(source,source_sha256={name:original}))
                        update=q.host_gate_test_updates(full)[0]
                        self.assertEqual(update['build_input_sha256'],original)
                        self.assertEqual(update['committed_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
                    with patch.object(q.gen,'git',return_value=b'previous committed gate'):
                        self.assertFalse(q.worktree_input_matches(source,name,original))
                        with self.assertRaises(RuntimeError):q.host_gate_test_updates(full)
                    other=root/'dependency';(other/name).parent.mkdir(parents=True,exist_ok=True)
                    (other/name).write_bytes(path.read_bytes())
                    self.assertFalse(q.worktree_input_matches(dict(root=str(other)),name,original))
            for name in ('src/program/generate.cpp','tests/sycl/native_dense_bank.cpp',
                         'tests/core/test_unlisted.py'):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b'changed native or unlisted source')
                with self.subTest(name=name),patch.object(q,'P',root),patch.object(q.gen,'git',return_value=path.read_bytes()):
                    self.assertFalse(q.worktree_input_matches(source,name,original))
                    self.assertTrue(q.worktree_input_matches(source,name,hashlib.sha256(path.read_bytes()).hexdigest()))


if __name__=='__main__':unittest.main()
