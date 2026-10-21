#!/usr/bin/env python3
"""Offline refusal tests for private-layout and same-native model admission."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import packed_bank as p


def private_log():
    strides=(2176000,1510400,1971200,1715200,2329600,1868800,2662400)
    rows=[]
    for layer in range(48):
        stride=strides[layer%7];slots=min(512,1022361600//stride)
        rows.append(f'strata packed expert bank: layer={layer} slots={slots} stride={stride} used_bytes={slots*stride} backing_bytes=1022361600 full_fit={int(slots==512)}')
    last=min(512,1022361600//strides[47%7])
    rows.append(f'strata MoE expert bank complete: uploads=100 requested_H2D_bytes=200 permanent_hits=300 permanent_slots=100 total_slots={last} final_drain=checked')
    return '\n'.join(rows)+'\n'


class LayoutEvidence(unittest.TestCase):
    def test_complete_ordered_private_layout(self):
        self.assertEqual(p.geometry(private_log(),1)['actual_private_layers'],48)

    def test_geometry_and_request_boundary_refusals(self):
        valid=private_log();lines=valid.splitlines()
        cases=[valid.replace('layer=0 ','layer=1 ',1), '\n'.join(lines[1:])+'\n',
            valid.replace('full_fit=1','full_fit=0',1),valid.replace('backing_bytes=1022361600','backing_bytes=1022361856',1),
            valid.replace('stride=2176000','stride=2176001',1),valid.replace('final_drain=checked','final_drain=pending'),
            valid.replace('uploads=100','uploads=0'), '\n'.join(lines[-1:]+lines[:-1])+'\n']
        for text in cases:
            with self.subTest(text=text[:90]),self.assertRaises(RuntimeError):p.geometry(text,1)
        with self.assertRaises(RuntimeError):p.geometry(valid,0)


class NativeReuseEvidence(unittest.TestCase):
    def test_component_success_does_not_admit_different_native_code_or_artifacts(self):
        artifact=dict(path='/native',sha256='a'*64)
        old=dict(source_commit='packed-native',binary=artifact,fixtures=dict(sycl_expert_bank_test=artifact,file_expert_source_test=artifact),
            compiler='compiler',configure_argv=['recipe'],recipe_reference='original',dependencies=[],inputs=[{}])
        bank=dict(passed=True,complete=True,ownership_closed=True,fresh_faults=[],boot_unchanged=True,mode='packed-validate',
            checks=[{},{}],build_source_commit=old['source_commit'],fixture=artifact,provider_fixture=artifact)
        for variant in ('valid','failed','owner','fault','fixture','binary','recipe','native_diff'):
            with self.subTest(variant=variant):
                full=json.loads(json.dumps(old));full['source_commit']='new-controller-source'
                value=copy.deepcopy(bank);diff=b''
                if variant=='failed':value['passed']=False
                if variant=='owner':value['ownership_closed']=False
                if variant=='fault':value['fresh_faults']=['fault']
                if variant=='fixture':full['fixtures']['sycl_expert_bank_test']['sha256']='b'*64
                if variant=='binary':full['binary']['sha256']='b'*64
                if variant=='recipe':full['recipe_reference']='other'
                if variant=='native_diff':diff=b'src/prefill/prefill.cpp\n'
                def git(*args):return json.dumps(old).encode() if args[0]=='show' else diff
                with patch.object(p.q,'current_build',return_value=(full,9836)),patch.object(p.gen,'committed',return_value={'packed_gpu':value}), \
                     patch.object(p.gen,'git',side_effect=git),patch.object(p.gen.common,'digest',return_value='digest'):
                    if variant=='valid':self.assertEqual(p.qualification()[0],full)
                    else:
                        with self.assertRaises(RuntimeError):p.qualification()


class NativeClosureEvidence(unittest.TestCase):
    def claim(self):
        return dict(binary={'sha256':'a'*64},args=['--serve','--spec','4'],boot_id='boot',
            native_validation_probe={'sha256':'b'*64},
            wrapper=dict(pid=43,start_ticks=100,sid=42),native=dict(pid=44,start_ticks=101,sid=42))

    def test_native_exit_identity_and_observer_refusals(self):
        for variant in ('valid','failed','binary','boot','args','probe','collector','profile','leader','ticks','alias','end_identity'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);claim=self.claim()
                record=dict(binary=copy.deepcopy(claim['binary']),native_validation_probe=copy.deepcopy(claim['native_validation_probe']))
                plan={'config':{'args':['--spec','4']}}
                if variant=='binary':claim['binary']['sha256']='c'*64
                if variant=='boot':claim['boot_id']='other'
                if variant=='args':claim['args'].append('--other')
                if variant=='probe':claim['native_validation_probe']={}
                if variant=='collector':claim['resource_collector']={}
                if variant=='profile':claim['native_profile']={}
                if variant=='leader':claim['native']['sid']=99
                if variant=='ticks':claim['native']['start_ticks']=True
                if variant=='alias':claim['native']['pid']=claim['wrapper']['pid']
                end=dict(copy.deepcopy(claim),exit_code=1 if variant=='failed' else 0)
                if variant=='end_identity':end['native']['pid']=55
                (root/'engine-start.json').write_text(json.dumps(claim))
                (root/'engine-exit.json').write_text(json.dumps(end))
                if variant=='valid':self.assertEqual(p.native_end(root,record,plan,'boot',42),end)
                else:
                    with self.assertRaises(RuntimeError):p.native_end(root,record,plan,'boot',42)

    def test_closed_benchmark_collection_calls_actual_native_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);arm=root/'0';(arm/'bench'/'run').mkdir(parents=True)
            result=root/'result.json';claim=self.claim()
            claim['boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            (arm/'engine-start.json').write_text(json.dumps(claim))
            (arm/'engine-exit.json').write_text(json.dumps(dict(claim,exit_code=0)))
            (arm/'engine.stderr').write_text('strata serve: prompt 100 tokens = warmup\n'
                'strata MoE expert bank complete: uploads=100 requested_H2D_bytes=200 permanent_hits=300 permanent_slots=383 total_slots=384 final_drain=checked\n'
                'strata serve: prompt 117489 tokens = long\n')
            (arm/'bench'/'run'/'results.json').write_text('{}')
            record=dict(binary=claim['binary'],native_validation_probe=claim['native_validation_probe'],
                plans=[dict(root=str(arm),config={'args':['--spec','4']},command=[],environment_overrides={})],
                processes=[],complete=False,ownership_closed=True,active=False,budgets={})
            (root/'plan.json').write_text(json.dumps(record))
            class Owner:
                active=None
                commands=[]
                def __init__(self,*args):pass
                def run(self,*args,**kwargs):return dict(exit_code=0,pid=42),None,None
            with patch.object(p,'ROOT',root),patch.object(p,'RESULT',result), \
                 patch.object(p.gen,'committed',return_value=record),patch.object(p,'current',return_value={'root':str(root),'environment':{}}), \
                 patch.object(p.subprocess,'check_output',side_effect=['-- cursor: test\n','']), \
                 patch.object(p.supervision,'Owner',Owner),patch.object(p.supervision,'closed',return_value=True), \
                 patch.object(p.admission,'official_evidence',return_value=({'generated_tokens':24},{'valid':True})), \
                 patch.object(p.admission,'mixed_evidence',return_value={'checked_retirements':96}), \
                 patch.object(p.evidence,'activation',return_value={'active':True}), \
                 patch.object(p.assets,'verify_pool'),patch.object(p.guards,'driver_guard',return_value={'driver':'xe'}):
                self.assertEqual(p.run(),0)
            closed=json.loads(result.read_text())
            self.assertEqual(closed['stage'],'partial')
            self.assertTrue(closed['processes'][0]['passed'])
            self.assertEqual(closed['processes'][0]['native_exit_code'],0)
            self.assertEqual(closed['processes'][0]['geometry']['requested_work']['uploads'],100)


if __name__=='__main__':unittest.main()
