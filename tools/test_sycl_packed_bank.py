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
                    if variant=='valid':
                        with patch.object(p.q,'host_gate_test_updates',return_value=[]):
                            self.assertEqual(p.qualification()[0],full)
                    else:
                        with self.assertRaises(RuntimeError):p.qualification()


class BuildInputIdentity(unittest.TestCase):
    def test_only_the_named_committed_host_test_can_differ(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);name='tests/core/test_resource_engine.py'
            path=root/name;path.parent.mkdir(parents=True);path.write_bytes(b'new committed test')
            source=dict(root=str(root));old='0'*64
            with patch.object(p.q,'P',root),patch.object(p.q.gen,'git',return_value=path.read_bytes()):
                self.assertTrue(p.q.worktree_input_matches(source,name,old))
                with patch.object(p.q.gen,'git',return_value=b'different committed test'):
                    self.assertFalse(p.q.worktree_input_matches(source,name,old))
                for other in ('tests/core/other_test.py','src/native.cpp','CMakeLists.txt'):
                    target=root/other;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
                    self.assertFalse(p.q.worktree_input_matches(source,other,old))
            with patch.object(p.q,'P',root/'other-project'):
                self.assertFalse(p.q.worktree_input_matches(source,name,old))


if __name__=='__main__':unittest.main()
