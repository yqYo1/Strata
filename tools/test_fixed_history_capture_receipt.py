# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Synthetic schema fixtures, never a runtime or production qualification."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import fixed_history_capture_receipt as adapter
from fixed_history_logits import ContractError, digest, ids_hash


class ReceiptTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.common = {'P': 2, 'N': 2, 'vocab': 3, 'max_context': 8, 'endian': 'little',
                       'stride': 1, 'position_convention': 'P-1+i predicts follow[i]',
                       'policy': {'bos': 'prompt-explicit', 'eos': 'score-all-stop-off', 'mask': 'all-follow'},
                       'prompt_ids': [0, 1], 'follow_ids': [2, 0],
                       'prompt_ids_sha256': ids_hash([0, 1]), 'follow_ids_sha256': ids_hash([2, 0])}
        self.start = '2026-10-13T01:00:01+00:00'
        self.end = '2026-10-13T01:00:02+00:00'
        self.env = {'STRATA_VERIFY_SEGMENTED': '1', 'STRATA_VERIFY_DEVICE_PLAN': '0'}
        model_files = [{'path': str(self.root / key), 'bytes': 5,
                        'pre_sha256': 'a' * 64, 'post_sha256': 'a' * 64}
                       for key in ('native', 'profile')]
        self.common['model'] = self.write('model.json', {'schema': 'strata.model-identity.v1', 'files': model_files})
        self.common['tokenizer'] = self.write('tokenizer', b'tokenizer')
        self.artifacts = {}
        contents = {'capture': struct.pack('<ii6f', 3, 2, 0, 1, 2, 3, 4, 5),
                    'binary': b'not a runnable binary', 'source': b'source identity', 'runtime': b'runtime identity',
                    'generator': b'generator fixture', 'verifier': adapter.SEGMENTED.encode(),
                    'prompt': b'0,1', 'follow': b'2,0', 'input_manifest': self.common,
                    'stdout': (b'prompt  : 0 1\noutput  : 2 0\ndecode                   2 tokens in 1.0 ms\n'
                               b'logits dumped            capture\n'),
                    'stderr': (adapter.SEGMENTED + '\n').encode()}
        for name, content in contents.items():
            self.artifacts[name] = self.write(name, content)
        provider = {'schema': 'strata.capture-provider.v1', 'provider': adapter.PROVIDER,
                    'native_pack': True, 'device_plan': 0, 'segmented': True,
                    'clip_scope': 'full-vocabulary-all-follow-no-position-clipping'}
        provider.update({key + '_sha256': self.artifacts[key]['sha256']
                         for key in ('generator', 'verifier', 'binary', 'runtime', 'source')})
        self.artifacts['provider_evidence'] = self.write('provider.json', provider)
        argv = [str(self.root / 'binary'), 'generate', '--native', str(self.root / 'native'),
                '--pack', str(self.root), '--expert-profile', str(self.root / 'profile'),
                '--expert-cache', '1', '--prefill', '2', '--tokens-file', 'prompt',
                '--spec-follow', 'follow', '--dump-logits', 'capture', '--max-new', '2',
                '--max-context', '8', '--logits-stride', '1', '--spec', '2',
                '--suffix-draft', '0', '--lookup-chain', '0', '--adapt-every', '0']
        self.command_hash = hashlib.sha256(json.dumps(argv, separators=(',', ':')).encode()).hexdigest()
        identity = {'pid': 100, 'start_ticks': 1000, 'sid': 100, 'pgid': 100}
        self.command = {'argv': argv, 'cwd': str(self.root), 'environment': self.env,
                        'started_utc': self.start, 'finished_utc': self.end,
                        'observation_complete': True, 'direct_child_reaped': True,
                        'session_empty': True, 'session_anchored': True, 'ownership_status': 'observed',
                        'errors': [], 'cleanup': [], 'survivors': [], 'exit_code': 0,
                        'pid': 100, 'direct_identity': identity, 'owners': [identity],
                        'logs': {key: {'bytes': (self.root / key).stat().st_size,
                                       'sha256': self.artifacts[key]['sha256']} for key in ('stdout', 'stderr')}}
        environment = self.write('environment.json', self.env)
        self.identities = {'schema': 'strata.capture-identities.v1', 'command_sha256': self.command_hash,
                           'environment_sha256': environment['sha256'], 'model_files': model_files,
                           'pack_path': str(self.root), 'native_path': str(self.root / 'native'),
                           'profile_path': str(self.root / 'profile'), 'boot_before': 'boot', 'boot_after': 'boot',
                           'started_utc': self.start, 'finished_utc': self.end,
                           'artifacts': {key: {
                               'pre_sha256': None if key in ('capture', 'stdout', 'stderr') else value['sha256'],
                               'post_sha256': value['sha256']} for key, value in self.artifacts.items()}}
        self.fault = {'schema': 'strata.capture-fault-health.v1', 'boot_before': 'boot', 'boot_after': 'boot',
                      'before_utc': '2026-10-13T01:00:00+00:00', 'after_utc': '2026-10-13T01:00:03+00:00',
                      'command_sha256': self.command_hash, 'kernel_complete': True, 'health_complete': True,
                      'records': [], 'health_records': [], 'kernel_evidence': self.write('kernel.json', []),
                      'health_evidence': self.write('health.json', [])}
        close = {'schema': 'strata.capture-close.v1', 'command_sha256': self.command_hash,
                 'capture_sha256': self.artifacts['capture']['sha256'], 'bytes': 32,
                 'closed_after_utc': '2026-10-13T01:00:03+00:00', 'close_error': None,
                 'forced': False, 'canceled': False, 'deadline_hit': False, 'budget_hit': False}
        self.sidecar = {'schema': 'strata.fixed-history-production-sidecar.v1', 'producer': 'schema-fixture',
                        'evidence_kind': 'synthetic-fixture', 'common': self.common,
                        'budget': {'max_vocab': 3, 'max_rows': 2, 'max_capture_bytes': 32},
                        'consumer': 'unit schema check', 'retire_after': 'test temporary-directory exit',
                        'owner_source': self.write('owner.py', b'fixture owner source'),
                        'owner_receipt': self.write('owner.json', {'commands': [self.command]}), 'command_index': 0,
                        'artifacts': self.artifacts, 'identities': self.write('identities.json', self.identities),
                        'environment': environment, 'fault': self.write('fault.json', self.fault),
                        'close': self.write('close.json', close)}
        self.owner_hash = self.sidecar['owner_source']['sha256']

    def write(self, name, content):
        path = self.root / name
        path.write_bytes(content if isinstance(content, bytes) else json.dumps(content).encode())
        return {'path': name, 'sha256': digest(path)}

    def run_fixture(self):
        sidecar_pin = self.write('sidecar.json', self.sidecar)
        with patch.object(adapter, 'OWNER_SHA256', self.owner_hash):
            return adapter.adapt(self.root / sidecar_pin['path'])

    def test_schema_fixture_never_qualifies(self):
        result = self.run_fixture()
        self.assertEqual(result['positions'], [1, 2])
        self.assertEqual(result['targets'], [2, 0])
        self.assertFalse(result['production_capture_qualified'])
        self.assertFalse(result['quality_qualified'])
        self.sidecar['evidence_kind'] = 'production'
        self.assertFalse(self.run_fixture()['production_capture_qualified'])

    def test_missing_real_owner_rejected(self):
        self.write('sidecar.json', self.sidecar)
        with self.assertRaises(ContractError):
            adapter.adapt(self.root / 'sidecar.json')

    def test_owner_negative_closure(self):
        original = copy.deepcopy(self.command)
        for key, value in (('exit_code', True), ('exit_code', 1), ('direct_child_reaped', False),
                           ('session_empty', False), ('errors', ['fault']), ('cleanup', ['SIGTERM'])):
            with self.subTest(key=key, value=value):
                self.command = copy.deepcopy(original)
                self.command[key] = value
                self.sidecar['owner_receipt'] = self.write('owner.json', {'commands': [self.command]})
                with self.assertRaises(ContractError):
                    self.run_fixture()

    def test_route_spoof_rejected(self):
        for extra in (['--serve'], ['--stop-eos'], ['--mtp', 'model'], ['--max-new', '1']):
            with self.subTest(extra=extra):
                command = copy.deepcopy(self.command)
                command['argv'] += extra
                self.sidecar['owner_receipt'] = self.write('owner.json', {'commands': [command]})
                with self.assertRaises(ContractError):
                    self.run_fixture()

    def test_changed_source_and_environment(self):
        self.identities['artifacts']['generator']['pre_sha256'] = '0' * 64
        self.sidecar['identities'] = self.write('identities.json', self.identities)
        with self.assertRaises(ContractError):
            self.run_fixture()
        self.identities['artifacts']['generator']['pre_sha256'] = self.artifacts['generator']['sha256']
        self.sidecar['identities'] = self.write('identities.json', self.identities)
        self.command['environment']['STRATA_VERIFY_SEGMENTED'] = '0'
        self.sidecar['owner_receipt'] = self.write('owner.json', {'commands': [self.command]})
        with self.assertRaises(ContractError):
            self.run_fixture()

    def test_stale_fault_and_reported_fault(self):
        for key, value in (('before_utc', self.end), ('records', ['GPU fault'])):
            original = copy.deepcopy(self.fault)
            self.fault[key] = value
            self.sidecar['fault'] = self.write('fault.json', self.fault)
            with self.assertRaises(ContractError):
                self.run_fixture()
            self.fault = original

    def test_pin_header_nonfinite_and_trailing_bytes(self):
        # Repin complete output identity so rejection must inspect its contents.
        for raw in (struct.pack('<ii6f', 3, 1, 0, 1, 2, 3, 4, 5),
                    struct.pack('<ii6f', 3, 2, 0, 1, float('nan'), 3, 4, 5),
                    struct.pack('<ii6f', 3, 2, 0, 1, 2, 3, 4, 5) + b'x'):
            with self.subTest(raw=raw):
                self.artifacts['capture'] = self.write('capture', raw)
                self.identities['artifacts']['capture']['post_sha256'] = self.artifacts['capture']['sha256']
                self.sidecar['identities'] = self.write('identities.json', self.identities)
                close = json.loads((self.root / 'close.json').read_text())
                close['capture_sha256'] = self.artifacts['capture']['sha256']
                close['bytes'] = len(raw)
                self.sidecar['close'] = self.write('close.json', close)
                with self.assertRaises(ContractError):
                    self.run_fixture()

    def test_duplicate_json_key(self):
        path = self.root / 'duplicate.json'
        path.write_text('{"passed":true,"passed":false}')
        with self.assertRaises(ContractError):
            adapter.bounded_json(path)

    def test_budget_and_history_mismatch(self):
        self.sidecar['budget']['max_capture_bytes'] = 31
        with self.assertRaises(ContractError):
            self.run_fixture()
        self.sidecar['budget']['max_capture_bytes'] = 32
        self.common['follow_ids_sha256'] = '0' * 64
        with self.assertRaises(ContractError):
            self.run_fixture()


if __name__ == '__main__':
    unittest.main()
