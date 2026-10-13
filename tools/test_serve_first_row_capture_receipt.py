# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""CPU source fixtures. No GPU/launch; production registry is never overridden.

Implementation handoff: UNEXECUTED. Main owns serialized CPU execution.
"""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from fixed_history_logits import ContractError
from fixed_history_capture_receipt import ids
from serve_first_row_capture_receipt import (adapt_serve, read_row, validate_record,
                                           reviewed_producers, validate_faults, closed_command, validate_retained_consumer)


class RowFormatTests(unittest.TestCase):
    def test_exact_header_and_finite_full_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'row.bin'
            path.write_bytes(struct.pack('<ii3f', 3, 1, 0, 1, -2))
            self.assertEqual(read_row(path, 3), (0, 1, -2))
            for content in (b'MAGIC!!!' + struct.pack('<3f', 0, 1, -2),
                            struct.pack('>ii3f', 3, 1, 0, 1, -2),
                            struct.pack('<ii3f', 3, 2, 0, 1, -2),
                            struct.pack('<ii2f', 3, 1, 0, 1),
                            struct.pack('<ii3f', 3, 1, 0, float('nan'), -2),
                            struct.pack('<ii3f', 3, 1, 0, float('inf'), -2)):
                with self.subTest(content=content):
                    path.write_bytes(content)
                    with self.assertRaises(ContractError):
                        read_row(path, 3)


class SemanticNegativeTests(unittest.TestCase):
    """Synthetic receipt exercises actual semantic checks, cannot register rows."""
    def setUp(self):
        self.registration = {'source_revision': 'base', 'source_pins': {'generate': 'p'},
                             'binary_sha256': 'b', 'provider_hashes': {'runtime': 'r'},
                             'model_files': {'model': 'm'}, 'fixture': {'total_tokens': 4},
                             'request_index': 0,
                             'retained_capture': {'path': 'kept.bin', 'sha256': 'h'}}
        self.fixture = [0, 1, 2, 3]
        capture = {'path': 'kept.bin', 'sha256': 'h', 'prompt_tokens': 2,
                   'predicted_position': 1, 'shape': [1, 248320], 'dtype': 'IEEE754FP32LE',
                   'bytes': 993288, 'finite': True, 'consumer': 'paired rows',
                   'next_review': 'review', 'byte_budget': 1048576, 'argmax': 1}
        request = {'prompt_tokens': 2, 'max_new': 1, 'complete': True,
                   'checkpoint_key_omitted': True,
                   'packet_sha256': hashlib.sha256(b'GEN 1 temperature=0 0 1\n').hexdigest()}
        self.record = {'active': False, **{k: True for k in ('passed', 'complete', 'main_reviewed',
                       'actual_first_row_capture_qualified', 'full_vocabulary_capture',
                       'phase_engine_lifetime_qualified')},
                       'source_revision': 'base', 'source_pins': {'generate': 'p'},
                       'binary_sha256': 'b', 'provider_hashes': {'runtime': 'r'},
                       'model-hashes-before': {'complete': True, 'files': {'model': 'm'}},
                       'model-hashes-after': {'complete': True, 'files': {'model': 'm'}},
                       'fixture': {'total_tokens': 4}, 'requests': [request], 'captures': [capture]}

    def test_bad_request_position_model_provenance_active_consumer(self):
        mutations = [lambda d: d.update(active=True),
                     lambda d: d.update(source_revision='caller-claim'),
                     lambda d: d['model-hashes-after']['files'].update(model='changed'),
                     lambda d: d['requests'][0].update(packet_sha256='bad'),
                     lambda d: d['captures'][0].update(predicted_position=2),
                     lambda d: d['captures'][0].update(consumer=''),
                     lambda d: d['captures'][0].update(path='retired.bin')]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                record = copy.deepcopy(self.record)
                mutate(record)
                with self.assertRaises(ContractError):
                    validate_record(record, self.registration, self.fixture, 0)

    def test_unreviewed_producer_cannot_register_raw_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sidecar.json'
            fields = 'receipt publication retention capture fixture source_root source_commit request_index model_files consumer retire_after'
            sidecar = {key: None for key in fields.split()}
            sidecar.update(schema='strata.serve-first-row-sidecar.v1', producer='layer-major-unqualified')
            path.write_text(json.dumps(sidecar))
            with self.assertRaisesRegex(ContractError, 'unreviewed actual SERVE producer'):
                adapt_serve(path)

    def test_bad_authoritative_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sidecar.json'
            fields = 'publication retention capture fixture source_root source_commit request_index model_files consumer retire_after'
            sidecar = {key: None for key in fields.split()}
            sidecar.update(schema='strata.serve-first-row-sidecar.v1',
                           producer=next(iter(reviewed_producers())),
                           receipt={'path': 'invented.json', 'sha256': '0' * 64})
            path.write_text(json.dumps(sidecar))
            with self.assertRaisesRegex(ContractError, 'authoritative reviewed evidence hash'):
                adapt_serve(path)


class FaultAndLifecycleTests(unittest.TestCase):
    def test_fault_missing_coverage_and_changed_boot(self):
        labels = ('kernel-before', 'kernel-after-prehealth', 'kernel-after-model', 'kernel-posthealth')
        record = {'boot_id': 'boot', 'kernel_checks': {k: {'fault_entries': [],
                  'relevant_entries': [], 'dump_present': False, 'filter': 'fault',
                  'total_rows': 0} for k in labels},
                  'evidence': {k: {'stdout': ''} for k in labels}}
        validate_faults(record)
        for change in ('fault', 'missing', 'boot'):
            d = copy.deepcopy(record)
            if change == 'fault':
                d['kernel_checks']['kernel-after-model']['fault_entries'] = ['fault']
            elif change == 'missing':
                del d['kernel_checks']['kernel-posthealth']
            else:
                d['kernel_checks']['kernel-after-model']['total_rows'] = 1
                d['evidence']['kernel-after-model']['stdout'] = '{"_BOOT_ID":"other"}'
            with self.subTest(change=change), self.assertRaises(ContractError):
                validate_faults(d)

    def test_unreaped_or_active_owner_rejected(self):
        c = {k: True for k in ('observation_complete', 'direct_child_reaped',
                              'session_empty', 'session_anchored')}
        c.update(ownership_status='observed', errors=[], cleanup=[], survivors=[], exit_code=0,
                 pid=1, direct_identity={'pid': 1, 'sid': 1, 'pgid': 1, 'start_ticks': 2},
                 owners=[{'pid': 1, 'sid': 1, 'pgid': 1, 'start_ticks': 2}],
                 started_utc='2026-10-13T00:00:00+00:00', finished_utc='2026-10-13T00:00:01+00:00')
        closed_command(c)
        for key in ('observation_complete', 'direct_child_reaped', 'session_empty'):
            d = copy.deepcopy(c)
            d[key] = False
            with self.subTest(key=key), self.assertRaises(ContractError):
                closed_command(d)


class ReviewedActualRouteTests(unittest.TestCase):
    def test_actual_route_and_wrong_alias_name(self):
        registration = next(iter(reviewed_producers().values()))
        record = json.loads(Path(registration['receipt']['path']).read_text())
        fixture = ids(Path(registration['fixture']['path']).read_text())
        index = registration['request_index']
        request, _, _ = validate_record(record, registration, fixture, index)
        self.assertEqual(request['prompt_tokens'], 32768)
        bad = copy.deepcopy(record)
        value = bad['environment'].pop('STRATA_PREFILL_GATED_ALIAS')
        bad['environment']['STRATA_PHASE_PREFILL_GATED_ALIAS'] = value
        with self.assertRaisesRegex(ContractError, 'qualified effective route'):
            validate_record(bad, registration, fixture, index)


class RetainedConsumerTests(unittest.TestCase):
    def test_retired_siblings_are_optional_selected_row_must_be_current(self):
        retention = {'deletions': [{'path': 'request-1.bin'}, {'path': 'request-2.partial'}]}
        consumer = {'consumer': 'paired numerical row', 'retire_after': 'Main review'}
        validate_retained_consumer(retention, 'request-2.bin', consumer)
        with self.assertRaisesRegex(ContractError, 'selected row retired'):
            validate_retained_consumer(retention, 'request-1.bin', consumer)
        for key in ('consumer', 'retire_after'):
            d = dict(consumer)
            d[key] = ''
            with self.subTest(key=key), self.assertRaises(ContractError):
                validate_retained_consumer(retention, 'request-2.bin', d)

    def test_missing_selected_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                read_row(Path(directory) / 'request-2.bin', 3)


if __name__ == '__main__':
    unittest.main()
