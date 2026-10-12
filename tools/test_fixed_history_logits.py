# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Analytical fixtures; no inference or GPU dependency."""
import copy
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from fixed_history_logits import ContractError, compare, digest, ids_hash


class ComparatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.common = {'P': 2, 'N': 2, 'vocab': 3, 'max_context': 4,
                       'endian': 'little', 'stride': 1,
                       'position_convention': 'P-1+i predicts follow[i]',
                       'policy': {'bos': 'prompt-explicit', 'eos': 'score-all-stop-off',
                                  'mask': 'all-follow'},
                       'prompt_ids': [0, 1], 'follow_ids': [1, 2]}
        for key in ('prompt_ids', 'follow_ids'):
            self.common[key + '_sha256'] = ids_hash(self.common[key])
        for key in ('model', 'tokenizer'):
            self.common[key] = self.pin(key, key.encode())
        self.manifest = {'schema': 'strata.fixed-history-comparison.v1',
                         'common': self.common, 'consumer': 'analytical unittest',
                         'retire_after': 'temporary-directory close',
                         'budget': {'max_vocab': 3, 'max_rows': 2, 'max_capture_bytes': 32}}
        for name in ('baseline', 'candidate'):
            arm = {'common': copy.deepcopy(self.common)}
            for key in ('binary', 'source', 'runtime'):
                arm[key] = self.pin(name + key, key.encode())
            self.manifest[name] = arm
            self.capture(name, [[0, 0, 0], [0, 0, 0]])

    def pin(self, name, data):
        path = self.root / name
        path.write_bytes(data)
        return {'path': name, 'sha256': digest(path)}

    def capture(self, name, rows, header=None, extra=b''):
        header = header or (3, 2)
        endian = '<' if self.common['endian'] == 'little' else '>'
        data = struct.pack(endian + 'ii', *header)
        data += b''.join(struct.pack(endian + str(len(row)) + 'f', *row) for row in rows)
        arm = self.manifest[name]
        arm['capture'] = self.pin(name + '.logits', data + extra)
        receipt = {'schema': 'strata.fixed-history-capture-receipt.v1',
                   'bindings': {key: arm[key]['sha256'] for key in
                                ('capture', 'source', 'runtime', 'binary')},
                   'common': copy.deepcopy(self.common), 'positions': [1, 2],
                   'targets': [1, 2], 'observed_rows': 2,
                   'source_provider': 'standalone-native-segmented-spec-follow-dump-logits',
                   'lifecycle': {'observed': True, 'reaped': True, 'owned_processes_empty': True,
                                 'closed': True, 'capture_close_ok': True, 'normal_exit': True,
                                 'fault_check_complete': True, 'forced': False, 'canceled': False,
                                 'deadline_hit': False, 'budget_hit': False, 'exit_code': 0,
                                 'fault_count': 0, 'boot_before': 'fixture', 'boot_after': 'fixture',
                                 'owned_pid': 1, 'owned_start': 'fixture'}}
        for key in ('supervisor_evidence', 'fault_evidence', 'provider_evidence'):
            receipt[key] = self.pin(name + key, b'fixture only; not production proof')
        self.receipt(name, receipt)

    def receipt(self, name, value):
        self.manifest[name]['receipt'] = self.pin(name + '.receipt', json.dumps(value).encode())

    def run_comparison(self):
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(self.manifest))
        return compare(path)

    def reject(self):
        with self.assertRaises((ContractError, KeyError, ValueError)):
            self.run_comparison()

    def test_equal_and_shifted(self):
        self.capture('candidate', [[20, 20, 20], [-30, -30, -30]])
        result = self.run_comparison()
        self.assertAlmostEqual(result['aggregates']['baseline']['mean_nll'], math.log(3))
        self.assertAlmostEqual(result['aggregates']['candidate']['perplexity']['value'], 3)
        self.assertEqual(result['aggregates']['KL_baseline_to_candidate']['mean'], 0)
        self.assertEqual([r['baseline']['target_rank'] for r in result['rows']], [2, 3])
        self.assertFalse(result['quality_qualified'])

    def test_same_generated_top1_different_distribution(self):
        self.capture('baseline', [[math.log(4), math.log(2), 0]] * 2)
        self.capture('candidate', [[math.log(4), 0, 0]] * 2)
        result = self.run_comparison()
        row = result['rows'][0]
        self.assertTrue(row['top1_agrees'])
        self.assertAlmostEqual(row['baseline']['target_nll'], math.log(7 / 2), places=6)
        self.assertAlmostEqual(row['candidate']['target_nll'], math.log(6), places=6)
        self.assertGreater(row['KL_baseline_to_candidate'], 0)
        self.assertGreater(row['KL_candidate_to_baseline'], 0)
        self.assertGreater(row['target_nll_delta_candidate_minus_baseline'], 0)

    def test_different_argmax_valid(self):
        self.capture('candidate', [[0, 2, 0], [0, 0, 2]])
        self.assertEqual(self.run_comparison()['aggregates']['top1_agreement'], 0)

    def test_overflow(self):
        self.capture('candidate', [[0, -1000, 0], [0, 0, -1000]])
        ppl = self.run_comparison()['aggregates']['candidate']['perplexity']
        self.assertTrue(ppl['overflow'])
        self.assertIsNone(ppl['value'])
        self.assertGreater(ppl['log_value'], 1000)

    def test_binary_failures(self):
        for rows, header, extra in [([[0]*3], (3, 2), b''),
                                    ([[0]*3]*2, (3, 1), b''),
                                    ([[0]*3]*2, (4, 2), b''),
                                    ([[0]*3]*2, (-3, 2), b''),
                                    ([[0]*3]*2, (3, 2), b'x'),
                                    ([[0]*3, [0, 0, float('nan')]], (3, 2), b''),
                                    ([[0]*3, [0, 0, float('inf')]], (3, 2), b'')]:
            with self.subTest(rows=rows, header=header, extra=extra):
                self.capture('candidate', rows, header, extra)
                self.reject()

    def test_identity_policy_and_geometry_failures(self):
        saved = copy.deepcopy(self.manifest)
        mutations = [('follow_ids', [1]), ('follow_ids', [1, 3]),
                     ('follow_ids', [2, 1]), ('prompt_ids', [1, 0]),
                     ('P', 3), ('N', 3), ('max_context', 2), ('stride', 2),
                     ('endian', 'big'), ('position_convention', 'i+P'),
                     ('policy', {'mask': 'skip-eos'}),
                     ('tokenizer', self.common['model']), ('model', self.common['tokenizer'])]
        for key, value in mutations:
            with self.subTest(key=key):
                self.manifest = copy.deepcopy(saved)
                self.manifest['common'][key] = value
                self.reject()
        self.manifest = copy.deepcopy(saved)
        del self.manifest['candidate']['common']['tokenizer']
        self.reject()

    def test_receipt_failure_fields(self):
        arm = self.manifest['candidate']
        original = json.loads((self.root / arm['receipt']['path']).read_text())
        for key, value in original['lifecycle'].items():
            with self.subTest(key=key):
                altered = copy.deepcopy(original)
                altered['lifecycle'][key] = not value if type(value) is bool else None
                altered['passed'] = True
                self.receipt('candidate', altered)
                self.reject()
        for key, value in [('positions', [0, 1]), ('targets', [2, 1]),
                           ('observed_rows', 1), ('source_provider', 'serve'),
                           ('bindings', {}), ('schema', 'unknown')]:
            with self.subTest(key=key):
                altered = copy.deepcopy(original)
                altered[key] = value
                self.receipt('candidate', altered)
                self.reject()
        altered = copy.deepcopy(original)
        altered['supervisor_evidence']['sha256'] = '0' * 64
        self.receipt('candidate', altered)
        self.reject()

    def test_missing_fields_and_pin(self):
        del self.manifest['candidate']['receipt']
        self.reject()
        self.manifest['candidate']['receipt'] = {'path': 'absent', 'sha256': '0'*64}
        self.reject()

    def test_duplicate_json_and_exact_coverage(self):
        result = self.run_comparison()
        self.assertEqual([r['position'] for r in result['rows']], [1, 2])
        self.assertEqual([r['target'] for r in result['rows']], [1, 2])
        path = self.root / 'duplicate.json'
        path.write_text('{"schema":1,"schema":2}')
        with self.assertRaises(ContractError):
            compare(path)

    def test_valid_big_endian_and_analytical_kl(self):
        self.common['endian'] = 'big'
        for name in ('baseline', 'candidate'):
            self.manifest[name]['common'] = copy.deepcopy(self.common)
        self.capture('baseline', [[0, 0, 0]] * 2)
        self.capture('candidate', [[math.log(2), 0, 0]] * 2)
        result = self.run_comparison()
        forward = (math.log(2 / 3) + 2 * math.log(4 / 3)) / 3
        reverse = .5 * math.log(1.5) + .5 * math.log(.75)
        self.assertAlmostEqual(result['aggregates']['KL_baseline_to_candidate']['mean'], forward, places=6)
        self.assertAlmostEqual(result['aggregates']['KL_candidate_to_baseline']['mean'], reverse, places=6)

    def test_cli_valid_and_rejections(self):
        self.run_comparison()
        script = Path(__file__).with_name('fixed_history_logits.py')
        manifest = self.root / 'manifest.json'
        for args in ([], [str(manifest), 'extra'], [str(self.root / 'missing.json')]):
            with self.subTest(args=args):
                outcome = subprocess.run([sys.executable, str(script), *args],
                                         capture_output=True, text=True, timeout=10, check=False)
                self.assertNotEqual(outcome.returncode, 0)
                self.assertEqual(outcome.stdout, '')
        outcome = subprocess.run([sys.executable, str(script), str(manifest)],
                                 capture_output=True, text=True, timeout=10, check=False)
        self.assertEqual(outcome.returncode, 0, outcome.stderr)
        result = json.loads(outcome.stdout)
        self.assertFalse(result['production_capture_qualified'])
        self.assertFalse(result['quality_qualified'])


if __name__ == '__main__':
    unittest.main()
