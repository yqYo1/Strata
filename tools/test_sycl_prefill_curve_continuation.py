#!/usr/bin/env python3
"""Host-only continuation/source guards; repository MIT. No GPU execution."""
import copy
import fcntl
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'sycl'))
import cache_map_prefill_curve as curve


class ContinuationGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = (curve.B/'owned-v0141-measurement.lock').open('a+')
        cls.addClassCleanup(cls.lock.close)
        fcntl.flock(cls.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        curve.base.guards.no_model_owners([])
        cls.revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        _, old, _ = curve.prefix_evidence()
        fields = ('binary', 'native_source_commit', 'native_validation_probe', 'configuration',
                  'environment_template', 'python_contract', 'python_selected', 'template',
                  'boot', 'driver', 'plans', 'plan')
        cls.row = {name: copy.deepcopy(old[name]) for name in fields}
        cls.row.update(sources=curve.pinned_sources(cls.revision), processes=[], request_hashes={})
        curve.inherit_prefix(cls.row)

    def test_original_parser_and_inherited_controls(self):
        self.assertTrue(curve.selfcheck()['passed'])
        self.assertTrue(curve.parser_qualification()['passed'])
        self.assertEqual(curve.ROOT.name, 'prefill-curve-v6')
        self.assertEqual(curve.KEY, 'cache_map_prefill_curve_v6')
        curve.prefix_qualification(self.row, check=True)
        self.assertEqual([p['index'] for p in self.row['processes']], [0, 1, 2])

    def test_source_mutations_rejected(self):
        source = Path(curve.__file__).read_text()
        expected = curve.scope_projection(source)
        cases = [
            ("n = plan['read'] - 1", "n = plan['read'] - 2"),
            ('(39254, True)', '(39255, True)'),
            ("STRATA_SYCL_PREFILL_GDN_FP16='1'", "STRATA_SYCL_PREFILL_GDN_FP16='0'"),
            ('import math\n', 'import math\nimport decimal\n'),
            ('import ast\n', 'import ast as renamed_ast\n'),
            ("ROOT = entry.model.ROOT / 'prefill-curve-v6'", "ROOT = entry.model.ROOT / str('prefill-curve-v6')"),
            ("\nKEY = 'cache_map_prefill_curve_v6'\n", "\nKEY = 'cache_map_prefill_curve_v9'\n"),
            ("\nKEY = 'cache_map_prefill_curve_v6'\n", "\nKEY = 'cache_map_prefill_curve_v5'\n"),
            ("'tools/sycl/official_model_entry.py', 'tools/sycl/gdn_model_sources.py')",
             "'tools/sycl/official_model_entry.py', 'tools/sycl/other.py')"),
            ("HOST_TESTS = ('tools/test_sycl_gdn_model_sources.py',",
             "HOST_TESTS = ('tools/unplanned_test.py',"),
            ('    inherit_prefix(row)\n', ''),
            ('    inherit_prefix(row)\n', '    inherit_prefix(row)\n    inherit_prefix(row)\n'),
            ('def current(row):\n    prefix_qualification(row, check=True)\n', 'def current(row):\n'),
            ('mtp = entry.template.mtp_route(text, n)', 'mtp = mtp_boundary_route(text, n)'),
        ]
        for before, after in cases:
            self.assertEqual(source.count(before), 1, before)
            with self.subTest(before=before):
                try:
                    accepted = curve.scope_projection(source.replace(before, after)) == expected
                except (RuntimeError, ValueError, SyntaxError):
                    accepted = False
                self.assertFalse(accepted)

    def test_prefix_mutations_rejected(self):
        cases = ['binary', 'boot', 'configuration', 'source', 'selection', 'command',
                 'sample', 'exit_type', 'process_count', 'request_body', 'proof_type']
        for name in cases:
            row = copy.deepcopy(self.row)
            if name == 'binary': row['binary']['sha256'] = '0'*64
            if name == 'boot': row['boot'] = 'different-boot'
            if name == 'configuration': row['configuration']['unplanned_setting'] = True
            if name == 'source': row['sources'][curve.SOURCE] = '0'*64
            if name == 'selection': row['plans'][0]['trace_progress'] = 1
            if name == 'command': row['plans'][0]['command'].append('--unplanned')
            if name == 'sample': row['processes'][0]['samples'][1]['prompt_ms'] += 1
            if name == 'exit_type': row['processes'][0]['native_exit_code'] = False
            if name == 'process_count': row['processes'].pop()
            if name == 'request_body': row['request_hashes']['1286'][0] = '0'*64
            if name == 'proof_type': row['inherited_prefix']['freshly_executed'] = 0
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                curve.prefix_qualification(row, check=True)

    def test_original_replay_proof_mutations_rejected(self):
        real = curve.base.committed
        for name, value in [('passed', 1), ('GPU_executed', 0), ('controller_sha256', '0'*64)]:
            data = copy.deepcopy(real(curve.RESULT))
            data[curve.PARSER_KEY][name] = value
            with self.subTest(name=name), patch.object(curve.base, 'committed', return_value=data), \
                    self.assertRaises(RuntimeError):
                curve.parser_qualification()


class MTPBoundaryRoute(unittest.TestCase):
    # Reduced callback markers from failed V6/read65537, evidence86c02f511.
    captured = ''.join('strata serve: DRAFT_PREFILL path='+path+' mode=2 cells=8192\n'
                       for path in ['batched']*3+['token']+['batched']*4)

    @classmethod
    def setUpClass(cls):
        cls.lock = (curve.B/'owned-v0141-measurement.lock').open('a+')
        cls.addClassCleanup(cls.lock.close)
        fcntl.flock(cls.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        curve.base.guards.no_model_owners([])

    def route(self, text, rows=65536):
        fn = getattr(curve, 'mtp_boundary_route', curve.entry.template.mtp_route)
        return fn(text, rows)

    def test_actual_mixed_callback(self):
        result = self.route(self.captured)
        self.assertEqual(result['fallback_callback_index'], 3)
        self.assertEqual(result['source_derived_active_rows'], 63)
        self.assertFalse(result['internal_needed_rows_and_shapes_observed'])

    def test_boundary_capacity_matrix(self):
        for read, first, active in [(40959, 0, 65), (40960, 0, 64), (40961, 0, 63),
                                    (65535, 24576, 65), (65536, 24576, 64), (65537, 24576, 63)]:
            for workspace in (64, 128):
                for ring in (64, 128):
                    with self.subTest(read=read, workspace=workspace, ring=ring):
                        q = curve.mtp_boundary_batch(read, first, 8192, workspace, ring)
                        expected = active if active<=64 or workspace==ring==128 else 64
                        self.assertEqual(q['active_rows'], active)
                        self.assertEqual(q['first_batch_rows'], expected)
                        self.assertEqual(q['path'], 'token' if active==63 else 'batched')
        q = curve.mtp_boundary_batch(65537, 0, 8192, 64, 64)
        self.assertTrue(q['skipped']);self.assertIsNone(q['first_batch_rows'])

    def test_bad_callback_sequences_rejected(self):
        batch='strata serve: DRAFT_PREFILL path=batched mode=2 cells=8192\n'
        token='strata serve: DRAFT_PREFILL path=token mode=2 cells=8192\n'
        bad = [self.captured.replace(token,batch),self.captured.replace(batch,token,1),
               self.captured.replace(token,'').replace(batch,token,1)+batch,
               self.captured.replace('cells=8192','cells=8191',1),
               self.captured.replace('mode=2','mode=1',1),self.captured+batch,
               self.captured.replace(token,''),self.captured.replace(token,token.replace('mode=2','mode=2 mode=2'))]
        for i, text in enumerate(bad):
            with self.subTest(mutation=i), self.assertRaises(RuntimeError):
                self.route(text)
        with self.assertRaises(RuntimeError):self.route(self.captured,65535)

    def test_other_lengths_keep_original_contract(self):
        for rows in (1285,4955,8192,24576,32768,117488):
            text=''.join(f'strata serve: DRAFT_PREFILL path=batched mode=2 cells={min(8192,rows-i)}\n'
                         for i in range(0,rows,8192))
            self.assertEqual(self.route(text,rows),curve.entry.template.mtp_route(text,rows))


if __name__ == '__main__':
    unittest.main(verbosity=2)
