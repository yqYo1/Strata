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


if __name__ == '__main__':
    unittest.main(verbosity=2)
