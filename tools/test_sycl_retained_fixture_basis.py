#!/usr/bin/env python3
"""Host-only qualification/retirement guards; repository LICENSE. No GPU work."""
import ast
import builtins
import copy
import fcntl
import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'sycl'))
import cache_map_model_entry as entry


class RetainedFixtureBasis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = (entry.B/'owned-v0141-measurement.lock').open('a+')
        cls.addClassCleanup(cls.lock.close)
        fcntl.flock(cls.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        entry.base.guards.no_model_owners([])
        cls.revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        cls.native = entry.model.verify(reuse_revision=cls.revision,
                                        controller_changes=entry.NATIVE_CONTROLS)

    def test_device_changes_only_admission_seams(self):
        old = entry.base.basis.build.git(entry.P, 'show', entry.HISTORY_COMMIT +
                    ':tools/sycl/checked_worker_device.py').decode()
        new = Path(entry.device.__file__).read_text()
        old_tree, new_tree = ast.parse(old), ast.parse(new)
        for name, replacements in {
            'sources': [('pins = base.controls(revision, controller=Path(__file__).resolve(),\n'
                         '                        reuse_revision=reuse_revision, controller_changes=controller_changes)',
                         'pins = base.controls(revision, controller=Path(__file__).resolve())')],
            'current': [('sources(row[\'source_commit\'], reuse_revision=reuse_revision,\n'
                         '                    controller_changes=controller_changes)', "sources(row['source_commit'])"),
                        ('native = model.verify(reuse_revision=reuse_revision, controller_changes=native_controller_changes)',
                         'native = model.verify()')]
        }.items():
            before = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            after = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            fragment = ast.get_source_segment(new, after)
            for a, b in replacements:
                self.assertEqual(fragment.count(a), 1)
                fragment = fragment.replace(a, b)
            normalized = ast.parse(fragment).body[0]
            normalized.args = copy.deepcopy(before.args)
            new_tree.body[new_tree.body.index(after)] = normalized
        self.assertEqual(ast.dump(new_tree, include_attributes=False),
                         ast.dump(old_tree, include_attributes=False))

    def test_complete_proof_never_reads_legacy_generation(self):
        root = str(entry.model.prior.ROOT) + os.sep
        original_open, original_io_open = builtins.open, io.open
        def guarded(original):
            def run(path, *args, **kwargs):
                if isinstance(path, (str, bytes, os.PathLike)):
                    self.assertFalse(os.path.abspath(os.fsdecode(path)).startswith(root), str(path))
                return original(path, *args, **kwargs)
            return run
        with patch.object(builtins, 'open', guarded(original_open)), \
                patch.object(io, 'open', guarded(original_io_open)):
            native, gate, providers = entry.proof()
        self.assertEqual(native['source_commit'], self.native['source_commit'])
        self.assertEqual(gate['mathematical_fixture_basis']['mathematical_fixtures'], 8)
        self.assertFalse(gate['mathematical_fixture_basis']['old_build_products_reopened'])
        self.assertGreater(len(providers), 0)

    def test_invalid_fixture_or_recipe_rejected(self):
        cases = ['fixture_hash', 'fixture_size', 'native_source', 'worker_missing',
                 'configure', 'environment', 'link_input', 'engine', 'database']
        name = next(n for n in self.native['binaries'] if n != 'strata')
        for case in cases:
            value = copy.deepcopy(self.native)
            if case == 'fixture_hash': value['binaries'][name]['sha256'] = '0'*64
            if case == 'fixture_size': value['binaries'][name]['bytes'] += 1
            if case == 'native_source': value['native_sources']['unplanned.cpp'] = '0'*64
            if case == 'worker_missing': value['native_sources'].pop(entry.model.WORKER_HEADER)
            if case == 'configure': value['configure_argv'].append('--unplanned')
            if case == 'environment': value['environment']['STRATA_UNPLANNED'] = '1'
            if case == 'link_input': value['actual_link_inputs']['frozen_inputs']['unplanned.a'] = '0'*64
            if case == 'engine': value['binaries']['strata']['sha256'] = '0'*64
            if case == 'database': value['compile_database']['sha256'] = '0'*64
            with self.subTest(case=case), self.assertRaises(RuntimeError):
                entry.mathematical_fixture_basis(value)


if __name__ == '__main__':
    unittest.main(verbosity=2)
