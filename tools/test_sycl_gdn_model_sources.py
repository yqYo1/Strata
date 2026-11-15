#!/usr/bin/env python3
"""Host-only regression for the selected GDN source admission; repository MIT.

Exercise the complete production projection against its retained 558-input
record. No compiler, model execution, runtime initialization or GPU work.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'sycl'))
import build_cache_map_model as model
import gdn_model_sources as source


class GdnSourceAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = (source.B / 'owned-v0141-measurement.lock').open('a+')
        cls.addClassCleanup(cls.lock.close)
        fcntl.flock(cls.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source.base.guards.no_model_owners([])
        cls.original = source.blob(model.prior.SELECTED, source.CMAKE)
        cls.current = (source.P / source.CMAKE).read_bytes()
        cls.expected = json.loads((model.ROOT / 'record.json').read_text())['native_sources']
        if type(cls.expected) is not dict or len(cls.expected) != 558:
            raise RuntimeError('use the original full 558-input map,not a compact summary')

    def project(self, cmake):
        read = Path.read_bytes
        def selected(path):
            return cmake if path == source.P / source.CMAKE else read(path)
        with patch.object(Path, 'read_bytes', selected):
            _, native = model.projection(worker=True)
        self.assertEqual({name: hashlib.sha256(raw).hexdigest()
                          for name, raw in native.items()}, self.expected)

    def test_original_projection(self):
        self.project(self.original)

    def test_current_unrelated_targets(self):
        self.assertNotEqual(self.current, self.original)
        self.assertEqual(self.current.split(source.MARKER)[1],
                         self.original.split(source.MARKER)[1])
        self.project(self.current)

    def test_unconsumed_prelude(self):
        self.project(b'# unrelated future manual target\n' + self.original)

    def test_comment_is_not_another_target(self):
        self.project(b'# add_executable(sycl_gdn_storage_pipeline_bench ignored)\n'
                     + self.original)

    def test_consumed_contract_still_rejects_changes(self):
        before, tail = self.original.split(source.MARKER)
        cases = {
            'missing_marker': before + tail,
            'duplicate_marker': self.original + source.MARKER,
            'fixture': before + source.MARKER + tail.replace(
                b'bench/sycl/gdn_storage_pipeline.cpp', b'bench/sycl/other.cpp'),
            'link': before + source.MARKER + tail.replace(b'PRIVATE strata_prefill', b'PRIVATE strata_core'),
            'math': before + source.MARKER + tail.replace(b'-fno-fast-math', b'-ffast-math'),
            'target': before + source.MARKER + tail.replace(
                b'sycl_gdn_storage_pipeline_bench', b'changed_pipeline'),
            'duplicate_target': b'add_executable(sycl_gdn_storage_pipeline_bench other.cpp)\n'
                + self.original,
            'duplicate_upper_command': b'ADD_EXECUTABLE(sycl_gdn_storage_pipeline_bench other.cpp)\n'
                + self.original,
            'extra_consumed_suffix': self.original + b'# not part of the frozen appendix\n',
        }
        self.assertEqual(len(set(cases.values())), len(cases))
        for name, cmake in cases.items():
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                self.project(cmake)


if __name__ == '__main__':
    unittest.main(verbosity=2)
