#!/usr/bin/env python3
"""Refuse the concrete unbound-environment/omitted-source failures; no GPU."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
import build_manifest as m


class BuildMetadata(unittest.TestCase):
    def setUp(self):
        self.row = dict(project_inputs={'src/core/mtp.cpp': 'a'*64},
            generated_inputs={'src/core/mtp.cpp': 'b'*64},
            dependency_inputs=[{'root':'/original/ggml','revision':'c'*40}],
            external_inputs={'/opt/mkl/lib/libmkl.so':'d'*64},
            environment={'EnableDirectSubmission':'0','TMPDIR':'/private/build/tmp'},
            compiler_inputs={'sdk':{'sha256':'e'*64},'tools':{'cmake':{'sha256':'f'*64}}})
        self.expected = m.pins(self.row)

    def test_committed_inputs_accept_without_mutation(self):
        before = copy.deepcopy(self.row)
        m.verify(self.row, self.expected)
        self.assertEqual(self.row, before)

    def test_changed_safety_environment_refuses(self):
        self.row['environment']['EnableDirectSubmission'] = '1'
        with self.assertRaises(RuntimeError): m.verify(self.row, self.expected)

    def test_removed_native_source_pin_refuses(self):
        del self.row['generated_inputs']['src/core/mtp.cpp']
        with self.assertRaises(RuntimeError): m.verify(self.row, self.expected)

    def test_every_omitted_map_and_changed_dependency_refuses(self):
        for field in m.FIELDS:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.row); changed.pop(field)
                with self.assertRaises(RuntimeError): m.verify(changed, self.expected)
        self.row['dependency_inputs'][0]['root'] = '/other/ggml'
        with self.assertRaises(RuntimeError): m.verify(self.row, self.expected)

    def test_changed_tool_library_or_source_hash_refuses(self):
        for field in ('compiler_inputs','external_inputs','project_inputs','generated_inputs'):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.row); changed[field] = {}
                with self.assertRaises(RuntimeError): m.verify(changed, self.expected)

    def test_incomplete_or_untyped_committed_evidence_refuses(self):
        for expected in ({}, [], dict(self.expected,environment=True),
                         dict(self.expected,environment='g'*64)):
            with self.subTest(expected=expected), self.assertRaises(RuntimeError): m.verify(self.row, expected)


if __name__ == '__main__': unittest.main()
