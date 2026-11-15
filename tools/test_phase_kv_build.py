"""Fail-closed controls for the separately selected KV build units.

Main-authored; repository LICENSE. No compiler, device or engine is executed.
"""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent / 'sycl'))
import build_named_kv as build


class BuildUnits(unittest.TestCase):
    def tearDown(self):
        build.select_unit('naming')

    def test_explicit_selection_can_return_to_original_unit(self):
        original = build.ROOT, build.KEY, build.SEED, build.SEED_KEY
        build.select_unit('split')
        self.assertNotEqual(build.ROOT, original[0])
        self.assertEqual(build.SEED_KEY, 'phase_kv_naming_native_v1')
        build.select_unit('naming')
        self.assertEqual((build.ROOT, build.KEY, build.SEED, build.SEED_KEY), original)

    def test_unknown_unit_is_rejected_without_changing_selection(self):
        with self.assertRaises(RuntimeError): build.select_unit('combined')
        self.assertEqual(build.MODE, 'naming')

    def test_split_rejects_an_additional_independent_source_change(self):
        with patch.object(build.previous.base.common.source, 'git',
                          return_value=b'src/kernels/sycl/other.cpp\n'):
            with self.assertRaisesRegex(RuntimeError, 'three-file B1'):
                build.split_unit_sources()

    def test_original_split_has_only_the_three_native_changes(self):
        # Real immutable Git inputs catch a guard accidentally counting the
        # independent uncompiled observer commits between A and B1.
        source = build.previous.base.common.source
        all_names = set(source.git(build.P, 'diff', '--name-only', build.UNIT,
                                   build.SPLIT_UNIT).decode().splitlines())
        native_names = set(source.git(build.P, 'diff', '--name-only', build.UNIT,
                                      build.SPLIT_UNIT, '--',
                                      *build.SOURCE_PREFIXES).decode().splitlines())
        self.assertEqual(native_names, build.SPLIT_FILES)
        self.assertGreater(len(all_names), len(native_names))

    def test_unclosed_naming_control_blocks_split_admission(self):
        gate = dict(stage='running', active=False, passed=True, complete=True,
                    ownership_closed=True, decision=dict(naming_control_eligible=True))
        data = {'phase_kv_naming_prefill_v1': gate}
        with patch.object(build.previous.base.common.source, 'git',
                          return_value=json.dumps(data).encode()):
            with self.assertRaisesRegex(RuntimeError, 'both original A phase controls'):
                build.split_prerequisite()

    def test_failed_naming_performance_control_blocks_split_admission(self):
        gate = dict(stage='closed', active=False, passed=True, complete=True,
                    ownership_closed=True, decision=dict(naming_control_eligible=False))
        data = {'phase_kv_naming_prefill_v1': gate}
        with patch.object(build.previous.base.common.source, 'git',
                          return_value=json.dumps(data).encode()):
            with self.assertRaisesRegex(RuntimeError, 'both original A phase controls'):
                build.split_prerequisite()


if __name__ == '__main__': unittest.main()
