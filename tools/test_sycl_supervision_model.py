#!/usr/bin/env python3
"""CPU-only admission corruption controls; repository LICENSE."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'sycl'))
import supervision_model as control


class PlacementInput(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='profile-admission-',
            dir=os.environ.get('STRATA_SUPERVISION_TEST_ROOT'))
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name); self.path = self.root / 'profile.bin'
        self.original = b'committed placement bytes'; self.path.write_bytes(self.original)
        self.config = dict(args=['--expert-profile', str(self.path)])
        self.addCleanup(patch.stopall)
        patch.object(control, 'P', self.root).start()
        patch.object(control.gen, 'git', return_value=self.original).start()

    def pin(self, expected=None):
        return control.profile_pin(self.config, 'source-revision', expected)

    def test_unchanged_committed_profile(self):
        pin = self.pin(); self.assertEqual(self.pin(pin), pin)
        self.assertEqual(pin['identity']['bytes'], len(self.original))

    def test_rewrite_with_restored_mtime_is_rejected(self):
        pin = self.pin(); old = self.path.stat()
        self.path.write_bytes(b'X' * len(self.original))
        os.utime(self.path, ns=(old.st_atime_ns, old.st_mtime_ns))
        with self.assertRaises(RuntimeError): self.pin(pin)

    def test_equal_byte_path_replacement_is_rejected(self):
        pin = self.pin(); other = self.root / 'other.bin'; other.write_bytes(self.original)
        self.config['args'][-1] = str(other)
        with self.assertRaises(RuntimeError): self.pin(pin)

    def test_missing_or_duplicate_profile_flag_is_rejected(self):
        for flags in ([], ['--expert-profile'], ['--expert-profile', str(self.path)] * 2):
            with self.subTest(flags=flags), self.assertRaises(RuntimeError):
                control.profile_pin(dict(args=flags), 'source-revision')

    def test_process_admission_stops_before_native_qualification(self):
        row = {key: {} for key in ('binary', 'qualification', 'model_assets', 'probe')}
        row.update(configuration=self.config, environment_template={}, template_sha256='template',
            runtime_sha256='runtime', NEO='provider', boot='boot', request_hashes=[], profile=self.pin())
        row['plan'] = dict(source_commit='source-revision', qualified_input_identity=control.input_identity(row))
        self.path.write_bytes(b'changed profile')
        with patch.object(control.runtime.validation, 'committed', return_value=row['plan']), \
                patch.object(control.gen, 'committed', return_value=dict(environment_template={})), \
                patch.object(control, 'qualification') as native:
            with self.assertRaises(RuntimeError): control.current(row)
            native.assert_not_called()


if __name__ == '__main__': unittest.main(failfast=True)
