"""CPU-only admission checks for the retained nine-ELF fallback; repository MIT.

These exercise the actual installed build, not copied hashes or a mock build.
Fixture/map corruption is simulated at the digest boundary without modifying
the sole validated fallback. Retire with its generation-specific verifier.
"""
import fcntl
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import gdn_model_build as built


class RetainedBuildAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = (built.B / 'owned-v0141-measurement.lock').open('a+')
        fcntl.flock(cls.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cls.addClassCleanup(cls.lock.close)
        built.base.guards.no_model_owners([])
        cls.row = json.loads((built.ROOT / 'record.json').read_text())

    def test_verified_build_does_not_require_parent_executables(self):
        with patch.object(built.retained, 'verify', side_effect=AssertionError(
                'retired parent executable verifier was called')):
            actual = built.verify()
        self.assertEqual(actual['binaries'], self.row['binaries'])

    def reject_digest_change(self, path, message):
        original = built.digest

        def changed(value):
            return '0' * 64 if Path(value) == Path(path) else original(value)

        with patch.object(built, 'digest', side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, message):
                built.verify()

    def test_changed_same_build_fixture_is_rejected(self):
        self.reject_digest_change(self.row['binaries'][built.COMPONENT]['path'],
                                  'same-build engine/fixtures changed')

    def test_changed_actual_compiler_map_is_rejected(self):
        self.reject_digest_change(self.row['compiled_inputs']['path'],
                                  'actual compiler map changed')


if __name__ == '__main__':
    unittest.main()
