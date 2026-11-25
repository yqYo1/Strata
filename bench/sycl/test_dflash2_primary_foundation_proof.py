"""Host checks for recovering the primary build foundation from Git."""

from pathlib import Path
import unittest
from unittest.mock import patch

import dflash2_primary_foundation_proof as foundation


class PrimaryFoundationProofTest(unittest.TestCase):
    def test_old_executables_are_not_read(self):
        primary = foundation.bank.previous.read(foundation.PRIMARY_ROOT / 'record.json')
        expected = primary['foundation']
        old_paths = {Path(entry['path']) for entry in expected['same_build_executables'].values()}
        self.assertEqual(len(old_paths), 12)

        def deny_old_executables(method):
            def checked(path, *args, **kwargs):
                self.assertNotIn(path, old_paths)
                return method(path, *args, **kwargs)
            return checked

        with (
            patch.object(Path, 'open', deny_old_executables(Path.open)),
            patch.object(Path, 'stat', deny_old_executables(Path.stat)),
        ):
            original, proof = foundation.historical_foundation(expected=expected)
        self.assertEqual(original['native_source_commit'], foundation.ORIGINAL.SOURCE)
        self.assertEqual(proof, expected)

    def test_changed_expected_proof_fails_admission(self):
        primary = foundation.bank.previous.read(foundation.PRIMARY_ROOT / 'record.json')
        changed = dict(primary['foundation'], original_recipe_sha256='invalid')
        with self.assertRaisesRegex(RuntimeError, 'pinned primary foundation changed'):
            foundation.historical_foundation(expected=changed)


if __name__ == '__main__':
    unittest.main()
