"""Host checks for recovering the primary build foundation from Git."""

from pathlib import Path
import unittest
from unittest.mock import patch

import dflash2_primary_foundation_proof as foundation
import dflash2_snapshot_source_recovery as snapshot
import dflash2_decode_route_census_build as census


class PrimaryFoundationProofTest(unittest.TestCase):
    def deny_paths(self, paths, method):
        def checked(path, *args, **kwargs):
            self.assertNotIn(path, paths)
            return method(path, *args, **kwargs)
        return checked

    def test_old_executables_are_not_read(self):
        primary = foundation.bank.previous.read(foundation.PRIMARY_ROOT / 'record.json')
        expected = primary['foundation']
        old_paths = {Path(entry['path']) for entry in expected['same_build_executables'].values()}
        self.assertEqual(len(old_paths), 12)
        with (
            patch.object(Path, 'open', self.deny_paths(old_paths, Path.open)),
            patch.object(Path, 'stat', self.deny_paths(old_paths, Path.stat)),
        ):
            original, proof = foundation.historical_foundation(expected=expected)
        self.assertEqual(original['native_source_commit'], foundation.ORIGINAL.SOURCE)
        self.assertEqual(proof, expected)

    def test_snapshot_recovery_without_legacy_executables(self):
        original = foundation.bank.previous.read(foundation.ORIGINAL.ROOT / 'record.json')
        target = foundation.bank.previous.read(snapshot.ROOT / 'record.json')
        old_paths = {Path(entry['path']) for row in (original, target) for entry in row['binaries'].values()}
        self.assertEqual(len(old_paths), 24)
        with (
            patch.object(Path, 'open', self.deny_paths(old_paths, Path.open)),
            patch.object(Path, 'stat', self.deny_paths(old_paths, Path.stat)),
        ):
            proof = snapshot.recover()
        self.assertEqual(proof['recovered_source_files'], 1171)
        self.assertEqual(proof['compiled_project_files'], 279)
        self.assertEqual(proof['original_binary_sha256'], target['binaries']['strata']['sha256'])

    def test_changed_expected_proof_fails_admission(self):
        primary = foundation.bank.previous.read(foundation.PRIMARY_ROOT / 'record.json')
        changed = dict(primary['foundation'], original_recipe_sha256='invalid')
        with self.assertRaisesRegex(RuntimeError, 'pinned primary foundation changed'):
            foundation.historical_foundation(expected=changed)

    def test_decode_census_uses_the_archived_decode_identity(self):
        before = census.bank.previous.historic(census.primary.DECODE, 'dflash2_snapshot_regular_code_v1')
        path = Path(before['binary']['path'])
        with (
            patch.object(Path, 'open', self.deny_paths({path}, Path.open)),
            patch.object(Path, 'stat', self.deny_paths({path}, Path.stat)),
        ):
            primary, controls = census.phase_references()
        self.assertEqual(controls['decode'], before['binary'])
        self.assertEqual(controls['prefill'], primary['binaries']['strata'])


if __name__ == '__main__':
    unittest.main()
