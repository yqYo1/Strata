"""Use the measured primary engine while blocking access to legacy binaries."""

from pathlib import Path
import unittest
from unittest.mock import patch

import dflash2_primary_continuation as continuation
import dflash2_primary_retained as retained


class RetainedPrimaryTest(unittest.TestCase):
    def test_baseline_equivalence_without_opening_legacy_binaries(self):
        reference = continuation.curve.baseline()
        build = retained.bank.previous.read(retained.PRIMARY.ROOT / 'record.json')
        inventories = (build['retained_references'], build['foundation']['same_build_executables'])
        legacy = {Path(row['path']) for inventory in inventories for row in inventory.values()}
        self.assertEqual(len(legacy), 14)

        def guard(method):
            def checked(path, *args, **kwargs):
                self.assertNotIn(path, legacy)
                return method(path, *args, **kwargs)
            return checked

        with patch.object(Path, 'open', guard(Path.open)), patch.object(Path, 'stat', guard(Path.stat)):
            self.assertEqual(continuation.experiment().baseline(), reference)

    def test_mismatched_full_context_receipt_refuses_admission(self):
        historic = retained.bank.previous.historic

        def changed(revision, key):
            row = historic(revision, key)
            if revision == retained.CONTEXT:
                row['outcomes']['full-copy']['final_slot_consumed'] = False
            return row

        with patch.object(retained.bank.previous, 'historic', changed):
            with self.assertRaisesRegex(RuntimeError, 'physical-context evidence'):
                retained.baseline()


if __name__ == '__main__':
    unittest.main()
