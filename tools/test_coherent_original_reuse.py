#!/usr/bin/env python3
"""CPU-only original-artifact binding controls; never start a native child."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import coherent_original_reuse as reader


class OriginalReferenceTests(unittest.TestCase):
    def test_exact_original_snapshot_and_frozen_proof(self):
        row,proof=reader.verify_original()
        self.assertEqual(proof['original_native_source'],reader.SOURCE)
        self.assertEqual(len(row['binaries']),12)
        self.assertEqual(proof,reader.verify_original(expected=proof)[1])
        wrong=dict(proof,original_native_source='0'*40)
        with self.assertRaises(RuntimeError):reader.verify_original(expected=wrong)

    def test_source_and_same_hash_other_artifact_cannot_replace_original(self):
        row=json.loads((reader.ROOT/'record.json').read_text())
        mutations=[lambda x:x.update(native_source_commit='0'*40),
            lambda x:x['binaries']['strata'].update(path=str(reader.B/'gu-small-model-native-v2/strata')),
            lambda x:x['actual_inputs']['class_definition_coherence'].update(partial_archive_reuse=True)]
        for i,mutate in enumerate(mutations):
            bad=copy.deepcopy(row);mutate(bad)
            with self.subTest(case=i),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'record.json').write_text(json.dumps(bad))
                with mock.patch.object(reader,'ROOT',root),self.assertRaises(RuntimeError):
                    reader.verify_original()


if __name__=='__main__':unittest.main()
