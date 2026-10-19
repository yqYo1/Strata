#!/usr/bin/env python3
"""Original-source and relocated-binary admission; no real Git/GPU execution."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import resource_engine as r


class SourceProofTests(unittest.TestCase):
    def test_relocation_requires_original_committed_reference_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);native=root/'retained-strata';native.write_bytes(b'engine')
            binary=dict(path=str(native),bytes=6,sha256=hashlib.sha256(b'engine').hexdigest())
            source=b'original source';recipe=b'original recipe'
            full=dict(source_commit='native-revision',binary=dict(binary,path='/original/build/strata'),
                project=dict(root='/project',revision='native-revision',source_sha256={'src/test.cpp':hashlib.sha256(source).hexdigest()}),
                dependencies=[],recipe_reference=dict(path='recipe.json',sha256=hashlib.sha256(recipe).hexdigest()))
            receipt=root/'source-build.json';receipt.write_text(json.dumps(full))
            receipt_pin=dict(path=str(receipt),sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
            reference=json.dumps(dict(retained_validation_reference=dict(source_commit=full['source_commit'],
                binary=binary,source_build_record=receipt_pin))).encode()
            record=dict(source_build_record=receipt_pin,build_source_commit=full['source_commit'],binary=binary,
                fallback_reference=dict(commit='relocation-revision',path='build.json',sha256=hashlib.sha256(reference).hexdigest()))
            def git(*args):
                return reference if args[-1]=='relocation-revision:build.json' else recipe
            with patch.object(r.generation,'git',side_effect=git),patch.object(r,'blobs',return_value=[('src/test.cpp',source)]):
                self.assertEqual(r.source_proof(record),1)
                bad=dict(record,fallback_reference=dict(record['fallback_reference'],sha256='0'*64))
                with self.assertRaises(RuntimeError):r.source_proof(bad)
                bad=dict(record,build_source_commit='other-source')
                with self.assertRaises(RuntimeError):r.source_proof(bad)
                native.write_bytes(b'changed engine')
                with self.assertRaises(RuntimeError):r.source_proof(record)
            with patch.object(r.generation,'git',side_effect=git),patch.object(r,'blobs',return_value=[('src/test.cpp',b'wrong source')]):
                with self.assertRaises(RuntimeError):r.source_proof(record)


if __name__=='__main__':unittest.main()
