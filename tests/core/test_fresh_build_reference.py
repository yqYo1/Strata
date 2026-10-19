#!/usr/bin/env python3
"""Retired candidate cache must not discard or widen fallback qualification."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import build_engine as b


class FreshReferenceTests(unittest.TestCase):
    def test_fresh_cache_carries_only_the_original_validated_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);native=root/'strata';native.write_bytes(b'engine')
            artifact=dict(path=str(native),bytes=6,sha256=hashlib.sha256(b'engine').hexdigest())
            fixture=root/'provider_test';fixture.write_bytes(b'fixture')
            fixture_artifact=dict(path=str(fixture),bytes=7,sha256=hashlib.sha256(b'fixture').hexdigest())
            source=b'original source';recipe=b'original recipe'
            full=dict(source_commit='native-revision',binary=dict(artifact,path='/original/strata'),
                fixtures=dict(provider_test=dict(path='/original/provider_test',sha256=fixture_artifact['sha256'])),
                project=dict(root='/project',revision='native-revision',source_sha256={'src/test.cpp':hashlib.sha256(source).hexdigest()}),
                dependencies=[],recipe_reference=dict(path='recipe.json',sha256=hashlib.sha256(recipe).hexdigest()))
            receipt=root/'source-build.json';receipt.write_text(json.dumps(full))
            reference=dict(source_commit=full['source_commit'],binary=artifact,fixtures=dict(provider_test=fixture_artifact),
                source_build_record=dict(path=str(receipt),sha256=hashlib.sha256(receipt.read_bytes()).hexdigest()))
            compact=dict(retirement=dict(complete=True),retained_validation_reference=reference)
            with patch.object(b,'git',return_value=recipe),patch.object(b,'blobs',return_value=[('src/test.cpp',source)]),patch.object(b,'ROOT',root/'new-cache'):
                self.assertEqual(b.fresh_retained_reference(compact),reference)
                for variant in ('unretired','extra_fixture','missing_fixture','wrong_source','wrong_hash','wrong_size','inside_cache'):
                    with self.subTest(variant=variant):
                        changed=copy.deepcopy(compact)
                        if variant=='unretired':changed['retirement']['complete']=False
                        if variant=='extra_fixture':changed['retained_validation_reference']['fixtures']['new_CPU_fixture']=artifact
                        if variant=='missing_fixture':changed['retained_validation_reference']['fixtures'].clear()
                        if variant=='wrong_source':changed['retained_validation_reference']['source_commit']='other'
                        if variant=='wrong_hash':changed['retained_validation_reference']['binary']['sha256']='0'*64
                        if variant=='wrong_size':changed['retained_validation_reference']['fixtures']['provider_test']['bytes']=8
                        if variant=='inside_cache':changed['retained_validation_reference']['binary']['path']=str(root/'new-cache/strata')
                        with self.assertRaises((RuntimeError,KeyError)):b.fresh_retained_reference(changed)


if __name__=='__main__':unittest.main()
