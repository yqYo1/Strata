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


class RetainedModelEvidenceTests(unittest.TestCase):
    """A reused build pathname must not admit an older engine's model results."""

    def setUp(self):
        self.build = dict(source_commit='native-source',
                          binary=dict(path='/reused/build/strata', bytes=20, sha256='a'*64))
        identity = dict(build_source_commit=self.build['source_commit'],
                        binary=copy.deepcopy(self.build['binary']), passed=True,
                        complete=True, active=False)
        observation = dict(passed=True, complete=True, ownership_closed=True, exit_code=0,
            boot_unchanged=True, fault_probe_successful=True, fresh_faults=[],
            sample=dict(ok=True, eligible=True, stream_complete=True,
                        actual_read_tokens=117489, actual_prompt_tokens=117489,
                        actual_reused_tokens=0))
        process = dict(passed=True, ownership_closed=True, native_exit_code=0, journal_valid=True,
            fault_probe=dict(fresh_faults=[], boot_unchanged=True),
            request=dict(ok=True, stream_complete=True, truncated=False,
                         read_tokens=33208, prompt_tokens=33208, reused_tokens=0,
                         generated_tokens=1500, generated_tokens_source='usage',
                         completion_includes_thinking=True, output_chars=1000, reasoning_chars=100))
        self.prefill = dict(identity, full_model_hashes_matched=15,
                            runs=[copy.deepcopy(observation) for _ in range(2)])
        self.generation = dict(copy.deepcopy(identity), stage='closed', ownership_closed=True,
                               processes=[copy.deepcopy(process) for _ in range(6)])

    def test_retention_does_not_promote_execution_to_clean_speed_or_model_adoption(self):
        scope = b.retained_model_evidence(self.build, self.prefill, self.generation)
        self.assertEqual(scope['prefill_read_tokens'], [117489])
        self.assertEqual(scope['generation_read_tokens'], [33208])
        self.assertFalse(scope['clean117K_speed_qualified'])
        self.assertFalse(scope['model_adoption_qualified'])

    def test_prior_source_or_binary_at_the_same_build_path_is_rejected(self):
        for field in ('prefill', 'generation'):
            for variant in ('source', 'binary', 'incomplete', 'failed', 'active'):
                with self.subTest(field=field, variant=variant):
                    prefill, generation = copy.deepcopy((self.prefill, self.generation))
                    value = prefill if field == 'prefill' else generation
                    if variant == 'source': value['build_source_commit'] = 'previous-native-source'
                    if variant == 'binary': value['binary']['sha256'] = 'b'*64
                    if variant == 'incomplete': value['complete'] = False
                    if variant == 'failed': value['passed'] = False
                    if variant == 'active': value['active'] = True
                    with self.assertRaises(RuntimeError):
                        b.retained_model_evidence(self.build, prefill, generation)

    def test_prefill_requires_full_input_closed_ownership_and_fault_evidence(self):
        for variant in ('missing_run', 'missing_model_hash', 'owner', 'exit', 'fault',
                        'missing_probe', 'boot', 'partial_input', 'reuse', 'stream'):
            with self.subTest(variant=variant):
                value = copy.deepcopy(self.prefill)
                row, sample = value['runs'][0], value['runs'][0]['sample']
                if variant == 'missing_run': value['runs'].pop()
                if variant == 'missing_model_hash': value['full_model_hashes_matched'] = 14
                if variant == 'owner': row['ownership_closed'] = False
                if variant == 'exit': row['exit_code'] = 1
                if variant == 'fault': row['fresh_faults'] = ['GPU fault']
                if variant == 'missing_probe': del row['fault_probe_successful']
                if variant == 'boot': row['boot_unchanged'] = False
                if variant == 'partial_input': sample['actual_read_tokens'] -= 1
                if variant == 'reuse': sample['actual_reused_tokens'] = 1
                if variant == 'stream': sample['stream_complete'] = False
                with self.assertRaises(RuntimeError):
                    b.retained_model_evidence(self.build, value, self.generation)

    def test_generation_requires_long_thinking_and_answer_with_native_closure(self):
        for variant in ('missing_process', 'stage', 'owner', 'exit', 'fault', 'boot',
                        'journal', 'short_input', 'reuse', 'short_output', 'thought_only',
                        'no_thought', 'unverified_count', 'truncated'):
            with self.subTest(variant=variant):
                value = copy.deepcopy(self.generation)
                row, sample = value['processes'][0], value['processes'][0]['request']
                if variant == 'missing_process': value['processes'].pop()
                if variant == 'stage': value['stage'] = 'running'
                if variant == 'owner': row['ownership_closed'] = False
                if variant == 'exit': row['native_exit_code'] = 1
                if variant == 'fault': row['fault_probe']['fresh_faults'] = ['GPU fault']
                if variant == 'boot': row['fault_probe']['boot_unchanged'] = False
                if variant == 'journal': row['journal_valid'] = False
                if variant == 'short_input': sample['read_tokens'] = sample['prompt_tokens'] = 1024
                if variant == 'reuse': sample['reused_tokens'] = 1
                if variant == 'short_output': sample['generated_tokens'] = 100
                if variant == 'thought_only': sample['output_chars'] = 0
                if variant == 'no_thought': sample['reasoning_chars'] = 0
                if variant == 'unverified_count': sample['generated_tokens_source'] = 'estimate'
                if variant == 'truncated': sample['truncated'] = True
                with self.assertRaises(RuntimeError):
                    b.retained_model_evidence(self.build, self.prefill, value)


if __name__=='__main__':unittest.main()
