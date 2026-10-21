#!/usr/bin/env python3
"""Retired candidate cache must not discard or widen fallback qualification."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import subprocess
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


class PackedRetentionTests(unittest.TestCase):
    """The original current generation remains recoverable after later rebuilds."""
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        def original(name):
            path = 'bench/results/2026-10-15-pr6-recovery-audit/' + name
            return json.loads(subprocess.check_output(['git', '-C', str(root), 'show',
                '385c7953de2e0a2fac49b5a016b7c78e39ec4610:' + path]))
        self.values = [original(n) for n in ('native-gdn-engine-build.json',
            'decode-query-profile-component-current.json', 'prefill-packed-bank-speed-current.json',
            'generation-reference-current.json')]

    def test_actual_current_evidence_is_retention_only(self):
        scope = b.packed_model_evidence(*self.values)
        self.assertEqual(scope['prefill_requests'], 24)
        self.assertEqual(scope['current_long_generation_requests'], 2)
        self.assertFalse(scope['sustained_decode_speed_qualified'])
        self.assertFalse(scope['model_adoption_qualified'])

    def test_reject_wrong_generation_incomplete_work_or_fault(self):
        for variant in ('component', 'prefill_source', 'fixture', 'owner', 'fault', 'short',
                        'reused', 'missing_process', 'wrong_pilot', 'no_answer', 'route', 'journal'):
            with self.subTest(variant=variant):
                build, component, prefill, generation = copy.deepcopy(self.values)
                row = prefill['processes'][0]
                pilot = generation['processes'][1]
                if variant == 'component': component['source_commit'] = 'another-native'
                if variant == 'prefill_source': row['build_source_commit'] = 'another-native'
                if variant == 'fixture': component['fixtures'] = {}
                if variant == 'owner': row['ownership_closed'] = False
                if variant == 'fault': row['fresh_faults'] = ['fault']
                if variant == 'short': row['requests'][0]['actual_read_tokens'] = 100
                if variant == 'reused': row['requests'][0]['actual_reused_tokens'] = 1
                if variant == 'missing_process': prefill['processes'].pop()
                if variant == 'wrong_pilot': pilot['binary_sha256'] = 'retired-binary'
                if variant == 'no_answer': pilot['requests'][0]['output_chars'] = 0
                if variant == 'route': pilot['routes'][0]['cache_payload_and_map_match'] = False
                if variant == 'journal': pilot['journal']['valid'] = False
                with self.assertRaises(RuntimeError):
                    b.packed_model_evidence(build, component, prefill, generation)


class CensusRetentionTests(unittest.TestCase):
    """Retain the actual census build without relabeling old clean speed evidence."""
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        def original(name):
            path = 'bench/results/2026-10-15-pr6-recovery-audit/' + name
            return subprocess.check_output(['git','-C',str(root),'show','dd9f1635c:'+path])
        self.build = json.loads(original('native-gdn-engine-build.json'))
        raw = original('decode-query-profile-component-current.json')
        self.component = json.loads(raw)
        self.census = json.loads(original('cpu-work-census-current.json'))
        self.hashes = ('f2ea644ca6b26aa8b0a7e03c4df022b44f92c8e932c5eee6e4561b52331c0a92',
            hashlib.sha256(raw).hexdigest())

    def evidence(self, build=None, component=None, census=None, hashes=None):
        return b.census_model_evidence(build or self.build, component or self.component,
            census or self.census, *(hashes or self.hashes))

    def test_actual_census_is_execution_retention_only(self):
        scope = self.evidence()
        self.assertEqual(scope['component_groups'],18)
        self.assertEqual(scope['generation_read_tokens'],[33214,33288])
        self.assertEqual(scope['original_inputs_recovered'],9838)
        for k in ('clean117K_speed_qualified','sustained_decode_speed_qualified',
                  'hardware_saturation_qualified','model_adoption_qualified'):
            self.assertIs(scope[k],False)

    def test_reject_mismatched_build_component_and_receipts(self):
        for variant in ('source','binary','fixture','component','host','group','boot','build_hash','component_hash'):
            with self.subTest(variant=variant):
                build, component = copy.deepcopy((self.build,self.component)); hashes = list(self.hashes)
                if variant=='source':build['source_commit']='another'
                if variant=='binary':build['binary']['sha256']='wrong'
                if variant=='fixture':build['fixtures']['cpu_work_census_host_test']['sha256']='wrong'
                if variant=='component':component['cpu_work_census_dispatch_contract_passed']=False
                if variant=='host':build['cpu_census_host_contract']['ownership_closed']=False
                if variant=='group':component['checks'][0]['passed']=False
                if variant=='boot':component['boot_unchanged']=False
                if variant=='build_hash':hashes[0]='wrong'
                if variant=='component_hash':hashes[1]='wrong'
                with self.assertRaises((RuntimeError,KeyError)):
                    self.evidence(build=build,component=component,hashes=hashes)

    def test_reject_unclosed_or_incomplete_model_bank_and_warmup(self):
        for variant in ('active','owner','exit','fault','probe','bank_fixture','bank_fault','bank_case',
                        'warmup','request_hash','missing_request'):
            with self.subTest(variant=variant):
                c = copy.deepcopy(self.census)
                if variant=='active':c['active']=True
                if variant=='owner':c['ownership_closed']=False
                if variant=='exit':c['native_exit_code']=1
                if variant=='fault':c['fresh_faults']=['fault']
                if variant=='probe':c['fault_probe_successful']=False
                if variant=='bank_fixture':c['bank']['checks'][0]['fixture']['sha256']='wrong'
                if variant=='bank_fault':c['bank']['fresh_faults']=['fault']
                if variant=='bank_case':c['bank']['checks'][0]['cases'].pop()
                if variant=='warmup':c['warmups']['successful_warmups']=3
                if variant=='request_hash':c['warmups']['request_sha256'][-1]='wrong'
                if variant=='missing_request':c['requests'].pop()
                with self.assertRaises((RuntimeError,KeyError)):self.evidence(census=c)

    def test_reject_short_incomplete_thought_only_and_wrong_restored_route(self):
        for variant in ('short','reused','thought_only','no_thinking','truncated','GPU_rows','cache','bank_route'):
            with self.subTest(variant=variant):
                c = copy.deepcopy(self.census); sample, route = c['requests'][0], c['routes'][0]
                if variant=='short':sample['read_tokens']=sample['prompt_tokens']=1024
                if variant=='reused':sample['reused_tokens']=1
                if variant=='thought_only':sample['output_chars']=0
                if variant=='no_thinking':sample['reasoning_chars']=0
                if variant=='truncated':sample['truncated']=True
                if variant=='GPU_rows':route['actual_GPU_BF16_rows']-=1
                if variant=='cache':route['cache_payload_and_map_match']=False
                if variant=='bank_route':route['private_bank']['actual_private_layers']=47
                with self.assertRaises((RuntimeError,KeyError)):self.evidence(census=c)


class ArchivedCensusRetentionTests(unittest.TestCase):
    """The current6c build cannot borrow the older census at a reused path."""
    revision = 'bedfdf312fe3cf28842c78263f1e5b69cdf5ec5c'
    prefix = 'bench/results/2026-10-15-pr6-recovery-audit/'

    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        self.raw = {name: subprocess.check_output(['git', '-C', str(root), 'show',
            self.revision + ':' + self.prefix + name]) for name in
            ('native-gdn-engine-build.json', 'decode-query-profile-component-current.json',
             'iq2-pair-model-current.json', 'cpu-work-census-current.json')}
        self.build = json.loads(self.raw['native-gdn-engine-build.json'])
        self.pilot = json.loads(self.raw['iq2-pair-model-current.json'])
        self.selected = dict(commit=self.revision, path=self.prefix + 'iq2-pair-model-current.json',
            sha256=hashlib.sha256(self.raw['iq2-pair-model-current.json']).hexdigest())
        self.reference = dict(model_evidence_kind='census', model_evidence_reference=self.selected,
            model_evidence_commits={name:self.revision for name in
                ('decode-query-profile-component-current.json', 'iq2-pair-model-current.json')},
            source_build_record=dict(sha256=self.pilot['qualification']['build_receipt_sha256']))

    def test_exact_archive_selects_current_build_without_promoting_speed(self):
        import generation_compare as comparison
        self.assertEqual(b.archived_model_observation(self.selected), self.pilot)
        scope = comparison.retained_model_scope(self.build, self.reference)
        self.assertEqual(self.build['source_commit'], '6c7f7f36d4dfa7d773af1d4da7a4d92676270632')
        self.assertEqual(scope['original_inputs_recovered'], 9845)
        self.assertEqual(scope['current_long_generation_requests'], 2)
        self.assertFalse(scope['clean117K_speed_qualified'])
        self.assertFalse(scope['model_adoption_qualified'])

    def test_wrong_or_missing_archive_never_falls_back_to_current_filename(self):
        for variant in ('hash', 'mutable_ref', 'outside_results', 'extra_field', 'missing'):
            with self.subTest(variant=variant):
                selected = copy.deepcopy(self.selected)
                if variant == 'hash': selected['sha256'] = '0' * 64
                if variant == 'mutable_ref': selected['commit'] = 'HEAD'
                if variant == 'outside_results': selected['path'] = 'other/receipt.json'
                if variant == 'extra_field': selected['fallback'] = 'cpu-work-census-current.json'
                if variant == 'missing': selected.pop('sha256')
                with self.assertRaises(RuntimeError): b.archived_model_observation(selected)

    def test_reader_rejects_other_build_even_with_a_valid_original_hash(self):
        import generation_compare as comparison
        selected = dict(commit=self.revision, path=self.prefix + 'cpu-work-census-current.json',
            sha256=hashlib.sha256(self.raw['cpu-work-census-current.json']).hexdigest())
        reference = copy.deepcopy(self.reference)
        reference['model_evidence_reference'] = selected
        reference['model_evidence_commits'] = {name:self.revision for name in
            ('decode-query-profile-component-current.json', 'cpu-work-census-current.json')}
        with self.assertRaises(RuntimeError): comparison.retained_model_scope(self.build, reference)

    def test_reader_rejects_unknown_kind_or_changed_source_receipt(self):
        import generation_compare as comparison
        for variant in ('kind', 'build_hash'):
            with self.subTest(variant=variant):
                reference = copy.deepcopy(self.reference)
                if variant == 'kind': reference['model_evidence_kind'] = 'unknown'
                if variant == 'build_hash': reference['source_build_record']['sha256'] = '0' * 64
                with self.assertRaises(RuntimeError): comparison.retained_model_scope(self.build, reference)


if __name__=='__main__':unittest.main()
