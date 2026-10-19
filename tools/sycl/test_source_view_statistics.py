#!/usr/bin/env python3
"""Offline tests against pseudoreplication and altered source-view decisions."""
import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
from bench_engine import (FUSION_FIELDS, down_pair_statistics, expert_bank_reuse_scope,
                          compact_build_record, committed_engine_build, expert_bank_qualified_record,
                          recorded_files_unchanged,
                          parse_counter_lines, qualified_build_source_matches,
                          source_view_pair_statistics, validate_window_work)
from build_engine import binary_origin_source


def arms(base, candidate):
    result = []
    for cohort in range(3):
        for name, values in (('A', base[cohort]), ('B', candidate[cohort])):
            result.append(dict(label=name + str(cohort), first_use_tps=1000 if name == 'A' else 500,
                subsequent_tps=values, subsequent_median=statistics.median(values),
                requested_expert_H2D_bytes=[10, 10, 10, 10], output_hashes=['output'], reasoning_hashes=['thought']))
    return result


class ProcessStatisticsTests(unittest.TestCase):
    def test_large_request_ratio_does_not_inflate_independent_process_gain(self):
        # One request doubles while the process median improves only0.4%.
        # Treating nine request ratios as independent would incorrectly pass.
        data = arms([[1, 100, 100]] * 3, [[2, 100.4, 100.4]] * 3)
        result = source_view_pair_statistics(data)
        self.assertAlmostEqual(result['paired_median_speed_geomean'], 1.004)
        self.assertFalse(result['performance_selection_gate_passed'])

    def test_two_large_gains_do_not_hide_a_regressing_process(self):
        result = source_view_pair_statistics(arms([[1000]*3]*3,
            [[1200]*3, [1200]*3, [999]*3]))
        self.assertGreater(result['paired_median_speed_geomean'], 1.005)
        self.assertFalse(result['performance_selection_gate_passed'])

    def test_cold_regression_is_retained_without_changing_frozen_warm_gate(self):
        result = source_view_pair_statistics(arms([[950]*3]*3, [[996]*3]*3))
        self.assertTrue(result['performance_selection_gate_passed'])
        self.assertEqual([p['first_use_ratio'] for p in result['pairs']], [0.5]*3)
        self.assertFalse(result['prefill1000_target_passed'])

    def test_target_requires_each_candidate_process_median(self):
        result = source_view_pair_statistics(arms([[950]*3]*3,
            [[1005]*3, [1005]*3, [999.9]*3]))
        self.assertGreater(statistics.mean(result['candidate_process_medians']), 1000)
        self.assertTrue(result['performance_selection_gate_passed'])
        self.assertFalse(result['prefill1000_target_passed'])

    def test_output_and_requested_payload_are_descriptive(self):
        data = arms([[950]*3]*3, [[1001]*3]*3)
        data[1]['requested_expert_H2D_bytes'][0] += 1
        data[1]['output_hashes'] = ['different']
        result = source_view_pair_statistics(data)
        self.assertTrue(result['performance_selection_gate_passed'])
        self.assertFalse(result['requested_expert_payload_matches_in_all_pairs'])
        self.assertFalse(result['pairs'][0]['output_hashes_match'])

    def test_duplicate_arm_is_not_six_processes(self):
        data = arms([[950]*3]*3, [[996]*3]*3)
        data[-1]['label'] = 'B0'
        with self.assertRaisesRegex(RuntimeError, 'six unique'):
            source_view_pair_statistics(data)


class DownStatisticsTests(unittest.TestCase):
    def test_down_and_source_view_keep_different_prospective_gates(self):
        data=arms([[1000]*3]*3,[[1003]*3]*3)
        self.assertTrue(down_pair_statistics(data)['performance_selection_gate_passed'])
        self.assertFalse(source_view_pair_statistics(data)['performance_selection_gate_passed'])

    def test_exact_boundary_is_not_a_strict_gain(self):
        self.assertFalse(down_pair_statistics(arms([[1000]*3]*3,[[1002]*3]*3))[
            'performance_selection_gate_passed'])

    def test_one_regressing_process_is_not_hidden_by_large_gains(self):
        result=down_pair_statistics(arms([[1000]*3]*3,[[1200]*3,[1200]*3,[999]*3]))
        self.assertGreater(result['paired_median_speed_geomean'],1.002)
        self.assertFalse(result['performance_selection_gate_passed'])

    def test_target_is_separate_and_each_candidate_process_matters(self):
        result=down_pair_statistics(arms([[950]*3]*3,[[999.9]*3,[1001]*3,[1001]*3]))
        self.assertTrue(result['performance_selection_gate_passed'])
        self.assertFalse(result['prefill1000_target_passed'])

    def test_median_cannot_override_original_samples(self):
        data=arms([[1000]*3]*3,[[1001]*3]*3);data[1]['subsequent_median']=1100
        with self.assertRaises(RuntimeError):down_pair_statistics(data)

    def test_nonfinite_or_incomplete_samples_are_rejected(self):
        for values in ([1003,float('nan'),1003],[1003,float('inf'),1003],[1003,1003]):
            data=arms([[1000]*3]*3,[values,[1003]*3,[1003]*3])
            with self.assertRaises(RuntimeError):down_pair_statistics(data)


class WorkCounterTests(unittest.TestCase):
    def setUp(self):
        self.prefix = 'strata MoE Down fusion:'
        self.valid = self.prefix+' fused_jobs=0 fused_rows=0 separate_jobs=10000 min_rows=2334'
        self.work = dict(windows='96', dequants='1000', jobs='10000',
            source_H2D_bytes=str(117488*48*10*4), GPU_resident_windows_only='1',
            operator_max='8192', maximum_rows='81920')
        self.gu = dict(fused_jobs='1000', fused_rows='5000000', separate_jobs='9000', min_rows='2334')
        self.down = dict(fused_jobs='0', fused_rows='0', separate_jobs='10000', min_rows='2334')

    def check_work(self):
        validate_window_work(self.work, self.gu, self.down, 117489, (81920,35568))

    def test_valid_disabled_and_selective_paths(self):
        counts = parse_counter_lines(self.valid, self.prefix, FUSION_FIELDS)
        self.assertEqual(counts, [self.down])
        self.check_work()
        self.down.update(fused_jobs='500', fused_rows='2000000', separate_jobs='9500')
        self.check_work()

    def test_duplicate_unknown_missing_and_malformed_fields_are_rejected(self):
        for text in (self.valid+' fused_jobs=1', self.valid+' surprise=1',
                     self.valid.replace(' fused_rows=0',''), self.valid.replace('jobs=0','jobs=-1'),
                     self.valid.replace('rows=0','rows=0.0'), self.valid.replace(': ',':'),
                     self.valid+' bare-token'):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                parse_counter_lines(text, self.prefix, FUSION_FIELDS)

    def test_gu_and_down_job_conservation_are_independent(self):
        for route in (self.gu,self.down):
            original=route['separate_jobs'];route['separate_jobs']=str(int(original)-1)
            with self.assertRaises(RuntimeError):self.check_work()
            route['separate_jobs']=original

    def test_fused_row_bounds_do_not_accept_impossible_totals(self):
        for rows in ('2333999','8192001',str(117488*48*10+1)):
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                self.gu['fused_rows']=rows;self.check_work()

    def test_window_transfer_and_geometry_mismatches_are_rejected(self):
        for key in self.work:
            original=self.work[key]
            self.work[key]='0'
            with self.subTest(key=key), self.assertRaises(RuntimeError):self.check_work()
            self.work[key]=original

    def test_impossible_total_job_capacity_is_rejected(self):
        self.work['jobs']='1000';self.gu.update(fused_jobs='0',fused_rows='0',separate_jobs='1000')
        self.down['separate_jobs']='1000'
        with self.assertRaises(RuntimeError):self.check_work()


class BinaryIdentityTests(unittest.TestCase):
    def test_matching_engine_keeps_origin_after_fixture_or_controller_changes(self):
        previous=dict(passed=True,source_commit='admission',binary_origin_source_commit='original-native',
            binary=dict(sha256='same-engine'),previous_output_comparison=dict(controller_only_admission=False))
        self.assertEqual(binary_origin_source(previous,'same-engine','new-fixture'),'original-native')

    def test_changed_or_unproved_engine_gets_current_source(self):
        previous=dict(passed=True,source_commit='old-native',binary=dict(sha256='old'))
        self.assertEqual(binary_origin_source(previous,'new','current'),'current')
        previous['passed']=False
        self.assertEqual(binary_origin_source(previous,'old','current'),'current')

    def test_original_legacy_identity_and_first_build(self):
        previous=dict(passed=True,source_commit='old-native',binary=dict(sha256='same'))
        self.assertEqual(binary_origin_source(previous,'same','current'),'old-native')
        self.assertEqual(binary_origin_source(None,'new','current'),'current')


class BuildQualificationTests(unittest.TestCase):
    def setUp(self):
        self.build = dict(source_commit='admission', binary_origin_source_commit='native',
            complete=True, passed=True, ownership_closed=True, active=False, fixtures={'fixture': {}},
            previous_output_comparison=dict(controller_only_admission=True,
                native_and_recipe_inputs_match=True, engine_bytes_match=True,
                fixture_bytes_match={'fixture': True}))

    def test_exact_and_proved_native_origin(self):
        self.assertTrue(qualified_build_source_matches('admission', self.build))
        self.assertTrue(qualified_build_source_matches('native', self.build))
        self.assertFalse(qualified_build_source_matches('nearby-revision', self.build))

    def test_native_or_recipe_change_cannot_reuse_old_gate(self):
        self.build['previous_output_comparison']['native_and_recipe_inputs_match'] = False
        self.assertFalse(qualified_build_source_matches('native', self.build))

    def test_binary_or_fixture_change_cannot_reuse_old_gate(self):
        self.build['previous_output_comparison']['engine_bytes_match'] = False
        self.assertFalse(qualified_build_source_matches('native', self.build))
        self.build['previous_output_comparison']['engine_bytes_match'] = True
        self.build['previous_output_comparison']['fixture_bytes_match']['fixture'] = False
        self.assertFalse(qualified_build_source_matches('native', self.build))

    def test_incomplete_fixture_proof_cannot_reuse_old_gate(self):
        self.build['previous_output_comparison']['fixture_bytes_match'] = {}
        self.assertFalse(qualified_build_source_matches('native', self.build))

    def test_active_or_failed_build_cannot_qualify_either_revision(self):
        self.build['active'] = True
        self.assertFalse(qualified_build_source_matches('admission', self.build))
        self.assertFalse(qualified_build_source_matches('native', self.build))
        self.build['active'] = False
        self.build['passed'] = False
        self.assertFalse(qualified_build_source_matches('admission', self.build))


class ExpertBankReuseTests(unittest.TestCase):
    def setUp(self):
        fixtures = {'sycl_expert_bank_test': {'sha256': 'bank'},
                    'file_expert_source_test': {'sha256': 'provider'},
                    'sycl_moe_down_scatter_bench': {'sha256': 'old-down'}}
        self.old = dict(source_commit='qualified', complete=True, passed=True,
            ownership_closed=True, active=False, binary={'sha256': 'engine'}, fixtures=fixtures,
            project=dict(root='project', revision='qualified', source_sha256={
                'src/prefill/moe_window.cpp': 'native', 'CMakeLists.txt': 'cmake',
                'bench/sycl/moe_down_scatter.cpp': 'old-down', 'tools/sycl/build_engine.py': 'old-controller'}),
            dependencies=[dict(root='dependency', revision='dep', source_sha256={'src.cpp': 'dep'})],
            compiler={'sha256': 'compiler'}, configure_argv=['cmake'], environment={'key': 'value'},
            recipe_reference={'sha256': 'recipe'}, compile_database_sha256='database',
            configured_compile_entries=10, device={'name': 'B570'},
            targets=['strata', *fixtures])
        self.new = copy.deepcopy(self.old)
        self.new['source_commit'] = self.new['project']['revision'] = 'current'
        self.new['project']['source_sha256']['bench/sycl/moe_down_scatter.cpp'] = 'new-down'
        self.new['project']['source_sha256']['tools/sycl/build_engine.py'] = 'new-controller'
        self.new['fixtures']['sycl_moe_down_scatter_bench']['sha256'] = 'new-down'
        self.new['targets'].reverse()

    def proof(self):
        return expert_bank_reuse_scope('qualified', self.new, self.old)

    def qualified(self):
        return dict(build_source_commit='qualified', provider_checks_passed=True,
            fixture=copy.deepcopy(self.old['fixtures']['sycl_expert_bank_test']),
            provider_fixture=copy.deepcopy(self.old['fixtures']['file_expert_source_test']))

    def test_exact_original_receipt_accepts_only_its_qualified_outputs(self):
        gpu = self.qualified(); historical = compact_build_record(self.old)
        expert_bank_qualified_record(gpu, historical, self.old)
        for key in ('provider_checks_passed', 'provider_fixture', 'build_source_commit', 'fixture'):
            saved = gpu.pop(key)
            with self.subTest(key=key), self.assertRaises((RuntimeError, KeyError)):
                expert_bank_qualified_record(gpu, historical, self.old)
            gpu[key] = saved

    def test_same_revision_rebuilt_receipt_cannot_substitute_for_original(self):
        original = copy.deepcopy(self.old); historical = compact_build_record(original)
        for key in ('binary', 'compiler', 'configure_argv', 'recipe_reference', 'dependencies'):
            replaced = copy.deepcopy(original); replaced[key] = {'forged': 'same source revision'}
            with self.subTest(key=key), self.assertRaises((RuntimeError, KeyError, TypeError)):
                expert_bank_qualified_record(self.qualified(), historical, replaced)

    def test_provider_failure_or_changed_fixture_cannot_use_same_source_shortcut(self):
        for key, value in (('provider_checks_passed', False),
                           ('provider_fixture', {'sha256': 'changed'})):
            gpu = self.qualified(); gpu[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                expert_bank_qualified_record(gpu, compact_build_record(self.old), self.old)

    def test_prepare_binds_committed_compact_and_full_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); state = root/'state'; (state/'main-native-gdn-engine').mkdir(parents=True)
            build = root/'compact.json'; full = state/'main-native-gdn-engine/record.json'
            full.write_text(json.dumps(self.old)); raw = json.dumps(compact_build_record(self.old)).encode()
            build.write_bytes(raw)
            with patch('bench_engine.P', root), patch('bench_engine.B', state), \
                 patch('bench_engine.BUILD', build), patch('bench_engine.subprocess.check_output', return_value=raw):
                admitted, receipts = committed_engine_build()
                self.assertEqual(admitted, json.loads(raw)); self.assertEqual(set(receipts), {str(build), str(full)})
                build.write_bytes(raw+b' ')
                with self.assertRaisesRegex(RuntimeError, 'commit the exact'): committed_engine_build()
                build.write_bytes(raw)
                changed = copy.deepcopy(self.old); changed['binary']['sha256'] = 'other-engine'
                full.write_text(json.dumps(changed))
                with self.assertRaisesRegex(RuntimeError, 'full build differs'): committed_engine_build()

    def test_launch_rejects_replaced_fallback_receipt_and_either_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/name for name in ('fallback-receipt', 'bank', 'provider')]
            for path in paths: path.write_bytes(path.name.encode())
            inputs = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            recorded_files_unchanged(inputs)
            for path in paths:
                raw = path.read_bytes(); path.write_bytes(b'replaced')
                with self.subTest(path=path.name), self.assertRaises(RuntimeError): recorded_files_unchanged(inputs)
                path.write_bytes(raw)

    def test_only_unchanged_bank_and_provider_are_qualified(self):
        proof = self.proof()
        self.assertEqual(set(proof['required_fixture_sha256']),
                         {'sycl_expert_bank_test', 'file_expert_source_test'})
        self.assertIn('No reuse of the Down fixture', proof['limits'])

    def test_same_engine_does_not_excuse_native_or_build_system_changes(self):
        for path in ('src/prefill/moe_window.cpp', 'CMakeLists.txt'):
            with self.subTest(path=path):
                saved = self.new['project']['source_sha256'][path]
                self.new['project']['source_sha256'][path] = 'changed'
                with self.assertRaises(RuntimeError): self.proof()
                self.new['project']['source_sha256'][path] = saved

    def test_each_required_fixture_and_engine_must_match(self):
        for output in ('sycl_expert_bank_test', 'file_expert_source_test', 'binary'):
            target = self.new['binary'] if output == 'binary' else self.new['fixtures'][output]
            saved = target['sha256']; target['sha256'] = 'changed'
            with self.subTest(output=output), self.assertRaises(RuntimeError): self.proof()
            target['sha256'] = saved

    def test_same_output_does_not_excuse_changed_recipe_compiler_or_dependencies(self):
        for key in ('dependencies', 'compiler', 'configure_argv', 'environment', 'recipe_reference',
                    'compile_database_sha256', 'configured_compile_entries', 'device'):
            saved = self.new[key]; self.new[key] = 'changed'
            with self.subTest(key=key), self.assertRaises(RuntimeError): self.proof()
            self.new[key] = saved

    def test_missing_source_or_extra_target_is_not_equivalent(self):
        self.new['project']['source_sha256'].pop('CMakeLists.txt')
        with self.assertRaises(RuntimeError): self.proof()
        self.setUp(); self.new['targets'].append('new-target')
        with self.assertRaises(RuntimeError): self.proof()
        self.setUp(); self.new['targets'].append('strata')
        with self.assertRaises(RuntimeError): self.proof()

    def test_active_failed_or_wrong_revision_cannot_reuse_old_validation(self):
        for record in (self.old, self.new):
            for key, bad in (('active', True), ('complete', False), ('passed', False), ('ownership_closed', False)):
                saved = record[key]; record[key] = bad
                with self.subTest(key=key), self.assertRaises(RuntimeError): self.proof()
                record[key] = saved
        self.new['project']['revision'] = 'nearby'
        with self.assertRaises(RuntimeError): self.proof()
        self.setUp()
        with self.assertRaises(RuntimeError): expert_bank_reuse_scope('nearby', self.new, self.old)


if __name__ == '__main__':
    unittest.main()
