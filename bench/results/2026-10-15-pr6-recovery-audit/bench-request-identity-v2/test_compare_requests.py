#!/usr/bin/env python3
"""Actual journal reader contract checks; host only and not yet executed."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_requests as utility

# Reuse all six meaningful corruption/order/parameter/budget tests against v2,
# rather than copying or weakening the earlier strict-reader fixtures.
spec = importlib.util.spec_from_file_location('original_request_identity_tests',
    HERE.parent / 'bench-request-identity-v1/test_compare_requests.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
original.utility = utility


class RequestIdentityTests(original.RequestIdentityTests):
    def task_fixture(self):
        return dict(comparison_identity=dict(thinking=True, reasoning_budget_tokens=0, temperature=0,
            model='same-model', quant='IQ3_S', context=262144,
            agent_files=[['project-file.txt', 'a' * 64]], agent_max_tokens=8192))

    def test_static_agent_task_identity_allows_unscored_answer_changes(self):
        a, b = self.task_fixture(), self.task_fixture()
        a['results'] = {'agent': {'answer_sha256': 'a' * 64}}
        b['results'] = {'agent': {'answer_sha256': 'b' * 64}}
        report = utility.compare_agent_tasks(a, b)
        self.assertTrue(report['passed'])
        self.assertFalse(report['quality_qualified'])
        self.assertFalse(report['exact_input_paired_speed'])

    def test_static_agent_task_identity_rejects_missing_or_changed_work(self):
        a = self.task_fixture()
        for name, value in [('agent_files', [['other-file.txt', 'b' * 64]]), ('thinking', False),
                            ('context', 32768), ('agent_max_tokens', 100), ('agent_files', [])]:
            b = self.task_fixture()
            b['comparison_identity'][name] = value
            with self.subTest(name=name, value=value):
                self.assertFalse(utility.compare_agent_tasks(a, b)['passed'])
        b = self.task_fixture()
        del b['comparison_identity']['agent_files']
        self.assertFalse(utility.compare_agent_tasks(a, b)['passed'])

    def test_generated_agent_history_may_differ_without_quality_claim(self):
        rows = copy.deepcopy(original.requests())
        changed = [row for row in rows if row['phase'] == 'agent' and row['turn'] > 0]
        for row in changed:
            row['request_sha256'] = 'a' * 64
        report = utility.compare(self.fixture('a'), self.fixture('b', rows))
        self.assertTrue(report['input_pair_qualified'])
        self.assertTrue(report['fixed_inputs_match'])
        self.assertTrue(report['agent_protocol_match'])
        self.assertFalse(report['inputs_match'])
        self.assertFalse(report['agent_exact_inputs_match'])
        self.assertEqual(report['status'], 'FIXED_INPUTS_MATCH_AGENT_FEEDBACK')
        self.assertEqual(report['agent_feedback_divergent_bodies'], len(changed))
        self.assertFalse(report['quality_qualified'])
        self.assertFalse(report['performance_qualified'])
        output = self.root / 'feedback-comparison.json'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(utility.main([str(self.root / 'a'), str(self.root / 'b'), str(output)]), 0)
        actual = json.loads(output.read_text())
        self.assertFalse(actual['inputs_match'])
        self.assertEqual(actual['status'], 'FIXED_INPUTS_MATCH_AGENT_FEEDBACK')

    def test_agent_turn_zero_body_still_has_to_match(self):
        rows = copy.deepcopy(original.requests())
        next(row for row in rows if row['phase'] == 'agent' and row['turn'] == 0)['request_sha256'] = 'b' * 64
        report = utility.compare(self.fixture('a'), self.fixture('b', rows))
        self.assertFalse(report['input_pair_qualified'])
        self.assertEqual(report['status'], 'FAIL')

    def test_agent_feedback_does_not_exempt_parameters_or_order(self):
        left = self.fixture('a')
        for key, value in [('max_tokens', 25), ('thinking', False), ('turn', 99), ('run', 8)]:
            rows = copy.deepcopy(original.requests())
            row = next(row for row in rows if row['phase'] == 'agent' and row['turn'] == 2)
            row[key] = value
            row['request_sha256'] = 'c' * 64
            with self.subTest(key=key):
                report = utility.compare(left, self.fixture(key, rows))
                self.assertFalse(report['input_pair_qualified'])
                self.assertTrue(report['errors'])


if __name__ == '__main__':
    unittest.main()
