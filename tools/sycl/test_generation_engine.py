#!/usr/bin/env python3
"""Offline negative evidence gates for the real long-generation controller."""
import copy
import unittest
from generation_engine import collect_samples, require_complete_generation, sample_eligible


def sample():
    return dict(eligible=True, speed_eligible=True, ok=True, stream_complete=True,
        finish_reason='stop', reused_tokens=0, prompt_tokens=33792, read_tokens=33792,
        usage_generated_tokens=2048, generated_tokens=2048, generated_tokens_source='usage',
        output_chars=2000, reasoning_chars=1000, completion_includes_thinking=True,
        drafts_offered=1000, drafts_accepted=600, acceptance_rate=.6,
        decode_tps_source='engine', decode_ms=20000, decode_tps=102.4, total_s=60,
        run=0, order=0)


class EvidenceTests(unittest.TestCase):
    def test_short_or_thinking_only_cannot_pass_client_labels(self):
        self.assertTrue(sample_eligible(sample()))
        for patch in ({'usage_generated_tokens': 1000, 'generated_tokens': 1000},
                      {'output_chars': 0}, {'read_tokens': 100}, {'reused_tokens': 1},
                      {'completion_includes_thinking': False}):
            with self.subTest(patch=patch):
                self.assertFalse(sample_eligible(dict(sample(), **patch)))

    def test_invalid_clocks_usage_or_acceptance_are_ineligible(self):
        for patch in ({'decode_tps_source': 'client'}, {'generated_tokens_source': 'stream_pieces'},
                      {'decode_ms': float('nan')}, {'decode_tps': float('inf')},
                      {'drafts_offered': 0}, {'drafts_accepted': 1001}, {'acceptance_rate': .7},
                      {'generated_tokens': 10}, {'stream_complete': False}):
            with self.subTest(patch=patch):
                self.assertFalse(sample_eligible(dict(sample(), **patch)))

    def test_preserves_failed_rows_and_requires_both_workloads(self):
        generation = dict(workload_version=4, thinking=True, reasoning_budget_tokens=0, max_tokens=16384,
            workloads={'regular-code': {'requests': [sample()]},
                       'varied-prose': {'requests': [dict(sample(), order=1, output_chars=0)]}})
        rows = collect_samples(generation)
        self.assertEqual(len(rows), 2)
        self.assertFalse(rows[1]['controller_eligible'])
        with self.assertRaises(RuntimeError):
            require_complete_generation(generation, rows)
        generation['workloads']['varied-prose']['requests'][0]['output_chars'] = 2000
        require_complete_generation(generation, collect_samples(generation))
        del generation['workloads']['varied-prose']
        with self.assertRaises(RuntimeError):
            require_complete_generation(generation, collect_samples(generation))

    def test_changed_thinking_workload_or_duplicate_order_cannot_pass(self):
        generation = dict(workload_version=4, thinking=True, reasoning_budget_tokens=0, max_tokens=16384,
            workloads={'regular-code': {'requests': [sample()]},
                       'varied-prose': {'requests': [dict(sample(), order=1)]}})
        for key, value in [('thinking', False), ('workload_version', 3), ('max_tokens', 8192)]:
            altered = copy.deepcopy(generation)
            altered[key] = value
            with self.assertRaises(RuntimeError):
                require_complete_generation(altered, collect_samples(altered))
        generation['workloads']['varied-prose']['requests'][0]['order'] = 0
        with self.assertRaises(RuntimeError):
            require_complete_generation(generation, collect_samples(generation))


if __name__ == '__main__':
    unittest.main()
