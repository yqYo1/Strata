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


class ArenaScreenTests(unittest.TestCase):
    def test_actual_copy_footer_conservation_and_request_binding(self):
        from arena_copy import copy_evidence
        request=dict(prompt_tokens=66,read_tokens=66,usage_generated_tokens=16)
        log=('ordinary-Arena copy enabled mode=0\n'
             'strata serve: prompt 66 tokens = 0 reused + 66 read in 26062 ms (2.5 tok/s), '
             '16 generated in 6220 ms (2.6 tok/s), drafts accepted 12 of 12, 0 checkpoints\n'
             'strata serve: SYCL_EXPERT_COPY submitted_blobs=1309 submitted_bytes=2684518400 '
             'completed_blobs=1309 completed_bytes=2684518400 host_view_groups=1309 '
             'gpu_miss_groups=1309 cpu_unique_misses=4285 cpu_entries=6456\n')
        self.assertEqual(copy_evidence(log,request,1)['logical_copy_bytes_per_emitted_token'],167782400)
        for broken in (log+log,log.replace('completed_blobs=1309','completed_blobs=1308'),
                       log.replace('cpu_entries=6456','cpu_entries=0'),log.replace('66 read','65 read')):
            with self.assertRaises(RuntimeError):copy_evidence(broken,request,1)
        with self.assertRaises(RuntimeError):copy_evidence(log,request,0)

    def test_screen_repeats_without_requiring_identical_generated_lengths(self):
        from arena_copy import screen
        def rows(speeds):
            return [dict(arm=arm,passed=True,ownership_closed=True,source_commit='a'*40,
                binary_sha256='b'*64,request=dict(sample(),controller_eligible=True,
                    request_sha256='c'*64,decode_tps=tps,decode_ms=1000*generated/tps,
                    generated_tokens=generated,usage_generated_tokens=generated))
                for arm,tps,generated in zip((0,1,1,0,0,1),speeds,(2048,3000,2500,2048,2048,4000))]
        faster=rows((10,12,12,10,10,12))
        self.assertEqual(screen(faster)['decision'],'candidate_faster_screen')
        self.assertFalse(screen(faster)['adoption_qualified'])
        self.assertEqual(screen(rows((12,10,10,12,12,10)))['decision'],'candidate_slower_screen')
        self.assertEqual(screen(rows((10,12,9,10,10,12)))['decision'],'inconclusive')
        for field,value in (('request_sha256','d'*64),('usage_generated_tokens',1),('decode_tps',float('nan'))):
            altered=copy.deepcopy(faster);altered[1]['request'][field]=value
            with self.assertRaises(RuntimeError):screen(altered)
        altered=copy.deepcopy(faster);altered[1]['binary_sha256']='d'*64
        with self.assertRaises(RuntimeError):screen(altered)


if __name__ == '__main__':
    unittest.main()
