import json
import unittest

import dflash2_reconciliation_client as reconciliation
import dflash2_scheduling_model as scheduling


class ReconciliationAccounting(unittest.TestCase):
    def setUp(self):
        fixture, _ = reconciliation.request_body()
        self.expected = fixture['expected']
        self.segments = list(reconciliation.native_segments(
            'strata serve: prompt 35915 tokens = 0 reused + 35915 read in 23000 ms (1561.5 tok/s), '
            '1000 generated in 64000 ms (15.6 tok/s), drafts accepted 200 of 700, 0 checkpoints (cancelled)\n'
            'strata serve: prompt 36916 tokens = 36914 reused + 2 read in 3 ms (666.7 tok/s), '
            '202 generated in 14000 ms (14.4 tok/s), drafts accepted 40 of 100, 0 checkpoints\n'))
        self.history = dict(prompt_total=35915, prompt_read=35915, reused=0, output_tokens=1205, engine_generated=1202,
                            drafts_accepted=240, drafts_offered=800, finish='stop', prompt_ms=23003.2,
                            decode_ms=77999.80000000002, reasoning_recoveries=1)
        self.raw = dict(request_sha256=reconciliation.WIRE_SHA256, first=23.1, total=101.2, done_received=True,
                        finish_reason='stop', reasoning_text='Reconcile the ordered evidence.',
                        text='Incident decision and supporting detail. ' * 16 + '\n```json\n'
                             + json.dumps(self.expected) + '\n```',
                        usage=dict(completion_tokens=1205, prompt_tokens=35915, prompt_tokens_details=dict(cached_tokens=0)),
                        timings=dict(prompt_n=35915, prompt_ms=23003.2, predicted_ms=77999.8, predicted_per_second=15.4,
                                     draft_n=800, draft_n_accepted=240))

    def score(self):
        return reconciliation.account(self.raw, self.history, self.segments, context=262144, expected=self.expected)

    def test_continuation_keeps_api_tokens_out_of_native_throughput(self):
        score = self.score()
        self.assertTrue(score['answer_correct'])
        self.assertTrue(score['weak_drafts_observed'])
        self.assertEqual(score['minimum_server_added_tokens'], 3)
        self.assertAlmostEqual(score['native_tg'], 1202000 / self.history['decode_ms'])

    def test_missing_continuation_cannot_match_the_api_request(self):
        self.segments.pop()
        with self.assertRaisesRegex(ValueError, 'history and native request counters disagree'):
            self.score()

    def test_short_native_response_cannot_qualify_weak_drafts(self):
        self.segments[1]['generated'] = 1
        self.history['engine_generated'] = 1001
        score = self.score()
        self.assertFalse(score['native_generation_long_enough'])
        self.assertFalse(score['weak_drafts_observed'])

    def test_an_unrelated_request_invalidates_history_join(self):
        before = {'requests': [{'id': 'old'}]}
        after = {'requests': [{'id': 'other'}, {'id': 'ours'}, {'id': 'old'}]}
        with self.assertRaisesRegex(ValueError, 'exactly one completed request'):
            reconciliation.history_row(before, after)

    def test_explicit_scheduling_packet_uses_its_counts_and_oracle(self):
        case = scheduling.CASE
        fixture, body = reconciliation.request_body(case)
        self.assertEqual(body['chat_template_kwargs']['enable_thinking'], True)
        self.assertEqual(body['max_tokens'], reconciliation.OUTPUT_CAP)
        delta = case.input_tokens - reconciliation.INPUT
        self.raw.update(request_sha256=case.wire_sha256,
                        text='Handover.\n\x60\x60\x60json\n' + json.dumps(fixture['expected'])
                             + '\n\x60\x60\x60')
        self.raw['usage']['prompt_tokens'] = self.raw['timings']['prompt_n'] = case.input_tokens
        self.history['prompt_total'] = self.history['prompt_read'] = case.input_tokens
        self.segments[0]['prompt'] = self.segments[0]['read'] = case.input_tokens
        self.segments[1]['prompt'] += delta
        self.segments[1]['reused'] += delta
        score = reconciliation.account(self.raw, self.history, self.segments,
                                       context=262144, expected=fixture['expected'], workload=case)
        self.assertTrue(score['answer_correct'])
        self.assertTrue(score['weak_drafts_observed'])
        scored = scheduling.decision([{'evidence': {'observation': score, 'samples': [score['sample']]}}])
        self.assertAlmostEqual(scored['draft_offers_per_native_output'], 800 / 1202)
        with self.assertRaisesRegex(ValueError, 'pinned fresh input'):
            self.score()

    def test_new_controller_uses_the_scheduling_entrypoint(self):
        context = scheduling.model.experiment(scheduling.CASE, source=scheduling.THIS,
                                              protocol=scheduling.PLAN, decide=scheduling.decision)
        self.assertEqual(context.KEY, 'dflash2_scheduling_model_v1')
        self.assertEqual(context.WORKLOAD_ID, 'dflash2_scheduling')
        self.assertTrue(context.command({'root': '/unused', 'port': 18800})[1].endswith(
            '/bench/dflash2_scheduling_model.py'))
