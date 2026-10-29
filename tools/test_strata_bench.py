"""HTTP/usage and eligibility regression checks; no model or GPU is needed."""
import io
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import long_generation_bench as lg
import strata_bench as bench


def response(offered=1600, accepted=1200, generated=1152):
    return {"first": .1, "total": 2, "text": "print(1)\n"*100, "reasoning_text": "reasoning",
            "pieces": 2, "finish_reason": "length", "done_received": True,
            "request_sha256": "fixture", "err": None,
            "usage": {"prompt_tokens": 33000, "completion_tokens": generated,
                      "prompt_tokens_details": {"cached_tokens": 0}},
            "timings": {"prompt_n": 33000, "prompt_ms": 1000, "prompt_per_second": 33000,
                        "predicted_ms": 1000, "predicted_per_second": generated,
                        "draft_n": offered, "draft_n_accepted": accepted}}


class BenchmarkTest(unittest.TestCase):
    def exact_suite(self, target=32768, runs=2):
        return bench.Suite('http://localhost','',65536,runs,True,log=lambda _:None,
            prefill_only=True,prefill_exact_tokens=target,workload_id='exact-fixture')

    @staticmethod
    def counted_chars(messages, timeout):
        return len(messages[0]['content'])+29

    def count_endpoint(self, base, key, messages, timeout):
        return self.counted_chars(messages,timeout)

    def exact_reply(self, suite, mutate=None):
        def reply(_base,_key,messages,cap,think):
            raw=response(generated=24)
            if messages[0]['content'].startswith('[bench '):
                self.assertEqual(len(suite.frozen_prefill_hashes),suite.runs)
                self.assertTrue(think)
                actual=self.counted_chars(messages,30)
                raw['usage']['prompt_tokens']=raw['timings']['prompt_n']=actual
                raw['timings']['prompt_per_second']=actual
                raw['request_sha256']=hashlib.sha256(json.dumps(bench.chat_body(messages,cap,think)).encode()).hexdigest()
                if mutate:
                    mutate(raw)
            return raw
        return reply

    def test_exact_prefill_freezes_all_bodies_before_any_timed_request(self):
        suite=self.exact_suite()
        with patch.object(bench,'count_chat_tokens',side_effect=self.count_endpoint) as counter, \
             patch.object(bench,'chat',side_effect=self.exact_reply(suite)) as chat:
            suite.run_prefill_only()
        result=suite.out['prefill_only']
        self.assertTrue(result['complete'])
        self.assertEqual(chat.call_count,3)  # One reported warmup, two timed requests.
        self.assertEqual(len(result['count_preflight']),2)
        self.assertLessEqual(counter.call_count,64)
        for row in result['requests']:
            self.assertEqual(row['actual_read_tokens'],32768)
            self.assertEqual(row['timing_frontend_state'],'post-count-preflight')
        for proof in result['count_preflight']:
            self.assertEqual(proof['counted_input_tokens'],32768)

    def test_exact_prefill_rejects_a_larger_count_that_meets_the_minimum(self):
        suite=self.exact_suite()
        def changed(raw):
            raw['usage']['prompt_tokens']+=1
            raw['timings']['prompt_n']+=1
        with patch.object(bench,'count_chat_tokens',side_effect=self.count_endpoint), \
             patch.object(bench,'chat',side_effect=self.exact_reply(suite,changed)) as chat:
            with self.assertRaisesRegex(bench.EngineRequestError,'actual prompt/read count differs'):
                suite.run_prefill_only()
        self.assertEqual(chat.call_count,2)
        self.assertFalse(suite.out['prefill_only']['complete'])

    def test_exact_prefill_rejects_changed_body_after_counting(self):
        suite=self.exact_suite()
        with patch.object(bench,'count_chat_tokens',side_effect=self.count_endpoint), \
             patch.object(bench,'chat',side_effect=self.exact_reply(suite,lambda raw:raw.update(request_sha256='changed'))):
            with self.assertRaisesRegex(bench.EngineRequestError,'timed body differs'):
                suite.run_prefill_only()

    def test_unreachable_count_is_bounded_and_does_not_calibrate_with_inference(self):
        suite=self.exact_suite(target=32769)
        with patch.object(bench,'count_chat_tokens',side_effect=lambda base,key,messages,timeout:2*len(messages[0]['content'])+100) as count, \
             patch.object(bench,'chat',return_value=response(generated=24)) as chat:
            with self.assertRaisesRegex(bench.EngineRequestError,'cannot construct the exact count'):
                suite.run_prefill_only()
        self.assertEqual(chat.call_count,1)
        self.assertLessEqual(count.call_count,32)

    def test_exact_prefill_rejects_conflicting_cli_scopes_before_launch(self):
        for tail in [['--prefill-tokens','65536'],['--prefill-scaling']]:
            with patch('sys.stderr',io.StringIO()),self.assertRaises(SystemExit) as stopped:
                bench.main(['--prefill-exact-tokens','32768',*tail])
            self.assertEqual(stopped.exception.code,2)

    def test_ttft_stops_at_reasoning_and_ignores_empty_stream_events(self):
        events = [dict(choices=[dict(delta=dict(role='assistant'))]),
                  dict(choices=[dict(delta={})]),
                  dict(choices=[dict(delta=dict(reasoning_content='first thought'))]),
                  dict(choices=[dict(delta=dict(content='later answer'))]),
                  dict(choices=[dict(delta={}, finish_reason='stop')],
                       usage=response()['usage'], timings=response()['timings'])]
        wire = b': keepalive\n\n' + b''.join(
            b'data: '+json.dumps(e).encode()+b'\n\n' for e in events) + b'data: [DONE]\n\n'
        with patch.object(bench.urllib.request, 'urlopen', return_value=io.BytesIO(wire)), \
             patch.object(bench.time, 'perf_counter', side_effect=[100, 100.25, 102]), \
             patch.object(bench.time, 'monotonic_ns', side_effect=[10, 20, 30]):
            raw = bench.chat('http://localhost', '', [], 24, True)
        self.assertEqual(raw['first'], .25)
        self.assertEqual(raw['total'], 2)
        self.assertEqual(raw['reasoning_text'], 'first thought')

    def test_scaling_preserves_actual_counts_first_use_and_warmed_order(self):
        suite = bench.Suite('http://localhost', '', 131072, 3, False, log=lambda _: None,
                            prefill_only=True, characterize_prefill=True)
        seen = []
        def prompt(tokens, run, fixed=False):
            self.assertTrue(fixed)
            return [dict(role='user', content=f'{tokens}:{run}')]
        def reply(_url, _key, messages, cap, think):
            self.assertEqual(cap, 24)
            self.assertTrue(think)
            raw = response(generated=24)
            text = messages[-1]['content']
            if ':' not in text:
                return raw
            tokens, run = map(int, text.split(':'))
            actual = tokens+17
            seen.append((tokens, run))
            raw['usage']['prompt_tokens'] = raw['timings']['prompt_n'] = actual
            raw['timings']['prompt_ms'] = 2000+2*actual
            raw['timings']['prompt_per_second'] = actual/(raw['timings']['prompt_ms']/1000)
            return raw
        with patch.object(suite, 'prompt_for', side_effect=prompt), \
             patch.object(bench, 'chat', side_effect=reply):
            suite.run()
        result = suite.out['prefill_scaling']
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['requests']), 18)
        self.assertEqual(result['first_long_request']['run'], -1)
        self.assertEqual(seen[0], (98304*5//4, -1))
        expected = [(n*5//4, r) for r in range(3) for n in bench.prefill_scaling.order(r)]
        self.assertEqual(seen[1:], expected)
        self.assertEqual(result['requests'][0]['actual_read_tokens'], 1297)
        self.assertAlmostEqual(result['linear_fit']['intercept_ms'], 2000)
        self.assertAlmostEqual(result['linear_fit']['ms_per_token'], 2)
        self.assertLess(result['approximation_models']['candidates']['affine']['max_held_out_relative_error'], 1e-9)
        self.assertIsNone(result['approximation_models']['selected_model'])
        self.assertNotIn('requests', suite.out['prompts']['1024-minimum'])
        rendered = bench.render_readme(dict(machine={}, results=suite.out,
            params=dict(mode='prefill-scaling'), complete=True))
        self.assertIn('TTFT s', rendered)
        self.assertIn('a+b*x+c*x^2', rendered)
        self.assertIn('Mean held-out error', rendered)

    def test_scaling_refuses_short_context_and_stops_on_reused_prefix(self):
        short = bench.Suite('', '', 123159, 1, True, prefill_only=True, characterize_prefill=True)
        with patch.object(bench, 'chat') as chat:
            with self.assertRaises(bench.EngineRequestError):
                short.run()
            chat.assert_not_called()
        suite = bench.Suite('', '', 131072, 1, True, prefill_only=True,
                            characterize_prefill=True, log=lambda _: None)
        def reply(*args):
            raw = response(generated=24)
            if len(calls) == 1:
                raw['usage']['prompt_tokens'] = raw['timings']['prompt_n'] = 120000
            if len(calls) == 2:
                raw['usage']['prompt_tokens_details']['cached_tokens'] = 1
            calls.append(1)
            return raw
        calls = []
        with patch.object(bench, 'chat', side_effect=reply):
            with self.assertRaises(bench.EngineRequestError):
                suite.run()
        self.assertEqual(len(calls), 3)
        result = suite.out['prefill_scaling']
        self.assertFalse(result['complete'])
        self.assertIsNone(result['linear_fit'])
        self.assertFalse(result['requests'][0]['eligible'])

    def test_bad_prefill_clock_stops_scaling_before_the_next_request(self):
        for invalid in (float('inf'),float('-inf'),float('nan'),None,True,0,-1,'1'):
            with self.subTest(prompt_ms=invalid):
                suite=bench.Suite('', '',131072,3,True,prefill_only=True,
                    characterize_prefill=True,log=lambda _:None)
                warmup=response(generated=24)
                bad=response(generated=24)
                bad['usage']['prompt_tokens']=bad['timings']['prompt_n']=120000
                bad['timings']['prompt_ms']=invalid
                with patch.object(bench,'chat',side_effect=[warmup,bad]) as chat:
                    with self.assertRaisesRegex(bench.EngineRequestError,'prompt_ms'):
                        suite.run()
                self.assertEqual(chat.call_count,2)
                cohort=suite.out['prefill_scaling']
                self.assertFalse(cohort['first_long_request']['eligible'])
                self.assertEqual(cohort['requests'],[])
                self.assertFalse(cohort['complete'])

    def test_approximation_checks_held_out_lengths_and_exposes_curvature(self):
        samples = [dict(actual_read_tokens=n, prompt_ms=1000+.2*n+1e-6*n*n)
                   for n in (1024, 4096, 16384, 32768, 65536, 98304)]
        models = bench.prefill_scaling.approximation_models(samples)['candidates']
        self.assertGreater(models['affine']['max_held_out_relative_error'], .1)
        self.assertLess(models['quadratic']['max_held_out_relative_error'], 1e-9)

    def test_larger_output_budget_is_explicit_and_keeps_body_and_context_gates(self):
        raw=response(generated=20000)
        self.assertFalse(lg.sample_eligibility(raw,bench.digest(raw))['eligible'])
        self.assertTrue(lg.sample_eligibility(raw,bench.digest(raw),maximum=32768,context=131072)['eligible'])
        for changed in ({'text':''},{'reasoning_text':''},{'done_received':False},
                        {'usage':dict(raw['usage'],completion_tokens=32769)}):
            bad=dict(raw,**changed)
            self.assertFalse(lg.sample_eligibility(bad,bench.digest(bad),maximum=32768,context=131072)['eligible'])
        for context in (65767,True,'131072'):
            self.assertFalse(lg.sample_eligibility(raw,bench.digest(raw),maximum=32768,context=context)['eligible'])
        for cap in (True,8192,16385):
            with self.assertRaises(ValueError):lg.sample_eligibility(raw,bench.digest(raw),maximum=cap)

    def test_budget_journal_and_suite_bind_the_same_thinking_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=bench.RequestJournal(Path(tmp),lambda text:text)
            suite=bench.Suite('http://localhost','',131072,1,True,log=lambda _:None,
                want_long_generation=True,long_generation_only=True,generation_max_tokens=32768,
                workload_id='budget-control',journal=journal)
            def chat(_base,_key,messages,cap,think,**kwargs):
                self.assertEqual(cap,32768);self.assertIs(think,True)
                self.assertEqual(kwargs['request_overrides']['reasoning_budget_tokens'],0)
                raw=response(generated=20000)
                payload=json.dumps(bench.chat_body(messages,cap,think,kwargs.get('tools'),
                    kwargs.get('request_overrides'))).encode()
                raw['request_sha256']=hashlib.sha256(payload).hexdigest()
                return raw
            with patch.object(bench,'chat',side_effect=chat):suite.run_long_generation()
            self.assertEqual(suite.out['long_generation']['max_tokens'],32768)
            events=[json.loads(line)for line in (Path(tmp)/'requests.jsonl').read_text().splitlines()]
            for i,name in enumerate(lg.WORKLOADS):
                start,end=events[2*i:2*i+2]
                self.assertEqual(start['max_tokens'],32768);self.assertIs(start['thinking'],True)
                self.assertEqual(start['request_sha256'],end['sample']['request_sha256'])
                row=suite.out['long_generation']['workloads'][name]['requests'][0]
                self.assertTrue(row['eligible']);self.assertEqual(row['eligible'],end['sample']['eligible'])
                self.assertEqual(row['usage_generated_tokens'],20000)

    def test_custom_budget_requires_scope_and_known_sufficient_context(self):
        with self.assertRaises(ValueError):bench.Suite('', '',131072,1,True,generation_max_tokens=32768)
        for context in (None,0,65536):
            suite=bench.Suite('http://localhost','',context,1,True,log=lambda _:None,
                want_long_generation=True,long_generation_only=True,generation_max_tokens=32768)
            with patch.object(bench,'chat') as chat:suite.run_long_generation()
            chat.assert_not_called();self.assertIn('long-generation',suite.out['skipped'])

    def test_different_output_budgets_cannot_be_compared(self):
        metric={'median':10,'min':10,'max':10,'n':3}
        result={'machine':{},'complete':True,'comparison_identity':{
            'workload':{'nonce':'budget-control','nonce_policy':'fixed-id'},
            'long_generation_max_tokens':16384},'results':{
            'long_generation':{'workloads':{'regular-code':{'decode_tps':metric,'eligible_runs':3}}}}}
        other=json.loads(json.dumps(result));other['comparison_identity']['long_generation_max_tokens']=32768
        comparison=bench.compare_table(result,other)
        self.assertIn('not comparable',comparison);self.assertNotIn('+0.0%',comparison)

    def test_long_generation_gate_blocks_before_http_and_excludes_warmups(self):
        events=[]
        def gate(stage,phase,run):
            events.append((stage,phase,run))
            if stage=='begin':raise RuntimeError('small route not qualified')
        suite=bench.Suite('http://localhost','',131072,1,True,long_generation_only=True,
            want_long_generation=True,request_gate=gate)
        with patch.object(bench,'chat',return_value=response()) as chat:
            suite.request('warm-up/chat',None,None,[],24,True)
            self.assertEqual(events,[])
            with self.assertRaisesRegex(RuntimeError,'small route not qualified'):
                suite.request('long-generation/regular-code',0,None,[],16384,True)
            self.assertEqual(chat.call_count,1)
        self.assertEqual(events,[('begin','long-generation/regular-code',0)])
        events.clear()
        suite.request_gate=lambda stage,phase,run:events.append(stage)
        with patch.object(bench,'chat',side_effect=RuntimeError('stream failed')):
            with self.assertRaisesRegex(RuntimeError,'stream failed'):
                suite.request('long-generation/regular-code',0,None,[],16384,True)
        self.assertEqual(events,['begin','end'])

    def test_profile_gates_exclude_warmup_and_wrap_complete_request(self):
        events=[]
        suite=bench.Suite('http://localhost','',131072,1,True,log=lambda _:None,prefill_only=True,
            prefill_tokens=32768,request_gate=lambda stage,phase,run:events.append(stage))
        def reply(*args,**kwargs):
            events.append('http-complete')
            return response(generated=24)
        with patch.object(bench,'chat',side_effect=reply):
            suite.request('chat/warmup',0,None,[{'role':'user','content':'short'}],24,True)
            self.assertEqual(events,['http-complete']);events.clear()
            suite.request('prefill-only',0,None,[{'role':'user','content':'long'}],24,True)
        self.assertEqual(events,['begin','http-complete','end'])

    def test_profile_begin_failure_never_sends_http_and_failure_still_ends(self):
        events=[]
        def gate(stage,phase,run):
            events.append(stage)
            if stage=='begin':raise RuntimeError('owner unavailable')
        suite=bench.Suite('http://localhost','',131072,1,True,prefill_only=True,request_gate=gate)
        with patch.object(bench,'chat') as chat:
            with self.assertRaisesRegex(RuntimeError,'owner unavailable'):
                suite.request('prefill-only',0,None,[],24,True)
            chat.assert_not_called()
        self.assertEqual(events,['begin'])
        events.clear();suite.request_gate=lambda stage,phase,run:events.append(stage)
        with patch.object(bench,'chat',side_effect=RuntimeError('stream failure')):
            with self.assertRaisesRegex(RuntimeError,'stream failure'):
                suite.request('prefill-only',0,None,[],24,True)
        self.assertEqual(events,['begin','end'])

    def test_larger_prefill_checks_context_before_sending_and_keeps_actual_counts(self):
        for minimum in (54613, 98304, 131072):
            required = minimum * 5 // 4 + 24 + 256
            with self.subTest(minimum=minimum):
                too_small = bench.Suite("http://localhost", "", required - 1, 1, False,
                                        log=lambda _: None, prefill_only=True, prefill_tokens=minimum)
                with patch.object(bench, "chat") as chat:
                    with self.assertRaises(bench.EngineRequestError):
                        too_small.run_prefill_only()
                    chat.assert_not_called()

                # Actual engine counts may differ from the construction target.
                # Qualification must keep those counts, thinking and zero reuse.
                raw = response(generated=24)
                actual = minimum + 17
                raw["usage"]["prompt_tokens"] = actual
                raw["timings"]["prompt_n"] = actual
                suite = bench.Suite("http://localhost", "", required, 1, False,
                                    log=lambda _: None, prefill_only=True, prefill_tokens=minimum)
                with patch.object(bench, "chat", return_value=raw) as chat, \
                     patch.object(suite, "prefill_prompt", return_value="fixed test prompt"):
                    suite.run_prefill_only()
                self.assertEqual(chat.call_count, 2)  # short warmup and one long request
                self.assertTrue(all(call.args[3] == 24 and call.args[4] is True
                                    for call in chat.call_args_list))
                result = suite.out["prefill_only"]
                self.assertTrue(result["complete"])
                self.assertEqual(result["eligible_runs"], 1)
                self.assertEqual(result["requests"][0]["actual_read_tokens"], actual)

    def test_streamed_reasoning_uses_completion_usage(self):
        events = [{"choices": [{"delta": {"reasoning_content": "thinking"}}]},
                  {"choices": [{"delta": {"content": "answer "*100}}]},
                  {"choices": [{"delta": {}, "finish_reason": "length"}],
                   "usage": response()["usage"], "timings": response()["timings"]}]
        wire = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events) + b"data: [DONE]\n\n"
        with patch.object(bench.urllib.request, "urlopen", return_value=io.BytesIO(wire)) as request, \
             patch.object(bench.time, "monotonic_ns", side_effect=[10,20,30]):
            raw = bench.chat("http://localhost", "", [{"role": "user", "content": "task"}], lg.MAX_OUTPUT_TOKENS, True,
                             request_overrides={"chat_template_kwargs": {"enable_thinking": True},
                                                "reasoning_budget_tokens": 0})
        body = json.loads(request.call_args.args[0].data)
        self.assertEqual(body["max_tokens"], lg.MAX_OUTPUT_TOKENS)
        self.assertTrue(body["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(raw["text"], "answer "*100)
        self.assertEqual(raw["reasoning_text"], "thinking")
        self.assertEqual(bench.digest(raw)["generated_tokens"], 1152)
        self.assertTrue(lg.sample_eligibility(raw, bench.digest(raw))["eligible"])
        self.assertEqual(bench.digest(raw)['client_interval'],dict(request_begin_monotonic_ns=10,
            first_output_monotonic_ns=20,request_end_monotonic_ns=30))

    def test_short_missing_usage_counters_and_truncated_stream_are_ineligible(self):
        for change in ({"usage": {}}, {"done_received": False}, {"finish_reason": "tool_calls"}, {"text": ""}):
            raw = response()
            raw.update(change)
            self.assertFalse(lg.sample_eligibility(raw, bench.digest(raw))["eligible"])
        for generated in (0, 128, 1023, lg.MAX_OUTPUT_TOKENS+1):
            raw = response(generated=generated)
            self.assertFalse(lg.sample_eligibility(raw, bench.digest(raw))["eligible"])
        for offered, accepted in ((0, 0), (200, 100), (1000, 1001), (None, None)):
            raw = response(offered=offered, accepted=accepted)
            result = lg.sample_eligibility(raw, bench.digest(raw))
            self.assertFalse(result["eligible"])
            if offered == 0:
                self.assertIsNone(result["acceptance_rate"])

    def test_real_word_list_survives_and_workloads_alternate(self):
        calls = []

        def chat(_base, _key, messages, cap, think, **kwargs):
            task = messages[-1]["content"]
            calls.append(task)
            background = task.split("<background>\n")[1].split("\n</background>")[0]
            self.assertGreater(len(background.split()), 30000)
            self.assertNotIn("Summarize what kinds", task)
            self.assertEqual(think, lg.THINKING)
            self.assertEqual(cap, lg.MAX_OUTPUT_TOKENS)
            return response(accepted=1200 if "regular-code" in task else 320)

        suite = bench.Suite("http://localhost", "", 262144, 3, False, log=lambda _: None)
        suite.tokens_per_word = 1
        with patch.object(bench, "chat", side_effect=chat):
            suite.run_long_generation()
        out = suite.out["long_generation"]
        self.assertEqual([x["workload"] for x in out["order"]],
                         ["regular-code", "varied-prose", "varied-prose", "regular-code",
                          "regular-code", "varied-prose"])
        self.assertTrue(out["classification"]["qualified"])
        self.assertEqual(out["workloads"]["regular-code"]["pooled_acceptance_rate"], .75)
        self.assertEqual(len(set(calls)), 6)
        # Neither a one-run screening nor uniformly high agreement qualifies as a strong/weak pair.
        self.assertFalse(lg.classify_pair(out["workloads"], 1)["qualified"])
        for row in out["workloads"]["varied-prose"]["requests"]:
            row["acceptance_rate"] = .75
        self.assertFalse(lg.classify_pair(out["workloads"], 3)["qualified"])

    def test_zero_draft_throughput_mode_keeps_requests_and_agreement_separate(self):
        bodies=[]
        for speed_only in (False,True):
            captured=[]
            def chat(_base,_key,messages,cap,think,**kwargs):
                captured.append((messages,cap,think,kwargs))
                return response(offered=0,accepted=0)
            suite=bench.Suite('http://localhost','',131072,3,True,log=lambda _:None,
                want_long_generation=True,long_generation_only=True,generation_speed_only=speed_only,
                workload_id='mtp-net-control')
            with patch.object(bench,'chat',side_effect=chat):suite.run_long_generation()
            bodies.append(captured)
            self.assertEqual(bool(suite.out['failures']),not speed_only)
            result=suite.out['long_generation'];self.assertFalse(result['classification']['qualified'])
            for case in result['workloads'].values():
                self.assertEqual(case['eligible_runs'],0);self.assertEqual(case['speed_eligible_runs'],3)
                for sample in case['requests']:
                    self.assertFalse(sample['agreement_eligible']);self.assertIsNone(sample['acceptance_rate'])
        self.assertEqual(bodies[0],bodies[1])

    def test_throughput_mode_rejects_corrupt_counters_and_wrong_scope(self):
        for offered,accepted in [(None,None),(0,1),(-1,0),(True,0)]:
            raw=response(offered=offered,accepted=accepted);sample=bench.digest(raw)
            sample.update(lg.sample_eligibility(raw,sample))
            self.assertFalse(lg.run_eligible(sample,True))
        with self.assertRaises(ValueError):
            bench.Suite('http://localhost','',131072,1,True,generation_speed_only=True)

    def test_selected_long_generation_dispatch_keeps_thinking_and_cannot_qualify_pair(self):
        for selected in lg.WORKLOADS:
            with self.subTest(selected=selected):
                suite = bench.Suite("http://localhost", "", 262144, 3, False, log=lambda _: None,
                                    want_long_generation=True, long_generation_only=True,
                                    generation_workload=selected, workload_id="cpu-overhead")
                raw = response(accepted=1200 if selected == "regular-code" else 320)
                with patch.object(bench, "chat", return_value=raw) as chat:
                    suite.run_long_generation()
                self.assertEqual(chat.call_count, 3)
                for call in chat.call_args_list:
                    self.assertIn(" " + selected + " run ", call.args[2][-1]["content"])
                    self.assertEqual(call.args[3], lg.MAX_OUTPUT_TOKENS)
                    self.assertIs(call.args[4], True)
                    self.assertEqual(call.kwargs["request_overrides"], {
                        "chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget_tokens": 0})
                out = suite.out["long_generation"]
                self.assertEqual(out["selected_workloads"], [selected])
                self.assertTrue(out["diagnostic_subset"])
                self.assertEqual(out["workloads"][selected]["eligible_runs"], 3)
                other = next(name for name in lg.WORKLOADS if name != selected)
                self.assertEqual(out["workloads"][other]["requests"], [])
                self.assertFalse(out["classification"]["qualified"])
        with self.assertRaises(ValueError):
            bench.Suite("http://localhost", "", 262144, 1, False, generation_workload="regular-code")

    def test_agent_replays_actual_model_answers(self):
        snapshots = []

        def chat(_base, _key, messages, _cap, *_args, **_kwargs):
            snapshots.append(json.loads(json.dumps(messages)))
            r = response(generated=40)
            r["text"] = f"actual answer {len(snapshots)}"
            r["finish_reason"] = "stop"
            return r

        suite = bench.Suite("http://localhost", "", 262144, 1, False, log=lambda _: None)
        with patch.object(suite, "agent_files", return_value=[("public.txt", "file")]*bench.AGENT_TURNS), \
                patch.object(bench, "chat", side_effect=chat):
            suite.run_agent()
        self.assertEqual(len(snapshots), 6)
        self.assertEqual(snapshots[1][2], {"role": "assistant", "content": "actual answer 1", "reasoning_content": "reasoning"})
        sample = suite.out["agent"]["session_samples"][0]
        self.assertTrue(sample["complete"])
        self.assertEqual(sample["generated_tokens"], 240)
        self.assertEqual(sample["acceptance_rate"], .75)

    def test_agent_truncation_and_changed_workload_cannot_qualify(self):
        suite = bench.Suite("http://localhost", "", 262144, 1, False, log=lambda _: None)
        raw = response(generated=40)
        raw["done_received"] = False
        with patch.object(suite, "agent_files", return_value=[("public.txt", "file")]*bench.AGENT_TURNS), \
                patch.object(bench, "chat", return_value=raw):
            suite.run_agent()
        self.assertFalse(suite.out["agent"]["session_samples"][0]["complete"])
        metric = {"median": 10, "min": 10, "max": 10, "n": 3}
        common = {"chat": {}, "prompts": {}, "agent": {}, "clients": {}}
        old = {"machine": {}, "results": {**common, "long_generation": {
            "workload_version": 1, "workloads": {"regular-code": {"decode_tps": metric}}}}}
        new = {"machine": {}, "results": {**common, "long_generation": {
            "workload_version": 2, "workloads": {"regular-code": {"decode_tps": metric}}}}}
        table = bench.compare_table(old, new)
        self.assertIn("regular-code long decode", table)
        self.assertNotIn("+0.0%", table)
        self.assertIn("not comparable", table)


@unittest.skipUnless(os.environ.get('STRATA_BENCH_REAL_PACK'), 'optional pinned real tokenizer fixture')
class RealPackExactCounts(unittest.TestCase):
    def test_boundary_bodies_with_real_tokenizer_without_inference(self):
        import threading
        import strata_tokenizer as ST
        from serve.frontend import ChatTemplate, openai_to_messages
        from serve.server import MockEngine, Service, Server, make_handler, host_names_for
        folder=Path(os.environ['STRATA_BENCH_REAL_PACK'])/'tokenizer'
        names=['vocab.json','merges.txt','token_type.json','tokenizer.json','chat_template.jinja']
        def snapshot():
            return {name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in names}
        before=snapshot()
        identity=hashlib.sha256(json.dumps(before,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.assertEqual(identity,os.environ['STRATA_BENCH_REAL_PACK_SHA256'])
        vocab=json.loads((folder/'vocab.json').read_text())
        tokens=[None]*len(vocab)
        for text,index in vocab.items():
            tokens[index]=text
        tok=ST.Tokenizer(tokens,(folder/'merges.txt').read_text().split('\n'),
                         json.loads((folder/'token_type.json').read_text()))
        engine=MockEngine(tok,'ok',max_context=131072)
        service=Service(engine,tok,ChatTemplate(folder/'chat_template.jinja'))
        service.host_names=host_names_for('127.0.0.1',service.allowed_hosts,service.trusted_origins)
        httpd=Server(('127.0.0.1',0),make_handler(service))
        worker=threading.Thread(target=lambda:httpd.serve_forever(poll_interval=.01),daemon=True)
        worker.start()
        base=f'http://127.0.0.1:{httpd.server_address[1]}'
        try:
            with patch.object(engine,'generate',side_effect=AssertionError('count-only test must never infer')):
                for index,target in enumerate([73729,73730,81921,81922]):
                    suite=bench.Suite(base,'',131072,1,True,log=lambda _:None,prefill_only=True,
                        prefill_exact_tokens=target,workload_id='real-pack-exact')
                    messages,proof=bench.prefill_scaling.exact_prompt(suite,index,
                        lambda messages,timeout:bench.count_chat_tokens(base,'',messages,timeout))
                    body=bench.chat_body(messages,24,True)
                    normalized,tools,kwargs=openai_to_messages(service.with_shared(body,'openai'))
                    ids=service.encode_prompt(normalized,tools,kwargs,store=False)
                    self.assertEqual(len(ids),target)
                    self.assertLessEqual(proof['count_requests'],32)
                    print('EXACT_REAL_PACK '+json.dumps(dict(target_input_tokens=target,
                        rendered_input_tokens=len(ids),count_requests=proof['count_requests'],
                        construction_seconds=proof['construction_seconds'],
                        body_sha256=hashlib.sha256(json.dumps(body).encode()).hexdigest(),
                        token_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
                        native_engine_executed=False,physical_engine_rows_qualified=False)),flush=True)
        finally:
            httpd.shutdown();httpd.server_close();worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(snapshot(),before)


if __name__ == "__main__":
    unittest.main()
