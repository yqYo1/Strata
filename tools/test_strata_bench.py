"""HTTP/usage and eligibility regression checks; no model or GPU is needed."""
import io
import json
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
        for minimum in (98304, 131072):
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


if __name__ == "__main__":
    unittest.main()
