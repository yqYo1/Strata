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
    def test_streamed_reasoning_uses_completion_usage(self):
        events = [{"choices": [{"delta": {"reasoning_content": "thinking"}}]},
                  {"choices": [{"delta": {"content": "answer "*100}}]},
                  {"choices": [{"delta": {}, "finish_reason": "length"}],
                   "usage": response()["usage"], "timings": response()["timings"]}]
        wire = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events) + b"data: [DONE]\n\n"
        with patch.object(bench.urllib.request, "urlopen", return_value=io.BytesIO(wire)) as request:
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

    def test_agent_replays_actual_model_answers(self):
        snapshots = []

        def chat(_base, _key, messages, _cap, *_args, **_kwargs):
            snapshots.append(json.loads(json.dumps(messages)))
            r = response(generated=40)
            r["text"] = f"actual answer {len(snapshots)}"
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
