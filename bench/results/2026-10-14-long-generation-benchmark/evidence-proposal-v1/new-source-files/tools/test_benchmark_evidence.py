#!/usr/bin/env python3
"""CPU-only evidence regression tests; no real server or GPU."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_bench as b
import long_generation_bench as lg


def raw(body="answer", offered=300):
    return {"first": .1, "total": 2., "text": body, "reasoning_text": "actual reasoning",
        "usage": {"completion_tokens": 1500, "prompt_tokens": 33000},
        "timings": {"predicted_per_second": 40., "predicted_ms": 1500., "cache_n": 0,
                    **({"draft_n": offered, "draft_n_accepted": offered // 2} if offered else {})},
        "err": None, "pieces": 10, "done_received": True, "finish_reason": "stop", "request_sha256": "a" * 64}


class EvidenceTests(unittest.TestCase):
    def test_interrupted_later_request_keeps_durable_sample_and_chain(self):
        with tempfile.TemporaryDirectory() as td:
            j = b.RequestJournal(Path(td), lambda x: x)
            suite = b.Suite("unused", "", 50000, 1, True, journal=j)
            with patch.object(b, "chat", side_effect=[raw(), KeyboardInterrupt()]):
                suite.request("agent", 0, 0, [], 2048, True)
                with self.assertRaises(KeyboardInterrupt):
                    suite.request("agent", 0, 1, [], 2048, True)
            rows = [json.loads(x) for x in (Path(td)/"requests.jsonl").read_text().splitlines()]
            self.assertEqual([r["kind"] for r in rows], ["start", "result", "start", "interrupted"])
            self.assertEqual(rows[1]["sample"]["generated_tokens"], 1500)
            previous = None
            for r in rows:
                digest = r.pop("sha256")
                self.assertEqual(r["previous_sha256"], previous)
                self.assertEqual(hashlib.sha256(json.dumps(r, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest(), digest)
                previous = digest
            self.assertFalse(json.loads((Path(td)/"journal-state.json").read_text())["complete"])

    def test_partial_sessions_retained_and_excluded(self):
        suite = b.Suite("unused", "", 50000, 2, True, log=lambda *a: None)
        d = b.digest(raw()); d["output_chars"] = 6
        complete = [dict(d, turn=i) for i in range(6)]
        partial = [dict(d, turn=i, ttft_s=99.) for i in range(5)]
        with patch.object(suite, "agent_session", side_effect=[complete, partial]):
            suite.run_agent()
        a = suite.out["agent"]
        self.assertEqual(sum(map(len, a["sessions"])), 11)
        self.assertEqual((a["complete_sessions"], a["partial_sessions"]), (1, 1))
        self.assertEqual(a["turn_ttft_s"]["median"], .1)
        self.assertEqual(a["turn_ttft_s"]["n"], 5)
        missing_done = copy.deepcopy(complete); missing_done[3]["stream_complete"] = False
        self.assertFalse(b.agent_complete(missing_done))

    def test_actual_agent_body_and_reasoning_replayed(self):
        suite = b.Suite("unused", "", 50000, 1, True, log=lambda *a: None)
        histories = []
        def reply(*args, **kwargs):
            histories.append(copy.deepcopy(args[2]))
            self.assertEqual(args[3:5], (2048, True))
            return raw()
        with patch.object(suite, "agent_files", return_value=[("f", "file")]*5), patch.object(b, "chat", side_effect=reply):
            turns = suite.agent_session(0)
        self.assertTrue(b.agent_complete(turns))
        self.assertEqual(histories[1][2], {"role": "assistant", "content": "answer", "reasoning_content": "actual reasoning"})
        with patch.object(b, "chat", return_value=raw(body="")):
            self.assertEqual(len(suite.agent_session(0)), 1)

    def test_thinking_wire_payload(self):
        class Stream:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __iter__(self): return iter([b'data: [DONE]\n'])
        with patch.object(b.urllib.request, "urlopen", return_value=Stream()) as open_request:
            b.chat("http://unused", "", [], 2048, True)
        body = json.loads(open_request.call_args.args[0].data)
        self.assertTrue(body["chat_template_kwargs"]["enable_thinking"])
        self.assertTrue(body["chat_template_kwargs"]["preserve_thinking"])
        self.assertEqual(body["reasoning_budget_tokens"], 0)

    def test_no_offer_baseline_has_speed_not_agreement(self):
        r = raw(body="a"*512, offered=0)
        eligibility = lg.sample_eligibility(r, b.digest(r))
        self.assertTrue(eligibility["speed_eligible"])
        self.assertFalse(eligibility["agreement_eligible"])
        self.assertIsNone(eligibility["acceptance_rate"])
        r["usage"]["completion_tokens"] = 20
        self.assertFalse(lg.sample_eligibility(r, b.digest(r))["speed_eligible"])

    def test_changed_identity_suppresses_comparison(self):
        stats = {"median": 10., "min": 10., "max": 10., "n": 3}
        old = {"results": {"chat": {"decode_tps": stats}, "prompts": {}, "agent": {}, "clients": {}},
               "machine": {}, "complete": True,
               "comparison_identity": {"client": "a", "model": "m", "quant": "q", "thinking": True}}
        new = copy.deepcopy(old); new["results"]["chat"]["decode_tps"]["median"] = 20.
        self.assertIn("+100.0%", b.compare_table(old, new))
        for field, value in [("client", "b"), ("model", "other"), ("quant", "other"), ("thinking", False)]:
            changed = copy.deepcopy(new); changed["comparison_identity"][field] = value
            table = b.compare_table(old, changed)
            self.assertNotIn("+100.0%", table)
            self.assertIn(field, table)

if __name__ == "__main__":
    unittest.main()
