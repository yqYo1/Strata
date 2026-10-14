from pathlib import Path
import hashlib,json,difflib
P=Path('/home/yayoi/orca/workspaces/Strata/sycl');D=P/'bench/results/2026-10-14-long-generation-benchmark/evidence-proposal-v1';S=D/'new-source-files'
p=S/'tools/strata_bench.py';s=p.read_text()
a=s.index('                t = threading.Thread(target=serve_load.one,');b=s.index('                t.start()',a)
s=s[:a]+'''                def worker(index=i):
                    if self.journal:
                        self.journal.append("start", "clients", rd, index)
                    try:
                        serve_load.one(self.base, self.key, "strata", index, 400, BURST_MAX_TOKENS,
                                       res, {"chat_template_kwargs": {"enable_thinking": True},
                                             "reasoning_budget_tokens": 0} if self.think else {},
                                       self.think, f" {self.tag('b', n, rd)}")
                        r = res[index]
                        sample = {k: v for k, v in r.items() if k != "text"}
                        sample["output_sha256"] = hashlib.sha256(r["text"].encode()).hexdigest()
                        if self.journal:
                            self.journal.append("result", "clients", rd, index, sample=sample)
                    except BaseException as error:
                        if self.journal:
                            self.journal.append("interrupted", "clients", rd, index, error=repr(error)[:300])
                        res[index] = {"first": None, "tokens": 0, "total": 0, "text": "", "err": repr(error)[:300]}
                t = threading.Thread(target=worker)
''' +s[b:]
s=s.replace('has_counts = all(type(x) is int for x in offered + accepted)','has_counts = all(type(o) is int and type(a) is int and 0 <= a <= o\n                             for o, a in zip(offered, accepted))')
s=s.replace('    rows = [f\'{\'\':32s}', '''    unqualified_long = bool(go and gn and any(
        case.get("eligible_runs", 0) < 3
        for gen in (go, gn) for case in gen.get("workloads", {}).values()))
    rows = [f'{\'\':32s}''') if False else s
s=s.replace('        ch = f"{(y - x)', '''        metric_compatible = compatible
        if "long decode" in k or "MTP acceptance" in k:
            metric_compatible = compatible and bool(go and gn) and all(
                c.get("eligible_runs", 0) >= 3 for g in (go, gn) for c in g["workloads"].values())
        ch = f"{(y - x)''')
s=s.replace('if compatible and x and y is not None else ""','if metric_compatible and x and y is not None else ""')
p.write_text(s)
# Tests are source only: main must apply and execute serially later.
tests='''#!/usr/bin/env python3
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
            def __iter__(self): return iter([b'data: [DONE]\\n'])
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
'''
(S/'tools/test_benchmark_evidence.py').write_text(tests)
identity=json.loads((D/'identity.json').read_text());patches=[];new={}
for n in identity['bases']:
    before=(P/n).read_text();after=(S/n).read_text()
    patches.append(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+n,tofile='b/'+n)))
    new[n]=hashlib.sha256(after.encode()).hexdigest()
n='tools/test_benchmark_evidence.py';after=(S/n).read_text();patches.append(''.join(difflib.unified_diff([],after.splitlines(True),fromfile='/dev/null',tofile='b/'+n)));new[n]=hashlib.sha256(after.encode()).hexdigest()
proposal=''.join(patches);(D/'proposal.patch').write_text(proposal);identity['proposed_sources']=new;identity['proposal_sha256']=hashlib.sha256(proposal.encode()).hexdigest();(D/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
