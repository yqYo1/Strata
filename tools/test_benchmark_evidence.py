#!/usr/bin/env python3
"""CPU-only evidence regression tests; no real server or GPU."""
import copy
import io
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import subprocess
import threading
import select
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_bench as b
import long_generation_bench as lg
sys.path.insert(0, str(Path(__file__).resolve().parent / 'sycl'))
import generation_engine as generation


def raw(body="answer", offered=300):
    return {"first": .1, "total": 2., "text": body, "reasoning_text": "actual reasoning",
        "usage": {"completion_tokens": 1500, "prompt_tokens": 33000},
        "timings": {"predicted_per_second": 40., "predicted_ms": 1500., "cache_n": 0,
                    **({"draft_n": offered, "draft_n_accepted": offered // 2} if offered else {})},
        "err": None, "pieces": 10, "done_received": True, "finish_reason": "stop", "request_sha256": "a" * 64}


class EvidenceTests(unittest.TestCase):
    @staticmethod
    def continuation_sample(reused=45588):
        def footer(n, cached, read, pp, generated, tg, accepted, offered):
            return (f'strata serve: prompt {n} tokens = {cached} reused + {read} read '
                f'in {pp} ms (1.0 tok/s), {generated} generated in {tg} ms (1.0 tok/s), '
                f'drafts accepted {accepted} of {offered}, 0 checkpoints\n')
        text = (footer(33283, 0, 33283, 40000, 12288, 1000000, 5000, 6000) +
                footer(45589, reused, 45589-reused, 3, 2432, 100000, 1000, 2000))
        sample = dict(prompt_tokens=33283, reused_tokens=0, read_tokens=33283,
            generated_tokens=14738, prompt_ms=40003., decode_ms=1100000.,
            drafts_accepted=6000, drafts_offered=8000, decode_tps=13.4, decode_tps_source='engine')
        return text, sample

    def test_api_binds_all_native_segments_and_separates_inserted_tokens(self):
        # Same physical counts as the failed historical prose request, with
        # literal synthetic clocks. Both a reused and a reread continuation count.
        for reused in (0, 45588):
            with self.subTest(reused=reused):
                text, sample = self.continuation_sample(reused)
                proof = generation.api_segment_evidence(text, sample, insertion_token_counts=(18,))
                self.assertEqual(proof['native_generated_tokens'], 14720)
                self.assertEqual(proof['server_inserted_tokens'], 18)
                self.assertEqual(proof['api_generated_tokens'], 14738)
                self.assertEqual(proof['segments'][1]['reused_tokens'], reused)
                self.assertEqual(proof['segments'][1]['read_tokens'], 45589-reused)
                self.assertAlmostEqual(proof['exact_native_decode_tps'], 14720/1100)

    def test_single_native_generation_keeps_native_counts_without_insertions(self):
        text, _ = self.continuation_sample()
        sample = dict(prompt_tokens=33283, reused_tokens=0, read_tokens=33283,
            generated_tokens=12288, prompt_ms=40000.5, decode_ms=1000000.5,
            drafts_accepted=5000, drafts_offered=6000, decode_tps=12.3, decode_tps_source='engine')
        proof = generation.api_segment_evidence(text.splitlines()[0], sample)
        self.assertEqual(proof['server_inserted_tokens'], 0)
        self.assertEqual(len(proof['segments']), 1)
        self.assertEqual(proof['clock_binding']['rounding_bound_ms'], .55)

    def test_native_segment_binding_refuses_unknown_work_and_wrong_totals(self):
        text, sample = self.continuation_sample()
        bad = [(text, sample, ()), (text, dict(sample, generated_tokens=14720), (18,)),
            (text, dict(sample, drafts_accepted=6001), (18,)),
            (text, dict(sample, read_tokens=33282), (18,)),
            (text, dict(sample, prompt_ms=40005.), (18,)),
            (text, dict(sample, decode_ms=1100002.), (18,)),
            (text, dict(sample, decode_ms=float('nan')), (18,)),
            (text, dict(sample, decode_tps=13.5), (18,)),
            (text, dict(sample, decode_tps_source='client'), (18,)),
            (text, dict(sample, generated_tokens=True), (18,)),
            (text.replace('45588 reused', '45587 reused'), sample, (18,)),
            (text.replace('2432 generated', '2431 generated'), sample, (18,)),
            (text.replace('1 read in', '1 of 2 read in'), sample, (18,)),
            ('\n'.join(reversed(text.splitlines())), sample, (18,)),
            (text+text, sample, (18,)), ('', sample, (18,)), (text, sample, (True,))]
        for i, (actual, api, allowed) in enumerate(bad):
            with self.subTest(case=i), self.assertRaises(RuntimeError):
                generation.api_segment_evidence(actual, api, insertion_token_counts=allowed)

    def test_completed_api_sample_is_durable_before_end_gate(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            journal = b.RequestJournal(directory, lambda x: x)
            observed = []
            def gate(stage, phase, run):
                if stage == 'end':
                    rows = [json.loads(x) for x in (directory / 'requests.jsonl').read_text().splitlines()]
                    self.assertEqual([r['kind'] for r in rows], ['start', 'result'])
                    self.assertEqual(rows[-1]['sample']['generated_tokens'], 1500)
                    proof = b.verify_request_journal(directory)
                    self.assertTrue(proof['valid'])
                    self.assertEqual(len(proof['pending_gates']), 1)
                observed.append(stage)
            suite = b.Suite('unused', '', 262144, 1, True, journal=journal,
                want_long_generation=True, long_generation_only=True, request_gate=gate)
            with patch.object(b, 'chat', return_value=raw()):
                suite.request('long-generation/regular-code', 0, None, [], 16384, True)
            self.assertEqual(observed, ['begin', 'end'])
            journal.state(complete=True, terminal='complete')
            proof = b.verify_request_journal(directory)
            self.assertTrue(proof['complete'])
            self.assertEqual(proof['pending_gates'], [])

    def test_failed_end_gate_preserves_one_api_result_and_failed_gate(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            journal = b.RequestJournal(directory, lambda x: x)
            def gate(stage, phase, run):
                if stage == 'end':
                    raise RuntimeError('native segment coverage failed')
            suite = b.Suite('unused', '', 262144, 1, True, journal=journal,
                want_long_generation=True, long_generation_only=True, request_gate=gate)
            with patch.object(b, 'chat', return_value=raw()), self.assertRaisesRegex(
                    RuntimeError, 'native segment coverage failed'):
                suite.request('long-generation/regular-code', 0, None, [], 16384, True)
            rows = [json.loads(x) for x in (directory / 'requests.jsonl').read_text().splitlines()]
            self.assertEqual([r['kind'] for r in rows], ['start', 'result', 'gate'])
            self.assertEqual(rows[1]['sample']['generated_tokens'], 1500)
            self.assertEqual(rows[2]['status'], 'failed')
            proof = b.verify_request_journal(directory)
            self.assertTrue(proof['valid'])
            self.assertFalse(proof['complete'])
            self.assertEqual(proof['pending'], [])
            self.assertEqual(len(proof['failed_gates']), 1)
            journal.state(complete=True, terminal='complete')
            self.assertFalse(b.verify_request_journal(directory)['valid'])

    def test_request_error_and_end_gate_error_are_recorded_separately(self):
        for failure in ('begin', 'http', 'digest'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as td:
                directory = Path(td)
                journal = b.RequestJournal(directory, lambda x: x)
                stages = []
                def gate(stage, phase, run):
                    stages.append(stage)
                    if stage == failure:
                        raise RuntimeError('primary begin failure')
                    if stage == 'end':
                        raise RuntimeError('secondary end failure')
                suite = b.Suite('unused', '', 262144, 1, True, journal=journal,
                    want_long_generation=True, long_generation_only=True, request_gate=gate)
                reply = raw()
                if failure == 'digest':
                    del reply['text']
                expected = 'primary begin failure' if failure == 'begin' else 'primary HTTP failure' if failure == 'http' else 'text'
                with patch.object(b, 'chat', side_effect=RuntimeError('primary HTTP failure')
                        if failure == 'http' else None, return_value=reply) as chat, self.assertRaisesRegex(
                        (RuntimeError, KeyError), expected):
                    suite.request('long-generation/regular-code', 0, None, [], 16384, True)
                self.assertEqual(chat.call_count, 0 if failure == 'begin' else 1)
                self.assertEqual(stages, ['begin'] if failure == 'begin' else ['begin', 'end'])
                rows = [json.loads(x) for x in (directory / 'requests.jsonl').read_text().splitlines()]
                self.assertEqual([r['kind'] for r in rows], ['start', 'interrupted', 'gate'])
                self.assertIn(expected, rows[1]['error'])
                self.assertIn('begin failure' if failure == 'begin' else 'secondary end failure', rows[2]['error'])
                proof = b.verify_request_journal(directory)
                self.assertTrue(proof['valid'])
                self.assertEqual(proof['pending'], [])
                self.assertEqual(proof['pending_gates'], [])
                self.assertEqual(len(proof['failed_gates']), 1)

    def test_completed_api_with_unfinished_gate_is_incomplete_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            journal = b.RequestJournal(Path(td), lambda x: x)
            index = journal.append('start', 'prefill-only', 0, gate_required=True)
            journal.append('result', 'prefill-only', 0, request_index=index, sample={'ok': True})
            proof = b.verify_request_journal(td)
            self.assertTrue(proof['valid'])
            self.assertFalse(proof['complete'])
            self.assertEqual(proof['pending'], [])
            self.assertEqual(len(proof['pending_gates']), 1)
            journal.state(complete=True, terminal='complete')
            self.assertFalse(b.verify_request_journal(td)['valid'])

    def test_warmup_reconciliation_rejects_failed_missing_reordered_and_changed_requests(self):
        phases = [('warm-up/chat',None,24,True),('calibration',200,1,False),
                  ('calibration',800,1,False),('warm-up/prompt',None,8,True),
                  ('long-generation/regular-code',0,16384,True)]
        entries = [dict(kind='identity')]
        for i,(phase,run,maximum,thinking) in enumerate(phases):
            start = dict(kind='start',index=2*i+1,phase=phase,run=run,turn=None,
                max_tokens=maximum,thinking=thinking,request_sha256=str(i)*64)
            entries += [start,dict(kind='result',phase=phase,run=run,turn=None,
                request_index=start['index'],sample=dict(ok=True,request_sha256=start['request_sha256']))]
        proof = generation.warmup_reconciliation(entries)
        self.assertEqual(proof['successful_warmups'],4)
        self.assertEqual(proof['successful_long_requests'],1)
        larger=copy.deepcopy(entries);larger[-2]['max_tokens']=32768
        with self.assertRaises(RuntimeError):generation.warmup_reconciliation(larger)
        self.assertEqual(generation.warmup_reconciliation(larger,maximum_output=32768)['successful_long_requests'],1)
        for variant in ('missing','extra','reordered','failed','thinking','hash','request_index'):
            with self.subTest(variant=variant):
                rows = copy.deepcopy(entries)
                if variant=='missing': del rows[1:3]
                if variant=='extra': rows += rows[1:3]
                if variant=='reordered': rows[1:3],rows[3:5] = rows[3:5],rows[1:3]
                if variant=='failed': rows[2]['sample']['ok']=False
                if variant=='thinking': rows[1]['thinking']=False
                if variant=='hash': rows[2]['sample']['request_sha256']='f'*64
                if variant=='request_index': rows[2]['request_index']=123
                with self.assertRaises(RuntimeError): generation.warmup_reconciliation(rows)




    def test_incomplete_standard_response_cannot_enter_speed_summary(self):
        for change in ({"usage": {}}, {"done_received": False}, {"finish_reason": None}):
            r = raw(); r.update(change)
            d = b.digest(r)
            self.assertFalse(d["ok"])
            self.assertIsNotNone(d["err"])
        r = raw(); r["usage"] = {}; r["timings"] = {}
        self.assertIsNone(b.digest(r)["decode_tps"])

    def test_malformed_sse_is_an_error_even_after_a_content_piece(self):
        stream = io.BytesIO(b'data: {"choices":[{"delta":{"content":"fragment"}}]}\n\n'
                            b'data: {broken json}\n\n'
                            b'data: [DONE]\n\n')
        with patch.object(b.urllib.request, "urlopen", return_value=stream):
            r = b.chat("http://unused", "", [], 100, True)
        self.assertEqual(r["text"], "fragment")
        self.assertEqual(r["err"], "malformed SSE JSON")
        self.assertFalse(b.digest(r)["ok"])

    def test_cap_limited_agent_body_is_visible_but_not_a_complete_turn(self):
        r = raw(); r["finish_reason"] = "length"
        d = b.digest(r); d["output_chars"] = len(r["text"])
        self.assertTrue(d["ok"])
        self.assertTrue(d["truncated"])
        self.assertFalse(b.agent_turn_complete(d))

    def test_interrupted_later_request_keeps_durable_sample_and_chain(self):
        with tempfile.TemporaryDirectory() as td:
            j = b.RequestJournal(Path(td), lambda x: x)
            suite = b.Suite("unused", "", 262144, 1, True, journal=j)
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
        suite = b.Suite("unused", "", 262144, 2, True, log=lambda *a: None)
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
        suite = b.Suite("unused", "", 262144, 1, True, log=lambda *a: None)
        histories = []
        def reply(*args, **kwargs):
            histories.append(copy.deepcopy(args[2]))
            self.assertEqual(args[3:5], (b.AGENT_THINK_MAX_TOKENS, True))
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

    def test_external_kill_keeps_first_result_and_second_pending_start(self):
        with tempfile.TemporaryDirectory() as td:
            script = ("import sys,time; from pathlib import Path; import strata_bench as b; "
                      "from test_benchmark_evidence import raw; "
                      "j=b.RequestJournal(Path(sys.argv[1]),lambda x:x); "
                      "s=b.Suite('unused','',262144,1,True,journal=j); "
                      "b.chat=lambda *a,**k:raw(); s.request('agent',0,0,[],8192,True); "
                      "b.chat=lambda *a,**k:(print('READY',flush=True),time.sleep(30))[1]; "
                      "s.request('agent',0,1,[],8192,True)")
            child = subprocess.Popen([sys.executable, "-c", script, td], cwd=Path(b.__file__).parent,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                self.assertTrue(select.select([child.stdout], [], [], 10)[0])
                self.assertEqual(child.stdout.readline().strip(), b"READY")
                child.kill(); child.wait(timeout=5)
                proof = b.verify_request_journal(td)
                self.assertTrue(proof["valid"])
                self.assertFalse(proof["complete"])
                self.assertEqual(proof["complete_prefix_rows"], 3)
                self.assertEqual(len(proof["pending"]), 1)
                self.assertEqual(proof["pending"][0]["turn"], 1)
                self.assertEqual(len(proof["pending"][0]["request_sha256"]), 64)
            finally:
                if child.poll() is None:
                    child.kill(); child.wait(timeout=5)
                child.stdout.close(); child.stderr.close()

    def test_concurrent_pending_and_hash_chain(self):
        with tempfile.TemporaryDirectory() as td:
            j = b.RequestJournal(Path(td), lambda x:x)
            barrier = threading.Barrier(8)
            def worker(n):
                index = j.append("start", "clients", 0, n)
                barrier.wait(timeout=5)
                j.append("result", "clients", 0, n, request_index=index, sample={"ok":True})
            threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
            for t in threads: t.start()
            for t in threads: t.join(timeout=10); self.assertFalse(t.is_alive())
            proof = b.verify_request_journal(td)
            self.assertTrue(proof["valid"])
            self.assertEqual(proof["complete_prefix_rows"], 16)
            self.assertEqual(proof["pending"], [])
            self.assertFalse(proof["complete"])
            rows = (Path(td)/"requests.jsonl").read_text().splitlines()
            rows[3] = rows[3].replace('"phase": "clients"', '"phase": "altered"')
            (Path(td)/"requests.jsonl").write_text("\n".join(rows)+"\n")
            self.assertFalse(b.verify_request_journal(td)["valid"])

    def test_failure_after_suite_or_during_report_is_incomplete(self):
        for stage in ("metadata", "report"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as td:
                machine = {"model":"fixture", "context":262144}
                collect = [machine, RuntimeError("metadata failed")] if stage == "metadata" else [machine, machine]
                with patch.object(sys, "argv", ["strata_bench", "--url", "http://unused", "--out", td, "--think"]), \
                     patch.object(b, "is_strata", return_value=True), \
                     patch.object(b, "collect_machine", side_effect=collect), \
                     patch.object(b.Suite, "run", return_value={}), \
                     patch.object(b, "write_results", side_effect=OSError("report failed")), \
                     patch.object(b.Suite, "agent_files", return_value=[]):
                    with self.assertRaises((RuntimeError, OSError)):
                        b.main()
                directory = next(Path(td).iterdir())
                state = json.loads((directory/"journal-state.json").read_text())
                self.assertEqual(state["terminal"], "interrupted")
                self.assertFalse(state["complete"])
                self.assertTrue(b.verify_request_journal(directory)["valid"])

    def test_zero_usage_cannot_become_positive_from_stream_pieces(self):
        r = raw()
        r["usage"]["completion_tokens"] = 0
        d = b.digest(r)
        d["output_chars"] = len(r["text"])
        self.assertEqual(d["generated_tokens"], 0)
        self.assertFalse(b.agent_turn_complete(d))

    def test_thinking_agent_requires_history_and_output_context_room(self):
        suite = b.Suite("unused", "", 10240, 1, True, log=lambda *a: None)
        with patch.object(suite, "agent_session") as request:
            suite.run_agent()
        request.assert_not_called()
        self.assertIn("agent", suite.out["skipped"])

    def test_changed_identity_suppresses_comparison(self):
        stats = {"median": 10., "min": 10., "max": 10., "n": 3}
        old = {"results": {"chat": {"decode_tps": stats}, "prompts": {}, "agent": {}, "clients": {}},
               "machine": {}, "complete": True,
               "comparison_identity": {"client": "a", "model": "m", "quant": "q", "thinking": True,
                                       "workload": {"workload_id": "fixture", "nonce": "fixture", "nonce_policy": "fixed-id"}}}
        new = copy.deepcopy(old); new["results"]["chat"]["decode_tps"]["median"] = 20.
        self.assertIn("+100.0%", b.compare_table(old, new))
        for field, value in [("client", "b"), ("model", "other"), ("quant", "other"), ("thinking", False),
                             ("workload", {"workload_id": "other", "nonce": "other", "nonce_policy": "fixed-id"})]:
            changed = copy.deepcopy(new); changed["comparison_identity"][field] = value
            table = b.compare_table(old, changed)
            self.assertNotIn("+100.0%", table)
            self.assertIn(field, table)
        missing = copy.deepcopy(new)
        del missing["comparison_identity"]["workload"]
        table = b.compare_table(old, missing)
        self.assertNotIn("+100.0%", table)
        self.assertIn("missing actual nonce/workload policy", table)

if __name__ == "__main__":
    unittest.main()
