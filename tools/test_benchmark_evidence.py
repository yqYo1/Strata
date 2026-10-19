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

    def test_engine_comparison_rejects_wrong_build_observers_and_unmatched_work(self):
        builds = {arm: dict(binary=dict(path='/engines/'+arm, sha256=arm), build_source_commit=arm)
                  for arm in ('old', 'new')}
        records = []
        for i, arm in enumerate(generation.ENGINE_ARMS):
            payload = raw(body='a'*512)
            row = dict(b.digest(payload), **lg.sample_eligibility(payload, b.digest(payload)),
                workload='regular-code', order=0, run=0, output_chars=512, reasoning_chars=16)
            row.update(eligible=True, acceptance_rate=.75, drafts_offered=400, drafts_accepted=300)
            row['decode_ms'] = 1000. * (1.01 if arm == 'new' else 1.)
            row['decode_tps'] = row['generated_tokens'] * 1000 / row['decode_ms']
            records.append(('receipt'+str(i), dict(mode='engine-compare', cohort=i, engine_arm=arm,
                engine_comparison_plan=generation.engine_plan(), comparison_builds=builds,
                **builds[arm], complete=True, passed=True, ownership_closed=True, active=False,
                exit_code=0, engine_exit_code=0, fresh_faults=[], boot_unchanged=True,
                request_journal=dict(valid=True, complete=True), native_identity=dict(pid=100+i,start_ticks=10+i),
                environment=dict(STRATA_MEASUREMENT_ROOT='/r/'+str(i), EnableDirectSubmission='0'),
                command=['bench','--port',str(i)], config={}, profile_sha256='same', runtime_asset_delta={},
                source_aggregate_sha256='same', qualification_files={}, source_commit='same', requests=[row],
                observer_records_absent=True)))
        result = generation.overhead_statistics(records, engine_compare=True)
        self.assertTrue(result['diagnostic_nonregression_gate_passed'])
        self.assertAlmostEqual(result['paired_latency_ratio_geomean'],1.01)
        self.assertEqual(len(result['arm_statistics']['old']['independent_process_tps']),3)
        slow = copy.deepcopy(records)
        for _, r in slow:
            if r['engine_arm']=='new': r['requests'][0]['decode_ms']*=1.2
        self.assertFalse(generation.overhead_statistics(slow, True)['diagnostic_nonregression_gate_passed'])
        for variant in ('wrong_binary','wrong_source','zero_flag','validation_layer','runtime_change',
                        'same_engine','work_change','duplicate_process'):
            with self.subTest(variant=variant):
                changed=copy.deepcopy(records); r=changed[1][1]
                if variant=='wrong_binary': r['binary']=changed[0][1]['binary']
                if variant=='wrong_source': r['build_source_commit']='other'
                if variant=='zero_flag': r['environment']['STRATA_DECODE_CPU_PROFILE']='0'
                if variant=='validation_layer': r['environment']['UR_ENABLE_LAYERS']='UR_LAYER_PARAMETER_VALIDATION'
                if variant=='runtime_change': r['runtime_asset_delta']={'other':'runtime'}
                if variant=='same_engine': r['comparison_builds']['new']=r['comparison_builds']['old']
                if variant=='work_change': r['requests'][0]['request_sha256']='b'*64
                if variant=='duplicate_process': r['native_identity']=changed[0][1]['native_identity']
                with self.assertRaises(RuntimeError): generation.overhead_statistics(changed, True)

    def test_overhead_requires_independent_matched_work_and_preserves_failed_speed_gate(self):
        records = []
        for i, arm in enumerate(generation.OVERHEAD_ARMS):
            sample = lg.sample_eligibility(raw(body='a'*512), b.digest(raw(body='a'*512)))
            row = dict(b.digest(raw(body='a'*512)), **sample, workload='regular-code',
                order=0, run=0, output_chars=512, reasoning_chars=16)
            row['eligible'] = True
            row['acceptance_rate'] = .75
            row['drafts_offered'], row['drafts_accepted'] = 400, 300
            row['decode_ms'] = 1000. * (1.02 if arm == 'on' else 1.)
            row['decode_tps'] = row['generated_tokens'] * 1000 / row['decode_ms']
            env = dict(STRATA_MEASUREMENT_ROOT='/r/'+str(i), EnableDirectSubmission='0')
            if arm == 'on': env.update(STRATA_DECODE_CPU_PROFILE='1', STRATA_DECODE_TIMING='1')
            record = dict(mode='cpu-overhead', cohort=i, observer_arm=arm,
                observer_comparison_plan=generation.overhead_plan(), complete=True, passed=True,
                ownership_closed=True, active=False, exit_code=0, engine_exit_code=0,
                fresh_faults=[], boot_unchanged=True, request_journal=dict(valid=True, complete=True),
                native_identity=dict(pid=100+i,start_ticks=10+i), environment=env, command=['bench','--port',str(i)],
                binary='same', build_source_commit='same', config={}, profile_sha256='same',
                runtime_asset_delta={}, source_aggregate_sha256='same', qualification_files={},
                source_commit='same', requests=[row])
            if arm == 'on':
                record.update(cpu_record_reconciliation=dict(successful_requests=5, valid_CPU_records=5,
                    warmup_and_calibration_records_excluded=4, all_request_counts_reconciled=True),
                    cpu_observations=[dict(valid=True)])
            else: record['observer_records_absent'] = True
            records.append(('receipt'+str(i),record))
        result = generation.overhead_statistics(records)
        self.assertTrue(result['observed_overhead_gate_passed'])
        self.assertAlmostEqual(result['paired_latency_ratio_geomean'],1.02)
        slow = copy.deepcopy(records)
        for _, r in slow:
            if r['observer_arm']=='on': r['requests'][0]['decode_ms']*=1.2
        self.assertFalse(generation.overhead_statistics(slow)['observed_overhead_gate_passed'])
        for variant in ('duplicate_process','safety_change','different_work','wrong_order','incomplete_accounting'):
            with self.subTest(variant=variant):
                changed=copy.deepcopy(records)
                r=changed[1][1]
                if variant=='duplicate_process': r['native_identity']=changed[0][1]['native_identity']
                if variant=='safety_change': r['environment']['EnableDirectSubmission']='1'
                if variant=='different_work': r['requests'][0]['drafts_offered']=800;r['requests'][0]['drafts_accepted']=600
                if variant=='wrong_order': r['cohort']=2
                if variant=='incomplete_accounting': r['cpu_record_reconciliation']['valid_CPU_records']=4
                with self.assertRaises(RuntimeError): generation.overhead_statistics(changed)

    def test_CPU_observation_reconciliation_rejects_missing_invalid_and_mismatched_requests(self):
        phases = ['warm-up/chat', 'calibration', 'calibration', 'warm-up/prompt',
                  'long-generation/regular-code', 'long-generation/varied-prose']
        inputs, outputs = [19, 240, 900, 1080, 33000, 33285], [24, 1, 1, 8, 1500, 1800]
        entries = [dict(kind='result', phase=phase, sample=dict(ok=True, prompt_tokens=n, generated_tokens=m))
                   for phase, n, m in zip(phases, inputs, outputs)]
        values = [dict(schema=1, input_tokens=n, generated_tokens=m, valid=True, single_stage=True,
            clock_deltas_valid=True, quiescent=True, worker_jobs_fit=True, main_subsets_fit=True,
            counter_errors=0, unwound=0, snapshot_read_errors=0, unexpected_mode_tasks=0,
            wall_through_commit_ms=1000., windows=max(0, m-1), worker_job_calls=100,
            native_quant_calls=10, sync_calls=10, main_thread_ns=10000, native_process_ns=20000)
            for n, m in zip(inputs, outputs)]
        rows = [dict(workload=phase.split('/', 1)[1], prompt_tokens=n, generated_tokens=m)
                for phase, n, m in zip(phases[-2:], inputs[-2:], outputs[-2:])]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); journal = root/'bench'/'run'/'requests.jsonl'
            journal.parent.mkdir(parents=True)

            def observe(records, events=entries):
                journal.write_text('\n'.join(json.dumps(e) for e in events)+'\n')
                log = '\n'.join('strata decode cpu: '+json.dumps(v) for v in records)
                return generation.cpu_observations(root, log, rows)

            selected, counts = observe(values)
            self.assertEqual([v['workload'] for v in selected], ['regular-code', 'varied-prose'])
            self.assertEqual(counts['valid_CPU_records'], 6)
            self.assertEqual(counts['warmup_and_calibration_records_excluded'], 4)
            for variant in ('missing', 'duplicate', 'invalid_warmup', 'clock_error', 'no_work', 'bad_subset'):
                with self.subTest(variant=variant):
                    changed = copy.deepcopy(values)
                    if variant == 'missing': changed.pop()
                    if variant == 'duplicate': changed[-1] = changed[-2]
                    if variant == 'invalid_warmup': changed[0]['valid'] = False
                    if variant == 'clock_error': changed[-1]['counter_errors'] = 1
                    if variant == 'no_work': changed[-1]['native_quant_calls'] = 0
                    if variant == 'bad_subset': changed[-1]['main_subsets_fit'] = False
                    with self.assertRaises(RuntimeError): observe(changed)
            changed = copy.deepcopy(entries); changed[-1]['phase'] = 'unexpected'
            with self.assertRaises(RuntimeError): observe(values, changed)
            journal.write_text('\n'.join(json.dumps(e) for e in entries[:-1])+'\n')
            log='\n'.join('strata decode cpu: '+json.dumps(v) for v in values[:-1])
            selected,counts=generation.cpu_observations(root,log,rows[:1])
            self.assertEqual(len(selected),1)
            self.assertEqual(counts['valid_CPU_records'],5)

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
