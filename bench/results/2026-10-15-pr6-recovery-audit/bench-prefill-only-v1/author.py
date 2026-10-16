from pathlib import Path
import difflib,hashlib,json
ROOT=Path.cwd(); D=ROOT/'bench/results/2026-10-15-pr6-recovery-audit/bench-prefill-only-v1'
base=(ROOT/'tools/strata_bench.py').read_text();s=base
s=s.replace('workload_id=None):','workload_id=None, prefill_only=False, prefill_tokens=65536):',1)
s=s.replace('        self.clients, self.want_long = want_clients, want_long','''        self.prefill_only, self.prefill_tokens = prefill_only, prefill_tokens
        if prefill_only and (want_clients or want_long or want_long_generation or long_generation_only or quick):
            raise ValueError("prefill-only cannot be combined with other suite scopes")
        if prefill_tokens not in (32768,65536):
            raise ValueError("prefill minimum must be 32768 or 65536")
        if prefill_only:
            self.think = True
        self.clients, self.want_long = want_clients, want_long''',1)
s=s.replace('        self.fail_fast_engine_error = fail_fast_engine_error','        self.fail_fast_engine_error = fail_fast_engine_error or prefill_only',1)
s=s.replace('        if phase.startswith("long-generation/"):', '''        if phase == "prefill-only":
            row.update(prefill_eligibility(raw, row, self.prefill_tokens, self.ctx))
            if not row["eligible"]:
                row.update(ok=False, err="; ".join(row["eligibility_errors"]))
        if phase.startswith("long-generation/"):''',1)
s=s.replace('        if raw.get("_interrupted") is not None:\n            raise raw["_interrupted"]','''        if raw.get("_interrupted") is not None:
            if self.prefill_only:
                self.out.setdefault("request_samples", []).append({"phase":phase,"run":run,"turn":turn,"sample":row})
            raise raw["_interrupted"]''',1)
anchor='    def run_long_generation(self):'
methods='''    def prefill_plan(self):
        return {"mode":"prefill-only", "requested_min_tokens":self.prefill_tokens,
                "construction_target_tokens":self.prefill_tokens * 5 // 4,
                "construction_tokens_per_word":1.1, "construction_overhead":40,
                "construction_version":1, "max_tokens":LONG_MAX_TOKENS,
                "thinking_requested":True, "workload":self.workload_identity(),
                "freshness":"distinct run tags; reported reuse must equal zero",
                "calibration":"none; body never repaired from server responses"}

    def prefill_prompt(self, run):
        # Reuse the official prompt generator, with fixed construction constants.
        # Independent of all server responses and mutable legacy calibration state.
        words=max(50,int((self.prefill_tokens * 5 // 4 - 40)/1.1))
        return [{"role":"user", "content":f"[bench {self.tag(run, self.prefill_tokens, 'prefill-only-v1')}]\\n" + long_prompt(words)}]

    def run_prefill_only(self):
        plan=self.prefill_plan()
        cohort={**plan,"requests":[],"complete":False,"eligible_runs":0}
        self.out["prefill_only"]=cohort
        required=plan["construction_target_tokens"]+LONG_MAX_TOKENS+256
        if type(self.ctx) is not int or self.ctx < required:
            reason=f"prefill-only needs known context >= {required}, got {self.ctx!r}; no body repair"
            self.out["terminal_failure"]={"phase":"prefill-only/setup","error":reason}
            self.fail("prefill-only/setup",{"err":reason})
            raise EngineRequestError(reason)
        # Only the existing bounded short chat warmup; no calibration requests.
        self.request("warm-up/chat",None,None,[{"role":"user","content":PROMPTS[0]}],24,True)
        label=f"{self.prefill_tokens}-minimum"
        self.out["prompts"][label]=cohort
        try:
            for run in range(self.runs):
                before=len(self.out.get("request_samples",[]))
                try:
                    self.request("prefill-only",run,None,self.prefill_prompt(run),LONG_MAX_TOKENS,True)
                finally:
                    # Failed and interrupted returned responses remain individual samples.
                    for sample in self.out.get("request_samples",[])[before:]:
                        if sample["phase"]=="prefill-only":
                            cohort["requests"].append({**sample["sample"],"run":run})
        finally:
            good=[r for r in cohort["requests"] if r.get("eligible") and r.get("ok")]
            cohort.update(eligible_runs=len(good),complete=len(good)==self.runs,
                          prompt_tps=stat(r["prompt_tps"] for r in good),
                          ttft_s=stat(r["ttft_s"] for r in good),
                          prompt_tokens=stat(r["actual_prompt_tokens"] for r in good),
                          read_tokens=stat(r["actual_read_tokens"] for r in good),
                          reused_tokens=stat(r["actual_reused_tokens"] for r in good))
        return self.out

'''
s=s.replace(anchor,methods+anchor,1)
s=s.replace('    def run(self):\n        self.warm_up()', '    def run(self):\n        if self.prefill_only:\n            return self.run_prefill_only()\n        self.warm_up()',1)
# Strict diagnostic counters: never infer prompt_n from usage-minus-cache.
anchor='class Suite:'
helper='''def prefill_eligibility(raw, row, minimum, context):
    usage,timings=raw.get("usage") or {},raw.get("timings") or {}
    prompt,read=usage.get("prompt_tokens"),timings.get("prompt_n")
    cache_usage=(usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    cache_engine=timings.get("cache_n")
    caches=[v for v in (cache_usage,cache_engine) if v is not None]
    errors=[]
    if not row.get("ok"): errors.append(row.get("err") or "incomplete response")
    for name,value in (("usage.prompt_tokens",prompt),("timings.prompt_n",read)):
        if type(value) is not int or value < minimum: errors.append(f"{name} must report >= {minimum}, got {value!r}")
    if not caches or any(type(v) is not int or v!=0 for v in caches):
        errors.append(f"every reported reuse count must be zero, got {caches!r}")
    generated=usage.get("completion_tokens")
    if type(generated) is not int or not 1<=generated<=LONG_MAX_TOKENS:
        errors.append("completion usage outside 1..24")
    if type(context) is not int or type(prompt) is not int or prompt+LONG_MAX_TOKENS+256>context:
        errors.append("actual prompt lacks verified context/output headroom")
    if type(read) is int and type(prompt) is int and read>prompt:
        errors.append("reported read count exceeds reported prompt count")
    pp=timings.get("prompt_per_second")
    if type(pp) not in (float,int) or not 0<pp<float("inf"):
        errors.append("missing finite positive engine prompt_per_second")
    return {"eligible":not errors,"eligibility_errors":errors,"requested_min_tokens":minimum,
            "actual_prompt_tokens":prompt,"actual_read_tokens":read,"read_tokens":read,
            "actual_reused_tokens":caches[0] if caches else None,
            "prompt_count_source":"usage.prompt_tokens","read_count_source":"timings.prompt_n",
            "reuse_counts_reported":{"usage":cache_usage,"timings":cache_engine},
            "thinking_requested":True,"reasoning_observed":bool(raw.get("reasoning_text")),
            "reasoning_tokens_reported":(usage.get("completion_tokens_details") or {}).get("reasoning_tokens")}


'''
s=s.replace(anchor,helper+anchor,1)
# Reuse Suite.prompt_for itself; opt-in fixed construction avoids arm-dependent calibration.
s=s.replace('    def prompt_for(self, tokens: int, run: int) -> list:\n        words = max(50, int((tokens - self.overhead) / self.tokens_per_word))',
            '    def prompt_for(self, tokens: int, run: int, fixed=False) -> list:\n        rate,offset=(1.1,40) if fixed else (self.tokens_per_word,self.overhead)\n        words = max(50, int((tokens - offset) / rate))',1)
start=s.index('        words=max(50,int((self.prefill_tokens')
end=s.index('\n\n    def run_prefill_only',start)
s=s[:start]+'        return self.prompt_for(self.prefill_tokens * 5 // 4,run,fixed=True)'+s[end:]
# Absent cohorts are valid for this bounded scope and interrupted results.
s=s.replace('out["chat"].get(', 'out.get("chat", {}).get(').replace('out["prompts"].items()', 'out.get("prompts", {}).items()')
for key in ('chat','agent','clients'):s=s.replace(f'= out["{key}"]',f'= out.get("{key}", {{}})')
anchor='    mc, out, p = res["machine"], res["results"], res["params"]'
s=s.replace(anchor,anchor+'''
    if p.get("mode")=="prefill-only":
        cohort=out.get("prefill_only",{})
        lines=["# Prefill-only diagnostic", "",
               f"Complete: {res.get('complete',False)}. Minimum actual prompt/read tokens: {p['prefill_tokens']}. "
               f"Thinking requested: true; output cap: {LONG_MAX_TOKENS}. No decode/agent qualification.", "",
               "One short warmup, then fresh long requests. No calibration or dynamic body repair.",
               "Workload provenance: `"+json.dumps(p.get("workload"),sort_keys=True)+"`", "",
               "| Run | Actual prompt | Actual read | Actual reused | Reasoning observed | Eligible | Prompt tok/s |",
               "| --- | ---: | ---: | ---: | --- | --- | ---: |"]
        for r in cohort.get("requests",[]):
            lines.append(f"| {r.get('run')} | {r.get('actual_prompt_tokens')} | {r.get('actual_read_tokens')} | "
                         f"{r.get('actual_reused_tokens')} | {r.get('reasoning_observed')} | {r.get('eligible')} | {r.get('prompt_tps')} |")
        lines += ["", "Individual counts, body SHA256, sources, failures and construction plan: `results.json`; "
                  "durable ordered requests: `requests.jsonl` and `journal-state.json`.", "",
                  "Failure: "+str(res.get("terminal_failure",out.get("terminal_failure",{}))), "",
                  "No physical-memory, quality or optimization-adoption result follows from this client diagnostic."]
        return "\\n".join(lines)+"\\n"
''',1)
s=s.replace('    ap.add_argument("--think",', '''    ap.add_argument("--prefill-only", action="store_true", help="only one short warmup and fresh long prefill requests; thinking on, output capped at 24")
    ap.add_argument("--prefill-tokens", type=int, choices=(32768,65536), default=None, help="minimum ACTUAL prompt/read tokens for --prefill-only (default 65536)")
    ap.add_argument("--think",''',1)
s=s.replace('    say = (lambda *x, **k: None)', '''    if a.prefill_tokens is not None and not a.prefill_only:
        ap.error("--prefill-tokens requires --prefill-only")
    if a.prefill_only and (a.clients or a.long or a.long_generation or a.long_generation_only or a.compare_only):
        ap.error("--prefill-only cannot be combined with other suite scopes or --compare-only")
    if a.prefill_only and a.runs<1:
        ap.error("--prefill-only needs positive --runs")
    if a.prefill_only:
        a.think=True
    a.prefill_tokens=a.prefill_tokens or 65536
    say = (lambda *x, **k: None)''',1)
s=s.replace('fail_fast_engine_error=a.fail_fast_engine_error, workload_id=a.workload_id)', 'fail_fast_engine_error=a.fail_fast_engine_error, workload_id=a.workload_id,\n                      prefill_only=a.prefill_only, prefill_tokens=a.prefill_tokens)',1)
s=s.replace('fail_fast_engine_error=a.fail_fast_engine_error, workload=suite.workload_identity())', 'fail_fast_engine_error=suite.fail_fast_engine_error, workload=suite.workload_identity(),\n                       mode="prefill-only" if a.prefill_only else "suite",\n                       prefill_plan=suite.prefill_plan() if a.prefill_only else None)',1)
s=s.replace('        say("Running the suite (long generation', '        say("Running prefill-only diagnostic ..." if a.prefill_only else "Running the suite (long generation',1)
s=s.replace('"params": {"workload": suite.workload_identity(),','"params": {"mode":"prefill-only" if a.prefill_only else "suite", "prefill_tokens":a.prefill_tokens if a.prefill_only else None, "workload": suite.workload_identity(),',1)
s=s.replace('"fail_fast_engine_error": a.fail_fast_engine_error, "runs": suite.runs', '"fail_fast_engine_error": suite.fail_fast_engine_error, "runs": suite.runs',1)
s=s.replace('"--think" if a.think else "",','"--think" if a.think else "",\n                                                    f"--prefill-only --prefill-tokens {a.prefill_tokens}" if a.prefill_only else "",',1)
s=s.replace('"comparison_identity": {"workload": suite.workload_identity(),', '"comparison_identity": {"mode":"prefill-only" if a.prefill_only else "suite", "prefill_plan":suite.prefill_plan() if a.prefill_only else None, "workload": suite.workload_identity(),',1)
s=s.replace('for n, v in suite.agent_files()]','for n, v in ([] if a.prefill_only else suite.agent_files())]',1)
s=s.replace('                say("Benchmark stopped after', '''                if a.prefill_only:
                    (directory / "README.md").write_text(red(render_readme(res)),encoding="utf-8")
                say("Benchmark stopped after''',1)
needle='''            journal.state(terminal="failed" if suite.out.get("terminal_failure") else "interrupted",
                          complete=False, qualified=False, error=repr(error)[:300])'''
s=s.replace(needle,needle+'''
            if a.prefill_only:
                partial={"schema":SCHEMA,"tool":"tools/strata_bench.py","date":datetime.date.today().isoformat(),
                         "params":{"mode":"prefill-only","prefill_tokens":a.prefill_tokens,"think":True,
                                   "runs":suite.runs,"workload":suite.workload_identity(),"started_server":started},
                         "comparison_identity":{"mode":"prefill-only","prefill_plan":suite.prefill_plan(),
                                                "workload":suite.workload_identity(),"client":client_fingerprint()},
                         "elapsed_s":time.time()-t0,"machine":pre,"results":suite.out,"complete":False,
                         "qualified":False,"terminal":"interrupted","terminal_failure":repr(error)[:300]}
                target=directory/"results.tmp"
                with target.open("w",encoding="utf-8") as handle:
                    json.dump(redact_obj(partial,red),handle,indent=1,ensure_ascii=False)
                    handle.flush();os.fsync(handle.fileno())
                os.replace(target,directory/"results.json")
                (directory/"README.md").write_text(red(render_readme(partial)),encoding="utf-8")''',1)
# Explicit diagnostic completion never means optimization qualification.
s=s.replace('"complete": terminal_error is None}\n            if terminal_error', '"complete": terminal_error is None}\n            if a.prefill_only:\n                res["qualified"]=False\n            if terminal_error',1)
# Preserve legacy terminal semantics while distinguishing diagnostic transport failures.
s=s.replace('journal.state(terminal="failed" if suite.out.get("terminal_failure") else "interrupted",',
            'journal.state(terminal=("interrupted" if isinstance(error,(KeyboardInterrupt,SystemExit)) else "failed") if a.prefill_only else ("failed" if suite.out.get("terminal_failure") else "interrupted"),',1)
s=s.replace('"qualified":False,"terminal":"interrupted","terminal_failure":repr(error)[:300]}',
            '"qualified":False,"terminal":"interrupted" if isinstance(error,(KeyboardInterrupt,SystemExit)) else "failed","terminal_failure":repr(error)[:300]}',1)
# Keep no-prompt failure handling robust; existing out has empty cohort dictionaries.
files={'tools/strata_bench.py':(base,s)}
b=(ROOT/'bench.sh').read_text();files['bench.sh']=(b,b.replace('# --long includes', '# --prefill-only [--prefill-tokens 32768|65536] runs only short warmup + fresh long prompts (thinking on).\n# --long includes'))
patch=[];pins={}
for name,(old,new) in files.items():
    patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    pins[name]={'base_sha256':hashlib.sha256(old.encode()).hexdigest(),'proposed_sha256':hashlib.sha256(new.encode()).hexdigest()}
(D/'source.diff').write_text(''.join(patch));(D/'pins.json').write_text(json.dumps(pins,indent=2)+'\n')
