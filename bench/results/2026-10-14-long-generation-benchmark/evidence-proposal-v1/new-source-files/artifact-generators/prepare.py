from pathlib import Path
import difflib,hashlib,json
P=Path('/home/yayoi/orca/workspaces/Strata/sycl')
D=P/'bench/results/2026-10-14-long-generation-benchmark/evidence-proposal-v1'
original={n:(P/n).read_text() for n in ['tools/strata_bench.py','tools/long_generation_bench.py','bench/results/2026-10-14-long-generation-benchmark/check_extended.py']}
files=dict(original)
s=files['tools/long_generation_bench.py']
s=s.replace('    offered, accepted = sample.get(', '    speed_reasons = list(reasons)\n    offered, accepted = sample.get(')
s=s.replace('return {"eligible": not reasons,', 'return {"eligible": not reasons, "agreement_eligible": not reasons,\n            "speed_eligible": not speed_reasons, "speed_ineligible_reasons": speed_reasons,')
files['tools/long_generation_bench.py']=s
s=files['tools/strata_bench.py']
s=s.replace('AGENT_MAX_TOKENS = 64','AGENT_MAX_TOKENS = 64\nAGENT_THINK_MAX_TOKENS = 2048')
s=s.replace('    if not think:\n        body["reasoning_effort"]', '    if think:\n        body["chat_template_kwargs"] = {"enable_thinking": True, "preserve_thinking": True}\n        body["reasoning_budget_tokens"] = 0\n    if not think:\n        body["reasoning_effort"]')
anchor='# ---------------------------------------------------------------------------------------------------- the suite'
helper='''def agent_turn_complete(d):
    return (d.get("ok") and d.get("stream_complete") and d.get("generated_tokens_source") == "usage"
            and type(d.get("generated_tokens")) is int and d["generated_tokens"] > 0
            and type(d.get("prompt_tokens")) is int and d["prompt_tokens"] > 0
            and d.get("output_chars", 0) > 0)


def agent_complete(turns):
    return len(turns) == AGENT_TURNS + 1 and all(agent_turn_complete(d) for d in turns)


class RequestJournal:
    """Compact durable rows. An unfinished start/state is incomplete evidence even after SIGKILL."""
    def __init__(self, directory, red):
        self.directory, self.red = directory, red
        self.index, self.previous = 0, None
        self.lock = threading.Lock()
        self.state(complete=False, terminal="running", pending=None)

    def state(self, **values):
        data = {"complete": False, "qualified": False, "last_index": self.index - 1,
                "last_sha256": self.previous, **values}
        tmp = self.directory / "journal-state.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(redact_obj(data, self.red), f, ensure_ascii=False)
            f.write("\\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.directory / "journal-state.json")
        # Persist the rename and journal's initial directory entry on Linux.
        if os.name == "posix":
            fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def append(self, kind, phase, run=None, turn=None, **values):
        with self.lock:
            row = redact_obj({"index": self.index, "previous_sha256": self.previous,
                "kind": kind, "phase": phase, "run": run, "turn": turn, **values}, self.red)
            encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            row["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
            with (self.directory / "requests.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\\n")
                f.flush()
                os.fsync(f.fileno())
            self.previous = row["sha256"]
            self.index += 1
            self.state(terminal="running", pending={"phase": phase, "run": run, "turn": turn}
                       if kind == "start" else None)


def client_fingerprint():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
            ("tools/strata_bench.py", "tools/long_generation_bench.py", "tools/ab_engine.py",
             "tools/agent_turns_bench.py", "tools/serve_load.py")}


'''
s=s.replace(anchor,helper+anchor)
s=s.replace('want_long_generation=False, long_generation_only=False):','want_long_generation=False, long_generation_only=False, journal=None):')
s=s.replace('        self.nonce = str(int(time.time()))','        self.journal = journal\n        self.nonce = str(int(time.time()))')
anchor='    def run_long_generation(self):'
helper='''    def request(self, phase, run, turn, messages, max_tokens, think=False, **kwargs):
        if self.journal:
            self.journal.append("start", phase, run, turn)
        try:
            raw = chat(self.base, self.key, messages, max_tokens, think, **kwargs)
        except BaseException as error:
            if self.journal:
                self.journal.append("interrupted", phase, run, turn, error=repr(error)[:300])
            raise
        row = digest(raw)
        row.update(output_chars=len(raw["text"]), reasoning_chars=len(raw["reasoning_text"]),
                   output_sha256=hashlib.sha256(raw["text"].encode()).hexdigest(),
                   reasoning_sha256=hashlib.sha256(raw["reasoning_text"].encode()).hexdigest())
        if phase.startswith("long-generation/"):
            row.update(long_generation.sample_eligibility(raw, row))
        if self.journal:
            self.journal.append("result", phase, run, turn, sample=row)
        return raw

'''
s=s.replace(anchor,helper+anchor)
s=s.replace('raw = chat(self.base, self.key, messages, lg.MAX_OUTPUT_TOKENS, lg.THINKING,','raw = self.request("long-generation/" + name, run, None, messages, lg.MAX_OUTPUT_TOKENS, lg.THINKING,')
s=s.replace('            offered = sum(r["drafts_offered"] for r in good)','            speed = [r for r in case["requests"] if r["speed_eligible"]]\n            offered = sum(r["drafts_offered"] for r in good)')
s=s.replace('case.update(eligible_runs=len(good), decode_tps=stat(r["decode_tps"] for r in good),','case.update(eligible_runs=len(good), speed_eligible_runs=len(speed),\n                        decode_tps=stat(r["decode_tps"] for r in speed),')
s=s.replace('r = digest(chat(self.base, self.key, [{"role": "user", "content": f"[{self.tag(\'cal\', words)}]\\n" + long_prompt(words)}], 1))','r = digest(self.request("calibration", words, None, [{"role": "user", "content": f"[{self.tag(\'cal\', words)}]\\n" + long_prompt(words)}], 1))')
s=s.replace('chat(self.base, self.key, [{"role": "user", "content": PROMPTS[0]}], 24, self.think)','self.request("warm-up/chat", None, None, [{"role": "user", "content": PROMPTS[0]}], 24, self.think)')
s=s.replace('chat(self.base, self.key, self.prompt_for(1024, -1), 8, self.think)','self.request("warm-up/prompt", None, None, self.prompt_for(1024, -1), 8, self.think)')
s=s.replace('digest(chat(self.base, self.key, [{"role": "user", "content": f"[{self.tag(run, i, \'c\')}] {p}"}],','digest(self.request("chat", run, i, [{"role": "user", "content": f"[{self.tag(run, i, \'c\')}] {p}"}],')
s=s.replace('digest(chat(self.base, self.key, self.prompt_for(tokens, run), LONG_MAX_TOKENS, self.think))','digest(self.request("prompt/" + label, run, None, self.prompt_for(tokens, run), LONG_MAX_TOKENS, self.think))')
s=s.replace('r = chat(self.base, self.key, msgs, AGENT_MAX_TOKENS, self.think, tools=TOOLS)','r = self.request("agent", run, t, msgs,\n                             AGENT_THINK_MAX_TOKENS if self.think else AGENT_MAX_TOKENS,\n                             self.think, tools=TOOLS)')
s=s.replace('            d.update(run=run, turn=t)','            d.update(run=run, turn=t, output_chars=len(r["text"]), reasoning_chars=len(r["reasoning_text"]),\n                     output_sha256=hashlib.sha256(r["text"].encode()).hexdigest(),\n                     reasoning_sha256=hashlib.sha256(r["reasoning_text"].encode()).hexdigest())')
s=s.replace('            if not d["ok"]:\n                self.fail("agent", d)','            if not agent_turn_complete(d):\n                d["err"] = d.get("err") or "incomplete turn or missing answer body/usage"\n                self.fail("agent", d)')
s=s.replace('msgs.append({"role": "assistant", "content": r["text"] or "Understood."})','msgs.append({"role": "assistant", "content": r["text"], "reasoning_content": r["reasoning_text"]})')
s=s.replace('        follow = [[d for d in s if d["ok"] and d["turn"] > 0] for s in sessions]','        complete_sessions = [s for s in sessions if agent_complete(s)]\n        follow = [[d for d in s if d["turn"] > 0] for s in complete_sessions]')
s=s.replace('        for s in sessions:\n            s = [d for d in s if d["ok"]]','        for s in complete_sessions:')
s=s.replace('for s in sessions if s and s[0]["ok"]','for s in complete_sessions')
s=s.replace('"complete": len(turns) == AGENT_TURNS + 1 and all(d["ok"] and d["stream_complete"]\n                    and d["generated_tokens_source"] == "usage" for d in turns),','"complete": agent_complete(turns),')
s=s.replace('        self.out["agent"]["session_samples"] = session_samples','''        self.out["agent"].update(session_samples=session_samples,
            complete_sessions=len(complete_sessions),
            partial_sessions=sum(bool(s) and not agent_complete(s) for s in sessions),
            failed_sessions=sum(not s or any(not agent_turn_complete(d) for d in s) for s in sessions),
            thinking=self.think, max_tokens=AGENT_THINK_MAX_TOKENS if self.think else AGENT_MAX_TOKENS,
            scope="Six actual sequential turns; body and reasoning replayed; complete sessions only in summaries")''')
s=s.replace('n = len([s for s in a["sessions"] if len(s) > 1])','n = a.get("complete_sessions", sum(x.get("complete", False) for x in a.get("session_samples", [])))')
s=s.replace('f"up to {AGENT_MAX_TOKENS}", n','f"up to {a.get(\'max_tokens\', AGENT_MAX_TOKENS)}", n')
s=s.replace('              f"Cache hit over the follow-up turns:', '              f"Complete sessions: {ag.get(\'complete_sessions\', 0)}; partial: {ag.get(\'partial_sessions\', 0)}; "\n              f"failed: {ag.get(\'failed_sessions\', 0)}. Thinking: {ag.get(\'thinking\')}; "\n              f"output cap: {ag.get(\'max_tokens\', AGENT_MAX_TOKENS)}. Statistics use complete six-turn sessions. "\n              f"Cache hit over the follow-up turns:')
s=s.replace('def write_results(res: dict, parent: Path, red, log_lines: list[str]) -> Path:\n    d = unique_dir(parent, results_dir_name(res["machine"], res["date"]))\n    d.mkdir(parents=True)','def write_results(res: dict, parent: Path, red, log_lines: list[str], directory=None) -> Path:\n    d = directory or unique_dir(parent, results_dir_name(res["machine"], res["date"]))\n    d.mkdir(parents=True, exist_ok=directory is not None)')
s=s.replace('        suite = Suite(base, key, pre.get("context"),','''        parent = Path(a.out) if a.out else ROOT / "bench" / "results"
        directory = unique_dir(parent, results_dir_name(pre, datetime.date.today().isoformat()))
        directory.mkdir(parents=True)
        journal = RequestJournal(directory, red)
        suite = Suite(base, key, pre.get("context"),''')
s=s.replace('long_generation_only=a.long_generation_only)\n        slots','long_generation_only=a.long_generation_only, journal=journal)\n        slots')
s=s.replace('        out = suite.run()','''        try:
            out = suite.run()
        except BaseException as error:
            journal.state(terminal="interrupted", complete=False, error=repr(error)[:300])
            raise''')
s=s.replace('"elapsed_s": time.time() - t0, "machine": machine, "results": out}', '''"elapsed_s": time.time() - t0, "machine": machine, "results": out,
               "comparison_identity": {"client": client_fingerprint(), "thinking": a.think,
                   "reasoning_budget_tokens": 0, "temperature": 0, "model": machine.get("model"),
                   "quant": machine.get("quant"), "context": machine.get("context"),
                   "long_generation_version": long_generation.WORKLOAD_VERSION,
                   "long_generation_thinking": long_generation.THINKING,
                   "long_generation_instructions": long_generation.WORKLOADS,
                   "calibration": [suite.tokens_per_word, suite.overhead],
                   "agent_files": [(n, hashlib.sha256(v.encode()).hexdigest()) for n, v in suite.agent_files()],
                   "agent_max_tokens": AGENT_THINK_MAX_TOKENS if a.think else AGENT_MAX_TOKENS},
               "complete": True}''')
s=s.replace('        d = write_results(res, parent, red, log_excerpt(log_path) if log_path else [])','        d = write_results(res, parent, red, log_excerpt(log_path) if log_path else [], directory=directory)\n        journal.state(complete=True, terminal="complete", qualified=False)')
a=s.index('    different_workload = bool(go and gn and any(');b=s.index('    rows = [f',a)
s=s[:a]+'''    io, inn = old.get("comparison_identity"), new.get("comparison_identity")
    differences = [k for k in set(io or {}) | set(inn or {}) if (io or {}).get(k) != (inn or {}).get(k)]
    compatible = bool(io and inn and not differences and old.get("complete") and new.get("complete"))
    if not io or not inn:
        differences.append("missing client/workload identity")
    if not old.get("complete") or not new.get("complete"):
        differences.append("incomplete result")
    different_workload = not compatible
''' +s[b:]
s=s.replace('if x and y is not None else ""','if compatible and x and y is not None else ""')
s=s.replace('notes.append("  long-generation workload/settings differ: those rates are not compared")','notes.append("  not comparable: " + ", ".join(sorted(differences)) + "; descriptive values only")')
files['tools/strata_bench.py']=s
s=files['bench/results/2026-10-14-long-generation-benchmark/check_extended.py']
s=s.replace("['--long', '--long-generation']","['--long', '--long-generation', '--think']")
s=s.replace("            paths=list((O/'stock-results').glob('*/results.json'))",'''            journals = list((O/'stock-results').glob('*/requests.jsonl'))
            record['request_journals'] = [{"path": str(p.relative_to(O)), "sha256": sha(p),
                "bytes": p.stat().st_size,
                "state": json.loads((p.parent/'journal-state.json').read_text())
                    if (p.parent/'journal-state.json').exists() else {"complete": False}}
                for p in journals]
            paths=list((O/'stock-results').glob('*/results.json'))''')
files['bench/results/2026-10-14-long-generation-benchmark/check_extended.py']=s
patch=''.join(''.join(difflib.unified_diff(original[n].splitlines(True),v.splitlines(True),fromfile='a/'+n,tofile='b/'+n)) for n,v in files.items())
(D/'proposal.patch').write_text(patch)
(D/'identity.json').write_text(json.dumps({'base_revision':'516c441f','native_binary_revision':'68da0262','applied':False,'tests_executed':False,'bases':{n:hashlib.sha256(v.encode()).hexdigest() for n,v in original.items()},'proposed_sources':{n:hashlib.sha256(v.encode()).hexdigest() for n,v in files.items()},'proposal_sha256':hashlib.sha256(patch.encode()).hexdigest()},indent=2)+'\n')
for n,v in files.items():
    target=D/'new-source-files'/n
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(v)
