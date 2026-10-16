"""Generate a review diff only. Does not modify product source or execute tests."""
from pathlib import Path
import difflib
import hashlib
import json
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
p = ROOT / 'tools/strata_bench.py'
assert hashlib.sha256(p.read_bytes()).hexdigest() == '3cfb1ed6e9851d41515d8e2c1dabbd18e9805b5aa6ed9acb9f8f7d9783d12f69', 'original source changed'
old = p.read_text()
new = old

def replace(a, b):
    global new
    assert new.count(a) == 1, a
    new = new.replace(a, b)

replace('class Suite:\n', 'class EngineRequestError(RuntimeError):\n    """A returned failed digest stopped an explicitly opted-in benchmark."""\n\n\nclass Suite:\n')
replace('long_generation_only=False, journal=None):', 'long_generation_only=False, journal=None, fail_fast_engine_error=False):')
replace('        self.journal = journal\n', '        self.journal = journal\n        self.fail_fast_engine_error = fail_fast_engine_error\n')
replace('        request_index = None\n', '''        # Defensive guard also covers callers that catch the terminal exception.
        if self.fail_fast_engine_error and self.out.get("terminal_failure"):
            raise EngineRequestError(self.out["terminal_failure"]["error"])
        request_index = None
''')
replace('        return raw\n\n    def run_long_generation', '''        if self.fail_fast_engine_error:
            sample = {"phase": phase, "run": run, "turn": turn,
                      "request_index": request_index, "sample": row}
            self.out.setdefault("request_samples", []).append(sample)
            if row.get("err") or not row.get("ok"):
                reason = row.get("err") or "no token came back"
                failure = {**sample, "classification": "engine_request_error", "error": reason}
                self.out["terminal_failure"] = failure
                self.fail(phase, row)
                # The sole result row has already been fsynced; never add interrupted.
                raise EngineRequestError(reason)
        return raw

    def run_long_generation''')
replace('    def run_clients(self, n: int):\n', '''    def run_clients(self, n: int):
        if self.fail_fast_engine_error:
            raise ValueError("--fail-fast-engine-error cannot be combined with --clients")
''')
replace('    ap.add_argument("--quiet", action="store_true")', '''    ap.add_argument("--fail-fast-engine-error", action="store_true",
                    help="stop after the first returned failed request digest (default off; incompatible with --clients)")
    ap.add_argument("--quiet", action="store_true")''')
replace('    a = ap.parse_args(argv)\n', '''    a = ap.parse_args(argv)
    if a.fail_fast_engine_error and a.clients:
        ap.error("--fail-fast-engine-error cannot be combined with --clients")
''')
replace('workload_version=long_generation.WORKLOAD_VERSION, requested_runs=max(1, a.runs))', 'workload_version=long_generation.WORKLOAD_VERSION, requested_runs=max(1, a.runs),\n                       fail_fast_engine_error=a.fail_fast_engine_error)')
replace('long_generation_only=a.long_generation_only, journal=journal)', 'long_generation_only=a.long_generation_only, journal=journal,\n                      fail_fast_engine_error=a.fail_fast_engine_error)')
replace('''            out = suite.run()
            machine = collect_machine(base, key, cfg)             # again: VRAM and the cache after the work
''', '''            terminal_error = None
            try:
                out = suite.run()
            except EngineRequestError as error:
                terminal_error = error
                out = suite.out
            # Do not ask a failed server for post-run metadata.
            machine = pre if terminal_error else collect_machine(base, key, cfg)
''')
replace('"params": {"runs": suite.runs,', '"params": {"fail_fast_engine_error": a.fail_fast_engine_error, "runs": suite.runs,')
replace('"--think" if a.think else "", f"--runs', '"--think" if a.think else "",\n                                                    "--fail-fast-engine-error" if a.fail_fast_engine_error else "", f"--runs')
replace('                   "complete": True}\n', '''                   "complete": terminal_error is None}
            if terminal_error is not None:
                res.update(terminal="failed", qualified=False, terminal_failure=out["terminal_failure"])
                # Save JSON before rendering: a partial suite need not support every report table.
                partial = directory / "results.tmp"
                with partial.open("w", encoding="utf-8") as handle:
                    json.dump(redact_obj(res, red), handle, indent=1, ensure_ascii=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(partial, directory / "results.json")
                journal.state(terminal="failed", complete=False, qualified=False,
                              error=str(terminal_error)[:300])
                say("Benchmark stopped after a failed request; partial results: " + str(directory / "results.json"))
                return 1
''')
# Preserve failed terminal state even if output notification itself raises.
replace('''            journal.state(terminal="interrupted", complete=False, error=repr(error)[:300])
            raise
''', '''            journal.state(terminal="failed" if suite.out.get("terminal_failure") else "interrupted",
                          complete=False, qualified=False, error=repr(error)[:300])
            raise
''')
diff = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='a/tools/strata_bench.py', tofile='b/tools/strata_bench.py'))
(HERE/'source.diff').write_text(diff)
manifest = {'status': 'PROPOSAL_ONLY', 'tests': 'NOT_RUN', 'original_files': {str(p.relative_to(ROOT)): hashlib.sha256(old.encode()).hexdigest()}, 'proposed_files': {'tools/strata_bench.py': hashlib.sha256(new.encode()).hexdigest()}}
(HERE/'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
