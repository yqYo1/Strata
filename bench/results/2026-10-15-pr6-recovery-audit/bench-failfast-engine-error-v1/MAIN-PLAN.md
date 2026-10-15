# Benchmark fail-fast proposal

PROPOSAL ONLY. All tests NOT_RUN. Product files unchanged. No build, GPU, server,
process lifecycle action, branch or Git mutation performed by this proposal author.
The artifact generator ran using python3; the unavailable `python` alias initially
returned command-not-found. That was not a test execution.

Review `source.diff` and `test_strata_bench_failfast.py`. The original source hash
is pinned in source-manifest.json and the generator. `apply_proposal.py` verifies
all original pins and refuses to overwrite an existing test before applying.
No full product source overlay is stored.

Scope: default-OFF `--fail-fast-engine-error`. Returned `digest.err` or false
`digest.ok` causes the first fatal result to stop the suite, after the single
journal result append/fsync. The exception is outside the chat BaseException
handler. A defensive terminal guard prevents another request even if a caller
catches it. `_interrupted` retains precedence and the original exception/terminal
row behavior. This option does not assert the root cause of an incomplete reply:
`engine_request_error` is an operational failed-digest category.

The opted-in suite retains every individual compact request sample, including
calibration and the failed response. This covers scenario-local arrays that have
not yet been assigned to suite.out when a failure aborts the phase. Partial
results.json uses the same metadata/comparison-identity construction, pre-run
machine metadata (no post-failure HTTP), original calibration state, complete:false,
terminal:failed, qualified:false and terminal_failure. The atomic JSON rename is
followed by journal-state's durable directory sync. The benchmark returns 1.
Default successful flow/reporting and normal server recovery are unchanged.

Concurrent --clients is explicitly rejected with the new option before server
access. Requests already in flight cannot obey a strict no-follow-up guarantee;
changing concurrency into serial work would change the benchmark's identity.
Default concurrency is unchanged. Main should review whether this explicit scope
restriction is acceptable before applying. This proposal handles the actual
serial formal run (no --clients), including calibration and 32K prompt failure.

The authoritative chronology is prompt-direct-bench-full-0-failure.ja.md:
request75/77 were the two 32K prompt successes, 79 the third 32K failure, and 81
post-abort automatic recovery long generation. R614's two-long-generation and
separate-attempt interpretations are rejected. No existing receipt is rewritten.
This proposal changes neither server recovery nor wrapper lifecycle collection.

Main-only review/application commands from repository root:

```sh
sha256sum tools/strata_bench.py
cat bench/results/2026-10-15-pr6-recovery-audit/bench-failfast-engine-error-v1/source-manifest.json
cat bench/results/2026-10-15-pr6-recovery-audit/bench-failfast-engine-error-v1/source.diff
python3 bench/results/2026-10-15-pr6-recovery-audit/bench-failfast-engine-error-v1/apply_proposal.py "$PWD"
```

Main-only serialized host checks, after source review/application:

```sh
python3 -m unittest discover -s tools -p 'test_strata_bench_failfast.py'
python3 -m unittest discover -s tools -p 'test_strata_bench.py'
python3 -m unittest discover -s tools -p 'test_benchmark_evidence.py'
```

Tests NOT_RUN: fake HTTP main integration fails during first calibration and
third 32K request; verifies exact chat/start/result counts, no interrupted duplicate,
valid hash chain with empty pending, failed/incomplete state, partial individual
samples, comparison identity and option in command. Additional cases preserve
default continuation, longgen semantic ineligibility, KeyboardInterrupt/SystemExit/
connection exceptions including returned `_interrupted`, false ok without err,
caller catching a terminal failure, and client-mode rejection. Existing host
regression suites are retained unchanged. No native rebuild is needed. Main owns
execution, result recording, final integration and Git work.

Retention: compact diff, test, pinned apply/generator scripts and review plan are
current consumers. No raw API text, tensor data, trace duplication or cleanup.
Review these proposal artifacts after Main accepts or rejects the source.
