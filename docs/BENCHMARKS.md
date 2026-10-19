# Benchmarks on this personal fork

The client comes from Niko1221/Strata commit
`7695859aa13afb248bdb4d36f9b2e95707db079d` (v0.1.43). Importing the client
does not import or qualify that engine version. `tools/ab_engine.py`,
`tools/agent_turns_bench.py` and `tools/serve_load.py` are unchanged dependencies.
This fork extends `tools/strata_bench.py` and `bench.sh`.

With a server already running on localhost:

```sh
./bench.sh --url http://127.0.0.1:18096 --long --long-generation --think
```

The default is three repetitions. This runs the upstream short chat, 4K/16K
prompts, **agent session and 32K prompt**, then two long-output workloads. It
takes longer than the original suite. The agent sends six requests per session,
adding five public project-file excerpts and the model's actual earlier answers.
With `--think`, each agent turn allows up to 8,192 output tokens and replays
both actual answer and reasoning; a thinking-only turn remains incomplete.
Context must allow the full history and output room (59,392 for this condition).
It offers tool definitions but does not execute tools. Keep per-turn
read/reused tokens, timings and MTP counters alongside session totals. Cache
reuse depends on the server configuration; it is measured, not assumed.

The new workloads request approximately 33K input tokens and up to 16,384 output
tokens. An eligible sample needs **at least 32,768 actual input tokens and
at least 1,024 actual generated tokens including thinking**, no reused prefix,
normal SSE completion and engine decode timings. Agreement eligibility additionally
requires at least 256 offered MTP drafts; a no-MTP baseline can provide throughput
while its acceptance ratio stays undefined. **Thinking stays enabled**, as in actual use. The v1 pilot spent its
entire 1,152-token budget on thought. The v2 thinking-off proposal was withdrawn
before GPU execution at the user's correction. Workload v3 used an 8,192-token cap. Its Japanese request was interrupted
by the measurement harness CPU limit before completion. Workload v4 allows
16,384 tokens, explicitly keeps xhigh thinking with no separate thought cap,
and requires at least 512 characters of actual answer body after thinking;
this is a body-presence check, not a character-to-token conversion. The total
completion count still includes thought. If thought again consumes the cap,
increase the finite output/context budget or revise the task. Do not compare
different workload versions as an engine optimization. A complete answer at
the cap is not required by this speed test.

- `regular-code`: a long Python module with regularly formatted individual tests.
- `varied-prose`: an original Japanese story with distinct viewpoints and vocabulary.

These are hypotheses about draft predictability. The pair is confirmed only
after at least three eligible runs per workload, with regular-code acceptance
at least 60% in every run and varied-prose at most 40%. The report
retains unconfirmed, short and failed samples; three-run commands exit nonzero
if the requested pair is not confirmed. Never relabel an unconfirmed case as
strong or weak. Task correctness and MTP speedup are separate, unqualified
questions; speed attribution needs a matched no-MTP arm.

For screening the new prompts alone:

```sh
./bench.sh --url http://127.0.0.1:18096 --long-generation-only --think --runs 1
```

One run screens length/counters but cannot confirm the pair. Results contain
every request, order and hashes, actual generated count, finish reason,
offered/accepted counts and eligibility reasons. Pooled ratios use summed counts.
Visible/reasoning stream chunks are not token counts. Each request has a durable start/body hash and compact result in a verified
JSONL hash chain; interrupted requests remain pending/incomplete. Agent
statistics use complete six-turn sessions only, while all partial rows remain.
Full successful response traces are unnecessary after these facts are recorded.

Use the same pinned unmodified Niko client on both engines for existing standard
results, and the same pinned extended client for added decode tests. Distinguish
warm-server requests from independent fresh-server repetitions, and prefill
from decode. About 1K generated tokens is a starting observation window. Increase
independent repetitions or length when variation still prevents a decision.
Do not pool the original 24-token decode with this new workload.

For the selected SYCL native16 engine, Main uses
`tools/sycl/generation_engine.py prepare check`, commits the prepared result,
then uses `run check`. The controller preserves the qualified runtime safety
settings, validates parameters and records progress without a full API trace.
It allows5400 wall seconds,54000 CPU seconds,96GiB owned-session RSS and64MiB
raw output for the two16K-cap requests. A deadline or size stop is incomplete.
It verifies the official request journal and preserves completed partial rows
if the suite cannot close.

After an eligible check, `prepare clean N` and `run clean N` collect independent
processes0,1,2 with observers off; commit every plan and result before advancing.
Keep native inputs, controller, workloads and configuration frozen throughout.
After committing process2, `summarize` recovers all four original receipts and
checks matched request hashes, independent process identities and source pins.
It reports all three clean samples and spread per workload, strong/weak
acceptance separately from the70tok/s speed target, and excludes the diagnostic
check from clean statistics. These are one request per workload per process
after the standard client warmup/calibration, not repeated warm requests of the
same workload. Source/plan commits precede execution; the controller does not
build, retry an engine, adopt a default or qualify the broader model gates.

For a prefill-only screen, `bench.sh --prefill-only --prefill-tokens N` accepts
32768,65536,98304 or131072 as the minimum **actual prompt and read tokens**.
Thinking stays on and output is capped at24 tokens; this does not qualify
decode. Fixed prompt construction targets1.25 times that minimum, independently
of server responses. The client checks context room before sending, and records
actual engine counts and zero reuse rather than relabeling the construction
target as measured tokens. Minimum98304 needs known context>=123160, fitting
the current131072 setting; minimum131072 needs>=164120, so any larger-context
engine run first needs its VRAM/workspace and boundary admission check.
These added sizes are prepared for a longer-input screening comparison; no
speed or model-context qualification is established by the client change.

```sh
python3 tools/test_strata_bench.py
python3 tools/test_benchmark_evidence.py
```

These checks exercise SSE usage including reasoning, invalid/short responses,
large input construction, alternating workload order and replay of real agent
answers. They require no GPU or model. Supervised measurements still follow
[ARTIFACT_RETENTION.md](ARTIFACT_RETENTION.md), including process ownership,
finite budgets, source/config identity and fresh GPU fault checks.

Component comparisons follow the same evidence rules. Time the production
entrypoint and its dispatch/fallback in both arms unless the committed plan
explicitly studies a different boundary. When dispatch can vary, check the
actual outcomes in the timing batches rather than inferring them from startup
flags or an earlier preflight. Keep reference calculations, resets and validation
outside the timed region unless they are the subject of the measurement.

Commit batch sizes, repetition order, timing eligibility and overhead checks
before collection. A declared minimum batch duration screens timing granularity;
it is not an accuracy or noise guarantee. A batch that fails the screen cannot
qualify a comparison. Changing its repetition plan needs a new committed plan
and new samples, with the original result preserved.
