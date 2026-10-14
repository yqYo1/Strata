# Benchmarks on this personal fork

The client comes from Niko1221/Strata commit
`7695859aa13afb248bdb4d36f9b2e95707db079d` (v0.1.43). Importing the client
does not import or qualify that engine version. `tools/ab_engine.py`,
`tools/agent_turns_bench.py` and `tools/serve_load.py` are unchanged dependencies.
This fork extends `tools/strata_bench.py` and `bench.sh`.

With a server already running on localhost:

```sh
./bench.sh --url http://127.0.0.1:18096 --long --long-generation
```

The default is three repetitions. This runs the upstream short chat, 4K/16K
prompts, **agent session and 32K prompt**, then two long-output workloads. It
takes longer than the original suite. The agent sends six requests per session,
adding five public project-file excerpts and the model's actual earlier answers.
It offers tool definitions but does not execute tools. Keep per-turn
read/reused tokens, timings and MTP counters alongside session totals. Cache
reuse depends on the server configuration; it is measured, not assumed.

The new workloads request approximately 33K input tokens and 1,152 output
tokens. An eligible sample needs **at least 32,768 actual input tokens and
1,024..1,152 actual generated tokens including thinking**, no reused prefix,
normal SSE completion, engine decode timings and at least 256 offered MTP
drafts. The count includes thinking if present. **Workload v2 disables thinking
for both cases**, so it measures the requested code/prose directly. The v1
thinking-on pilot used its whole output budget on thought and failed to separate
MTP regimes. Keep its result; do not compare different workload versions as an
engine optimization. This suite does not require a complete answer at the cap.

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
./bench.sh --url http://127.0.0.1:18096 --long-generation-only --runs 1
```

One run screens length/counters but cannot confirm the pair. Results contain
every request, order and hashes, actual generated count, finish reason,
offered/accepted counts and eligibility reasons. Pooled ratios use summed counts.
Visible/reasoning stream chunks are not token counts. Full successful response
traces are unnecessary after these facts are recorded.

Use the same pinned unmodified Niko client on both engines for existing standard
results, and the same pinned extended client for added decode tests. Distinguish
warm-server requests from independent fresh-server repetitions, and prefill
from decode. About 1K generated tokens is a starting observation window. Increase
independent repetitions or length when variation still prevents a decision.
Do not pool the original 24-token decode with this new workload.

```sh
python3 tools/test_strata_bench.py
```

These checks exercise SSE usage including reasoning, invalid/short responses,
large input construction, alternating workload order and replay of real agent
answers. They require no GPU or model. Supervised measurements still follow
[ARTIFACT_RETENTION.md](ARTIFACT_RETENTION.md), including process ownership,
finite budgets, source/config identity and fresh GPU fault checks.
