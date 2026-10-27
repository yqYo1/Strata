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

For a new comparison needing more output room, add
`--long-generation-max-tokens 32768` (or65536) on both engines. The default
remains16384. The budget is included in the wire request, request journal,
parameters and comparison identity; different budgets are not comparable.
A larger budget requires known context capacity, and eligibility checks actual
input plus the requested budget and256 reserved context slots. Thinking remains
enabled; a thinking-only output still fails the answer-body gate. Preserve an
earlier capped failure and use a fresh committed plan, rather than replacing it.

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

For a throughput comparison that includes a no-draft baseline, add
`--long-generation-speed-only` to `--long-generation-only` on **both** engines.
The input bodies, thinking and output budgets stay identical. Normal complete
generation, at least 32K input/1K output, answer body, engine clocks and valid
draft counters remain required; zero offered/accepted drafts are allowed.
Agreement eligibility and classification remain separate and unqualified when
there are no drafts. This mode does not turn short or incomplete output into
a usable speed sample. Its selection is recorded in parameters and identity.

For a focused diagnostic, add `--long-generation-workload regular-code` or
`--long-generation-workload varied-prose` to `--long-generation-only`. It keeps
the same warmup, prompt, thinking and output budget, records the selected subset
and leaves the other case unrun. Even three subset runs do not qualify the MTP
pair. The default still executes both workloads.

Use the same pinned unmodified Niko client on both engines for existing standard
results, and the same pinned extended client for added decode tests. Distinguish
warm-server requests from independent fresh-server repetitions, and prefill
from decode. About 1K generated tokens is a starting observation window. Increase
independent repetitions or length when variation still prevents a decision.
Do not pool the original 24-token decode with this new workload.

One input length does not establish a throughput curve. Record total prefill
milliseconds against actual newly read tokens, alongside pp and client TTFT,
at several short and long lengths. Compare simple approximate formulas using
residuals, held-out lengths and neighboring slopes. `T(n) = a + b*n` is only
one hypothesis; keep nonlinear or chunk-boundary behavior
visible. Separate execution routes before assigning a physical meaning to `a`;
short prefill through decode placement and long layer-major prefill can have
different fixed costs. First-use, warmed requests and independent processes are
separate samples. Repeat the sweep after relevant placement, chunking, scheduling
or kernel changes; local component screens need not repeat the entire curve.
Decode depends on starting context, actual generation length and MTP acceptance;
record these and characterize more than one context/length before generalizing.

TTFT is measured from immediately before sending the client HTTP request to the
first nonempty `reasoning_content` or `content` delta. It includes the first
thinking output, rather than waiting for thinking to finish. Keepalives, empty
deltas and role-only events do not stop the clock. This includes request/queue,
prompt processing and first-output delivery time; model startup is reported
separately. It is already collected by the benchmark and must accompany speed
results, rather than being inferred from pp.

For the length characterization, with a qualified server configured for at least
123160 context slots:

```sh
./bench.sh --url http://127.0.0.1:18096 --prefill-scaling --runs 3
```

This new mode requests minima1024,4096,16384,32768,65536,98304; construction
targets are1.25 times those values and the actual engine token counts are the
fit inputs. It retains one maximum-length first request separately, then runs
three same-process sweeps in ascending, descending and rotated order. Retain
the first visit to each length separately from its later visits: a maximum-length
request primes the process but does not prove that every shape is warmed. Thinking is
enabled and output is capped at24. Each request must finish normally with zero
reuse, valid engine timings and actual context headroom. A failure stops the
sweep and remains a failed sample. Each length keeps all individual times and
TTFTs. Affine, quadratic and linear-plus-`n log n` hypotheses report residuals
and leave-one-length-out prediction errors; none is selected automatically.
Prefer a simple adequate approximation, compare its errors with observed spread,
and split by execution route or boundaries when appropriate. These fits and
adjacent slopes do not establish a
physical fixed cost, linearity, independent-process repeatability or hardware
saturation. A supervised model run still needs a committed plan, runtime and
binary admission; adding this client mode is not a completed measurement.

The owned128/borrowed1280 capacity admission and comparison are closed. Their
[compact result](../bench/results/2026-10-15-pr6-recovery-audit/cache-comparison-current.json)
keeps all six independent samples, common evidence and arm-specific route
counts. Restore controller source`cba7b0a7c` and the recorded plan revisions
for its exact recipe; closed roots must never be retried. Merge common process
evidence and request fields with each sample, and attach its arm's route,
to replay the unchanged `summarize` decision. Main verified that reconstruction
matches the original result exactly. The+3.7041% decode geomean misses the
prospective>5% screen; prefill separately passes nonregression. Further1256-slot
capacity work requires a fresh committed plan. Weak-MTP, agent-session, quality
and physical262144/session/RESTORE qualification remain separate.

The observed IQ2 pilot is closed at commit`bedfdf312`; its launcher and full
model outcome remain recoverable there. The current `iq2_pair_model.py prepare`
creates a six-process comparison, followed by one `run` invocation per process.
Commit its generated plan and each closed outcome before advancing. Its
[current plan/result](../bench/results/2026-10-15-pr6-recovery-audit/iq2-pair-model-current.json)
pins that original pilot, its same-build safety/bank gates and the actual CPU
baseline. The order is PAIR0/1/1/0/0/1, with four official warmups and both
thinking-on long-generation workloads in each process. Source, native binary,
model/profile, placement, runtime and observer settings stay fixed; only the
pre-exec pair flag changes. Per-request counters prove dispatch and target-work
conservation. Judge prefill and decode separately using actual tokens and engine
milliseconds; the frozen plan records repetition, performance and MTP-class
gates. Validation, stage trace, census and clocks are charged observers. A pass
does not qualify clean speed, saturation or adoption. Never retry a closed root.

`generation_engine.py` is now a shared admission/evidence library, with no
launcher CLI. Restore closed long-generation launchers and their original
plans from Git when reproducing them; current compact decisions are in
[the maintained prefill results](SYCL_NATIVE16_PREFILL.md).

For later decode attribution, use a separate observer process after the frozen
cohort. `STRATA_DECODE_TIMING` reports wall-clock regions: its CPU label includes
pool completion, and per-layer host time includes callbacks and GPU submission.
These nested counters are not thread CPU time or measured RAM/PCIe traffic.
The engine's decode clock also stops before the final `wait_commit`; record
successful completion separately. Do not enable `STRATA_VERIFY_PROFILE` on the
current SYCL backend: verifier capture calls `gpu_stamp`, which explicitly throws
because device global-timer stamps are unsupported. GPU attribution needs a
separately qualified profiling-event path outside capture. Leave both observer
variables absent in clean processes; setting either to `0` still enables it.

The earlier whole-decode CPU-clock observer and its launchers were retired after
their
failed overhead/nonregression gates; they are historical evidence, not current
benchmark commands. Their source remains recoverable at
`d50d31347cc8505a40181db8308735bfa0058427`. The maintained
[resource reader](../bench/sycl/client_resources.py), `resource_engine.py` and
`cpu_capacity.py` retain the separate resource-observation and CPU-capacity
methods. Their closed roots cannot be rerun; a new experiment needs its own
committed plan and admission. Scheduled CPU includes stalls and polling, and
GPU busy counters do not measure arithmetic efficiency, RAM bandwidth or
physical PCIe traffic. Keep those unresolved quantities explicit.

The closed CPU-service comparison separately measured the
decode-pool leaf observer's cost:original6e40,observer-compiled but disabled,and
the same observer binary enabled. Its frozen nine-process order uses the
regular-code subset,thinking on,and the unchanged16384-token output budget.
Each process runs the four official warmups and requires one eligible32K-input/1K-output
request. Source,model,placement,runtime and common observers remain fixed.
All nine processes closed normally, but the overall overhead gate failed:
one off/original prefill ratio was0.857704, below0.95. The independent decode
screens passed; they do not override that outcome. Do not rerun the closed
controller or use this as qualified attribution, clean speed or inference
adoption. The maintained prefill decisions and `cpu-service-current.json`
retain its individual samples, original criteria and remaining gates.

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
