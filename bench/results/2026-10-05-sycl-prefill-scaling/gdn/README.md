# Arc B570 GDN prompt recurrence

These runs use the same Arc B570 10 GB, Ryzen 5 5600X, 128 GB installed
system RAM and compute runtime 26.35.39758.10 as the parent study.
The compiler is Intel oneAPI 2026.1. The recurrence state, FP32 output
and normalized FP16 output are compared in full with the original exported
SYCL pipeline. The reference is a GPU implementation, rather than an
independent CPU oracle.

## Register and subgroup study

The [probe source](register-study/probe.cpp), [raw output](register-study/probe.log)
and [summary](register-study/summary.json) record 30 cases: six input lengths
and five candidates. All state and output bits match, with finite outputs.
The selected candidate keeps the original key-head kernel body and requests
subgroup size 16 and 256 GRFs. Neither setting changes the floating point
operation order. The device reports zero spill bytes for that candidate.

| Candidate | Subgroup | GRFs | Spill bytes | Original 8,192 tokens, ms | Candidate, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Paired row groups | 32 | default | 4,864 | 15.477 | 35.493 |
| Paired row groups | 32 | 256 | 0 | 15.472 | 14.069 |
| Original key-head body | 32 | 256 | 5,760 | 15.475 | 21.443 |
| Original key-head body | 16 | default | 3,520 | 15.476 | 23.989 |
| Original key-head body | 16 | 256 | 0 | 15.481 | 12.272 |

For the last candidate, the synthetic 4,096-token time changes from
7.285 to 6.165 ms. Each time is the median of three warmed rounds of five
calls, alternating the original and candidate order, with queue waits
around the round. Compilation and other GPU runs do not compete with
these timed calls. These are isolated recurrence measurements, so they do
not establish a model-wide gain.

The candidate is slower for one and 17 tokens. The engine's new
`STRATA_GDN_KEYHEAD_TUNED=1` selector applies only to the pipelined path
with at least 256 tokens. It defaults to off. Explicit serial and
non-pipelined selectors still use their existing paths.

## Integrated parity

The new `gdn_variant_parity` CTest checks input lengths 1, 17, 255, 256,
257, 1,024, 4,096 and 8,192. It checks every state and output bit, finite
values and guards around all three writable arrays. It also checks that
the variant refuses the short inputs and that the fallback matches.
The [JIT output](jit-parity.log) passes all eight cases and reports zero
spill bytes. [All 29 JIT CTests](jit-ctest.log) pass without skips.

## Other candidates

The [local-fence and subgroup-shuffle probe](barriers/probe.cpp) and
[output](barriers/probe.log) preserve all bits in 12 cases. At 8,192 tokens,
the shuffle candidate takes about 34.22 ms versus 15.48 ms for the original.
The [swizzled SLM probe](swizzled-slm/probe.cpp) and
[output](swizzled-slm/probe.log) also preserve the bits but take about
52.15 ms versus 15.47 ms. These candidates are not used by the engine.

## Initial real-model profile

The [original](resident-initial/original/run.json) and
[tuned](resident-initial/tuned/run.json) runs use one immutable JIT binary,
SHA256 `61f08da5b70da0160ae42cc0776cb91c34e57213cdc731e033178aa16b947a4a`,
with attention batch 128, attention layout 3 and the tuned prompt selector.
Context and prefill chunk are 8,192 tokens; cache capacity is 128 experts;
I/O uses 16 threads and direct reads. Diagnostic MTP exclusion, phase markers
and transfer markers are enabled in both cases. Prompt and conversation
caches are disabled. Each configuration has one validated warmup and one
measured request at each length.

| Input tokens | Original GDN interval, ms | Tuned GDN interval, ms | Original resident prompt, ms | Tuned resident prompt, ms |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 269.616 | 219.410 | 14,719.5 | 14,679.0 |
| 8,087 | 530.603 | 431.262 | 18,351.4 | 18,261.8 |

All eight first heads contain 248,320 finite floats and match the accepted
original head in every bit; output IDs also match. Every request reports
one chunk, a full input read and `RESUME 0`. Expert transfer bytes are equal
across configurations: 45,402,470,400 at 4,096 tokens and 46,989,286,400 at
8,087. GDN phase intervals fall by about 19%, while the total wall difference
is much smaller. These single profiled repeats do not establish a stable
overall speed gain; paired ordinary CLI runs are measured separately.

The [unchanged key-head control](original-keyhead-control/run.json), using
SG32 with the default GRF setting in the preceding accepted AOT binary,
also matches all heads. It takes 3,428.097 ms in the GDN interval at 8,087
tokens and is not selected.

## Ordinary CLI comparison

The [paired CLI summary](paired-cli-wall/summary.json) uses three pairs of
ordinary CLI runs on the same immutable JIT binary. Configuration order
alternates original/tuned, tuned/original, original/tuned. Input-length order
also alternates. Phase markers, transfer markers and PLE preload are disabled.
The persistent JIT cache is enabled in both configurations. Context, chunk,
cache, I/O and attention settings match the profile above. Model loading and
generation are outside the prefill timer.

| Input tokens | Original median, ms | Tuned median, ms | Elapsed reduction | Original tokens/s | Tuned tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 13,550.2 | 13,506.0 | 0.33% | 302.28 | 303.27 |
| 8,087 | 17,141.0 | 17,036.5 | 0.61% | 471.79 | 474.69 |

All twelve complete first heads contain 248,320 finite floats, match the
accepted original head in every bit, and produce the same output token ID.
Every run reports one chunk. The incremental rate between these two lengths
changes from 1,111.45 to 1,130.43 tokens/s. This incremental rate measures
latency growth over the two prefixes; total throughput for the 8,087-token
input is 474.69 tokens/s.

The [normal MTP server check](normal-mtp-serve.json) also passes four requests:
first, repeat, another prompt, and restoration of the first prompt. The repeated
and restored requests reuse their checkpoint and return identical four token
IDs and finite logprobs. That existing check uses short prefill chunks, which
take the variant's fallback path. A separate long-prompt check exercises the
new path with normal MTP and conversation snapshots.

The [long-prompt MTP check](long-mtp-serve/summary.json) uses a 1,025-token
completed chat input: 1,016 tokens from the code-review fixture followed by
the verified assistant suffix from the writing fixture. Context is 2,048,
prefill chunk 1,024, cache capacity 128, speculative window 4, and the
conversation cache is enabled. Timing instrumentation is disabled. Both
original and tuned settings use the same JIT binary and normal MTP pack.

The initial request processes 1,024 prompt tokens in one chunk, exercising
the tuned recurrence. Both settings produce identical complete finite first
heads (248,320 floats), four output token IDs and finite logprobs. Repeated
and restored requests reuse 1,018 tokens and match their first request's IDs
and logprobs. All four requests also match between the two settings. These
are correctness checks; their server times are not a speed comparison.
