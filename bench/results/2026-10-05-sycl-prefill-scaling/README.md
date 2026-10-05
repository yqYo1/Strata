# Arc B570 prompt-length scaling

This measurement separates expert-weight transfer work from the way prompt
latency grows with input length. The device is an Intel Arc B570 10 GB, with a
Ryzen 5 5600X and 128 GB of installed system RAM (125.72 GiB reported by
`MemTotal`). The model is the native IQ3_S
Qwen3.8-Flash-Next pack. The selected compute runtime is 26.35.39758.10.

The earlier 826- and 4,007-token measurements used different context limits and
chunk settings. The engine also defaults to a separate 256-token first chunk
to overlap PLE disk reads. That repeats the streamed expert-weight work and
makes those two measurements unsuitable for estimating a per-token cost.

[prefill_profile.py](../../../sycl/tools/prefill_profile.py) fixes the context
limit, cache capacity, prefill chunk, and input token prefix across lengths.
It sets `STRATA_PREFILL_FIRST=0` and checks that every run processes exactly
one chunk. It alternates the length order between repetitions. `--warmup` first runs
and validates each length, preserving those records while excluding them
from medians and slopes. `--reverse-order` starts with descending lengths.
Use both settings for matched comparisons of newly compiled GPU modules.
Model loading
and generation are outside the prefill timer. By default it omits the MTP
draft layer; `--mtp DIR` also measures the production draft-KV work.

The code-review fixture contains the beginning of three repository files from
revision `e274e17`. Its prefixes are diagnostic causal inputs, rather than
separately completed chat turns. The fixture metadata records the source
excerpts and tokenizer hashes. This avoids repeating one short sentence to
fill a long context.

| Measurement | Meaning |
| --- | --- |
| `expert_bytes`, `expert_copies` | Actual non-resident expert blobs copied to the GPU, including layer totals. Dense weights loaded once at startup are outside this count. |
| `dma_active_ms` | Sum of the memcpy events' device start-to-end durations. Excludes queue gaps. |
| `copy_span_ms` | Time from the first copy's device start to the last copy's device end. Includes gaps for routing, staging, and occupied ring slots. |
| `host_copy_worker_ms` | Sum of CPU staging/read durations across the worker threads, excluding their buffer waits. Parallel worker durations are not wall time. |
| `phase_ms` | Compute-queue intervals between phase markers. Includes launch gaps and waits; this is not a sum of kernel execution times. |
| `wall_ms` | Elapsed prefill time captured before querying and printing the profiling records. |
| `marginal_ms_per_token` | Difference in median wall time divided by the token-count difference for the two longest inputs. |

DMA, staging, and compute overlap. Their durations cannot be added to get
wall time, or subtracted from wall time to isolate computation. MoE routing
can change the transferred bytes with input length. Interpret the measured
latency slope alongside those bytes, rather than assuming the whole model
is copied for every input. Attention and other token-dependent work can also
make the slope change with context length.

Use `--mode profile` for phase markers and transfer events, `--mode transfer`
for transfer events alone, and `--mode wall` for elapsed time with both forms
of instrumentation disabled. The wall runs provide the speed comparison;
the profiled runs explain it. Compare first-logit hashes at matching lengths
to check that instrumentation leaves the model output unchanged.

The initial instrumentation check used the earlier 826-token, two-chunk
configuration, with MTP and 512 cache slots. It copied 24,468 expert blobs
(50,434,304,000 bytes), with 7,926.15 ms of active DMA and 2,619.90 worker-ms
of CPU staging. All 248,320 first-logit floats were bit-identical to the
unprofiled accepted AOT binary, and the eight generated token IDs matched.
The profiled prefill wall time was 13,830.73 ms; phase markers introduce
overhead, so this is not a speed improvement over the unprofiled baseline.

The first length sweep fixes the context at 8,192, the prefill chunk at 8,192,
and the cache at 128 slots, with MTP omitted. All three inputs fit one chunk.
The wall-time result is the median of three runs, ordered short-to-long,
long-to-short, then short-to-long. Transfer measurements are from one separate
profiled run per length, rather than from the uninstrumented speed runs.

| Input tokens | Median wall time | Overall prefill | Profiled expert bytes | Profiled active DMA |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 12.657 s | 161.81 tok/s | 42.496 GB | 6.674 s |
| 4,096 | 17.340 s | 236.22 tok/s | 45.402 GB | 7.129 s |
| 8,087 | 24.582 s | 328.98 tok/s | 46.989 GB | 7.374 s |

Between the two longest inputs, wall time grows by 7.243 seconds for 3,991
additional tokens: 1.815 ms/token, or 551.05 additional tokens per second.
This is a measured local slope, including routing, attention, and PLE reads;
it is not the GPU's pure arithmetic throughput. The transferred expert bytes
grow by 1.587 GB over the same interval. The 8,087-token profiled run reads
90,360 SSD pages for the PLE table; the prompt path reports 6.169 seconds of
PLE handling, most of it waiting for the row gather.

The model files are on ZFS (`rpool/USERDATA/yayoi`, 128 KiB records,
compression off, `primarycache=metadata`), and the direct reader uses
the Linux default of 16 blocking-read threads. This storage and reader setting
is part of the measurement condition. It affects the token-dependent row
reads as well as the GPU work. Future comparisons keep the file and GPU
settings fixed and report any change to the reader concurrency explicitly.

All nine wall runs produce finite first logits that match the corresponding
profiled run bit for bit across all 248,320 values. The generated token also
matches at each input length. See the [profiled record](profile/run.json),
[wall record](wall/run.json), and [parity checks](profile-wall-parity.json).

Increasing `STRATA_IO_THREADS` from 16 to 64 did not improve the longest-pair
slope: 1.833 ms/token versus 1.815. PLE handling times were also similar. All
nine runs retain the baseline's first-logit bits. The [64-thread record](io64/run.json)
is an experiment, rather than a new default.

The CPU-only [PLE probe](../../../sycl/tools/ple_read_probe.cpp) calls the
original table gather and decoder. Its [record](ple-probe/run.json) separates
cold reads from page-cache hits: the first 8,087-token mmap gather took
21.098 seconds, and the repeated gather took 0.157 seconds. The later direct
reads also benefited from the warmed pages. Their speed is not evidence that
64 threads beat 16. The embedding hashes agree in all modes and repetitions.

`--preload-ple` is an explicit diagnostic mode. It completes the same PLE
gather before the GPU/expert-transfer timer, reuses its embedding, and reports
the preload separately. The CLI's enclosing timer still includes it. It
requires one chunk, and does not disable the PLE or replace its values.
The [preloaded record](preloaded/run.json) has one profiled run per length:
GPU/expert-transfer stage times of 12.267, 14.705, and 18.598 seconds. The
4,096-to-8,087 local slope is 0.975 ms/token (1,025 additional tokens/s).
This diagnostic slope is not overall prefill throughput or a measured speed
improvement. All three first-logit arrays match the unmodified baseline bit
for bit. The extra phase markers separate PLE read waits from combine and
hyper-connection write/normalization time.

[The measurement plot](prefill-scaling.png) shows elapsed time and transfer
work on separate axes. Normal prefill points are medians with the observed
three-run range; the preloaded diagnostic and transfer points each have only
one profiled run per length. Reproduce the PNG and [SVG](prefill-scaling.svg)
with `python plot.py` in an environment containing Matplotlib.

The [attention experiments](attention/run.json) test the existing XMX kernel
and larger launch batches of the original fallback. On this B570, both XMX
chunk sizes are slower than the fallback in the synthetic INT8 and FP16
cases. Those precision checks pass, but the unchanged upstream test then
refuses Q4_0 K and exits 2; it is not a complete passing XMX test.

`STRATA_PREFILL_ATTN_BATCH` selects 1 to 1,024 queries per fallback launch
(default 32), capped at the current chunk length. The buffer estimate and
relayout use the same count. Each query keeps the same scratch layout,
selected cells, and arithmetic. The parity test's fourth argument compares
that batch with 32 and requires finite, bit-identical outputs. Warmed timings
alternate order across three rounds; the 32-versus-32 control measures the
timing bias. INT8, FP16, and Q4_0 pass across the tested batch sizes.

One matched profiled model run per length and batch gives the following
diagnostic result. These runs preload the same PLE input, with its time
reported separately, and do not establish an end-to-end speed improvement.

| Input tokens | Batch 32 attention | Batch 128 attention | Batch 32 GPU/transfer stage | Batch 128 GPU/transfer stage |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 1.001 s | 0.955 s | 14.821 s | 14.694 s |
| 8,087 | 2.270 s | 2.165 s | 18.758 s | 18.501 s |

The stage's longest-pair slope is 0.986 versus 0.954 ms/additional token.
All four first-logit arrays match the accepted baseline across all 248,320
float values, bit for bit, and the generated IDs match. See the
[batch 32 record](attention/model-batch32/run.json) and
[batch 128 record](attention/model-batch128/run.json). This is a small pilot
comparison; 128 is available for measurement rather than a new default.

The next [layout study](attention/layout-study/run.json) changes how the
original attention reads its shared query tile and assigns its work to
subgroups. `STRATA_PREFILL_ATTN_LAYOUT=1` transposes the query tile on subgroup
32; `=2` pairs the original lanes on subgroup 16; `=3` does both. Unset or
zero keeps the original layout. The 16-lane version computes the original
lane and lane+16 dots independently, then performs the original XOR-16
addition before the XOR-8/4/2/1 reduction. It keeps 64-cell partials, the
merge, softmax, per-dimension value accumulation, and scratch size.

In the synthetic 8,192-cell, 2,049-query test, layout 3 with 128 queries per
launch takes 51.685 ms for INT8, versus 58.769 ms for the original 32-query
launches. The corresponding FP16 times are 47.196 and 54.419 ms. A larger
batch alone takes 55.884 and 51.906 ms. Two runs per layout use reversed
layout order; each timing is the median of three warmed rounds of ten
launches, with old/new order alternated. All 64 checks, including K8V4 with
signed Q4_0 value scales, give finite, bit-identical outputs and pass the
existing FP64 reference bounds. Transposing alone gives no improvement here.

The [initial model runs](attention/layout-study/model-initial-0/run.json)
are archived as warm-up evidence. The first 4,096-token run spends 3.677
seconds in the selection interval rather than about 0.094 seconds, consistent
with newly compiled GPU modules. Its automatically computed length slope
must not be used as steady performance. At 8,087 tokens, after the shorter
run, attention takes 2.167 seconds for layout 0 and 1.989 seconds for
[layout 3](attention/layout-study/model-initial-3/run.json). These are single
profiled observations. All four model runs match the accepted first-logit
arrays and output IDs; a warmed, matched comparison is still required before
changing the default. Both attention tests pass through CTest on the B570.


The [warmed profile comparison](attention/layout-study/paired-profile/summary.json)
uses three pairs of runs at each length. Configuration order is baseline,
layout 3; then layout 3, baseline; then baseline, layout 3. The second pair
also reverses the input-length order. Each configuration first runs both
lengths as recorded, excluded warm-ups. All 16 first-logit arrays, including
warm-ups, are finite and match the accepted baseline across all 248,320
floats, bit for bit. Generated IDs also match.

| Input tokens | Original attention | Batch 128, layout 3 attention | Original GPU/transfer stage | Batch 128, layout 3 GPU/transfer stage |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 1.001 s | 0.877 s | 14.737 s | 14.623 s |
| 8,087 | 2.272 s | 1.989 s | 18.607 s | 18.315 s |

These are medians of three profiled runs with PLE preloaded. At the longer
input, attention time decreases by 12.4%; the measured stage time decreases
by 1.57%. The slope of the median stage times changes from 0.9695 to 0.9251
ms/additional token. Individual pair slopes range from 0.9643 to 0.9863 for
the original and 0.9185 to 0.9319 for the variant. All runs transfer the same
45.402 GB at 4,096 tokens and 46.989 GB at 8,087 tokens. These observations
explain the attention improvement; the separate normal elapsed-time
comparison below assesses overall prefill speed. The default remains unchanged.


The [normal elapsed-time comparison](attention/layout-study/paired-wall/summary.json)
repeats the same three paired orders with both profiling forms and PLE preload
disabled. It uses the same immutable binary, token prefixes, memory settings
and 16 PLE reader threads. The OS row cache is warm. Loading the model and
reading the final held-out prompt token during decode are outside this timer.
All 16 complete first-head arrays and generated IDs, including the four
excluded warm-up runs, match the accepted baseline.

| Input tokens | Original prefill time | Batch 128, layout 3 prefill time | Original overall prefill | Batch 128, layout 3 overall prefill |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 13.737 s | 13.648 s | 298.18 tok/s | 300.12 tok/s |
| 8,087 | 17.560 s | 17.283 s | 460.53 tok/s | 467.93 tok/s |

Each time is the median of three measured runs. At 8,087 tokens the observed
ranges are 17.554–17.563 seconds for the original and 17.280–17.302 seconds
for the variant: a 1.58% decrease in median latency and a 1.61% increase in
overall prefill throughput. The 4,096-token variant has a wider range,
13.597–13.719 seconds, versus 13.734–13.739 for the original. The slope of the
median times is 0.9581 versus 0.9107 ms/additional token; individual variant
pair slopes range from 0.8930 to 0.9228. Overall throughput at 8,087 tokens
is 467.93 tok/s; the 1,098 additional tok/s from the local slope is a separate
metric. This comparison does not include the MTP draft layer or measure decode.


The [compact XMX probe](attention/layout-study/compact-xmx/run.json) reduces
requested shared local memory from 86,784 to 62,208 bytes by reusing regions
after their last readers and processing one output tile per subgroup at a
time. All six supported-case outputs match the original XMX CH32 kernel bit
for bit. However, both K8V4 cases fail the existing FP64 and FP32 fallback
accuracy bounds: matching the original XMX does not establish adequate
precision. The [complete log](attention/layout-study/compact-xmx/wall-timer-probe.log)
retains those failures. This variant is not used in the engine. The initial
event-timer attempt also failed because the queue did not enable profiling;
that record is retained separately. The subsequent probe uses synchronized
wall time, with the same mathematical operations and accuracy bounds.

The [selection probe](attention/layout-study/selection-study/run.json) keeps
floating block scores unchanged and varies the integer radix selector. It
compares 256, 512 and 1,024 threads, with serial or parallel searches of the
256 histogram bins. All 48 cases match both the original GPU selector and
an independent stable CPU sort, including ties, signed zero, NaN, infinities,
denormals, identity selections, incomplete tails and capacity/live bounds.
At 8,192 cells and 129 queries, the 256-thread, nine-key parallel version
takes 0.0408 ms in the continuous-score case, versus 0.3447 ms for the
original. These are medians of three warmed rounds of three launches,
with order alternated; they are synthetic wall times, not model throughput.
The probe reports zero spill bytes for that version, while its 33-key version
reports 17,216. Any engine variant therefore needs an explicit live-size bound
and model validation. Both probes retain their source and binary hashes.

The bounded prompt selector is now available with
`STRATA_PREFILL_TOPK_TUNED=1`. It uses 256 threads, nine keys per thread and
an integer parallel histogram scan. It keeps the original floating scores,
thresholds, tie order and ascending output IDs. The original selector remains
the default and handles more than 2,304 live blocks or unsupported geometry.
The integrated test checks 28 cases against the original GPU selector and
an independent CPU sort; all pass, including both sides of the size bound.
All 28 registered JIT tests pass in the
[complete test log](attention/layout-study/resident-control/jit-ctest.log).

The [resident comparison](attention/layout-study/selection-model/resident-profile/summary.json)
uses the IQ3_S model on the same B570 and Ryzen 5 5600X. Each configuration
has one validated warmup and three measured requests at each length, with
opposite starting orders. Batch 128/layout 3, the 8,192-token chunk, 128
expert-cache entries, PLE direct IO and 16 readers are fixed. Every request
has a full uncached read and one prompt chunk. All 16 complete first heads
contain 248,320 finite values and match the accepted CLI reference bit for
bit; output IDs also match. Expert transfer volumes stay unchanged.

| Input tokens | Original selector phase | Tuned selector phase | Original resident request | Tuned resident request |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 93.539 ms | 42.178 ms | 14.781 s | 14.734 s |
| 8,087 | 307.274 ms | 157.781 ms | 18.535 s | 18.362 s |

These resident request times include setup and phase instrumentation.
They support a comparison within this method; normal CLI elapsed times
need their own confirmation.

`sycl/tools/prefill_resident_profile.py` repeats a completed CLI fixture in
one loaded engine and checks every full first head against that CLI record.
It disables prompt and conversation caches, requires `RESUME 0`, validates
the full read and one prompt chunk, and excludes validated warmups from
medians. `--mode wall` removes phase markers. Its timer includes request
setup and must be compared with other resident requests.

For a reference without MTP, the tool explicitly enables
`STRATA_SERVE_NO_MTP=1`. This diagnostic requires one GPU, no prompt or
conversation cache, and no suffix drafts. It keeps the requested prompt
chunk or fails; it skips MTP allocation and drafting and verifies one token
at a time. The normal server still requires MTP and uses its original
memory margin and chunk retry policy.

The [control record](attention/layout-study/resident-control/run.json)
checks four resident requests against the original CLI with batch 32/layout
0. Both warmups and repeated 4k/8k requests use one chunk and match every
first-logit bit. The separate
[normal MTP server check](attention/layout-study/resident-control/normal-mtp-serve.json)
generates four tokens and four finite logprob records in each of four
requests. Repeated and restored conversations reuse checkpoints and
reproduce the original output IDs and logprobs exactly.

The [normal CLI confirmation](attention/layout-study/selection-model/paired-cli-wall/summary.json)
uses three pairs at 8,087 tokens, reversing the configuration order in the
second pair. It has no phase markers, transfer timing or PLE preload, and
keeps the same chunk, cache and attention settings. All six complete first
heads and output IDs match the accepted original reference. Median prefill
time falls from 17.278 to 17.147 seconds, or 468.07 to 471.64 tok/s.
The elapsed reduction is 0.76% for this input and hardware. This is an
additional selector gain after batch 128/layout 3, and an overall CLI
rate rather than the local input-length slope.

The [FP16 library group probe](moe/f16-library-group-study/run.json)
compares direct oneMKL 2026.1 GEMM calls with grouped GEMM calls for the
native experts' 2,560-by-640 down and 1,280-by-2,560 gate/up matrices.
Groups contain two, four or eight experts, with varied row counts and
maximum counts of 1, 8, 32 or 128. Output rows use stride `N+3` for
guard columns, so those measured times include that stride. It measures
three warmed rounds of ten
calls in alternating order, including host submission and queue completion.
All 24 shapes preserve output padding and finite values, and sampled
independent FP64 references stay within the existing normalized error
bound. Eighteen shapes also match the original output bits everywhere;
six change some bits. Grouped calls are slower in every shape. For four
down products with up to 128 rows each, individual calls take 0.962 ms
and the group takes 3.077 ms. This library batching route is not adopted;
the raw log distinguishes the numeric bound from complete bit identity.

The final [B570 AOT validation](aot-validation/validation.json) uses
`STRATA_SYCL_AOT=bmg-g21`, precise floating point and correctly rounded
FP32 divide/sqrt. All 28 registered tests pass, with no skipped tests.
The real IQ3_S model then runs 4,096 and 8,087 input tokens with batch
128/layout 3 and the tuned selector; both use one chunk and match all
248,320 finite first-logit bits and output IDs from the original CLI.
These are single correctness runs. The AOT MTP server also passes four
requests, including checkpoint reuse and exact output/logprob restoration.
AOT covers Strata's SYCL kernels; oneMKL remains its separately supplied
library.

## GDN register tuning

The [GDN study](gdn/README.md) keeps the original key-head arithmetic while
requesting subgroup size 16 and 256 GRFs for at least 256 input tokens.
`STRATA_GDN_KEYHEAD_TUNED=1` enables it; it defaults to off. Full state and
both output formats match the original pipeline in the eight integrated
parity cases. The Arc B570 reports no register spill for the selected kernel.

Three paired ordinary CLI runs, with batch 128, layout 3 and the tuned prompt
selector already enabled in both cases, change the 8,087-token median from
17,141.0 to 17,036.5 ms (0.61% elapsed reduction, 471.79 to 474.69 tokens/s).
The 4,096-token median changes from 13,550.2 to 13,506.0 ms. The complete
first heads and output IDs match in all twelve runs. See the linked study for
the synthetic controls, profiled phase intervals and comparison settings.

## Aligned FP16 library controls

The [padding study](moe/f16-padding-study/run.json) uses the engine's aligned
output stride `N` for 2,560-by-640 down and 1,280-by-2,560 gate/up products.
It tests nine active row counts and rounding to 32, 64, 128 or 256 rows, with
queue profiling both disabled and enabled. Each mode has 72 cases. All
outputs are finite, padding and guards are correct, and sampled FP64 error
bounds pass. Complete active output bits match in 65 cases per mode; seven
gate/up cases change some bits. Input staging, zero filling and output copying
make every padded candidate slower than the original. This route is not
used by the engine.

The [wrapper control](moe/f16-wrapper-study/run.json) compares the original
`Gemm::f16` implementation with direct oneMKL calls at ten row counts,
including 1, 3, 8 and 17, for both product shapes. All 20 complete outputs
match bit for bit and preserve guards. Profiling is enabled in the queue,
matching the engine. Wrapper and direct timings remain close; for the
160-row down product they are about 37.5 and 37.3 microseconds. The much
longer profiled engine phase intervals also include scheduling gaps and
streamed-weight waits, as described above.

The [attention query-order study](attention-query-order/README.md) keeps
layout 3's arithmetic and exchanges the query/chunk workgroup dimensions.
All 32 standalone cases and all 30 JIT tests pass. In three matched normal
CLI pairs at cache 128 and context/chunk 8,192, the 8,087-token median falls
from 17,034.7 to 16,924.2 ms (0.65%), or 474.74 to 477.84 overall tok/s.
The longest-pair incremental rate improves from 1,128.58 to 1,146.41
additional tokens/s; it is distinct from overall throughput. All twelve
complete finite first heads and IDs match the accepted baseline. Short and
long normal-MTP checkpoint checks also pass. The option is
`STRATA_PREFILL_ATTN_LAYOUT=4`; the default layout remains 0.
