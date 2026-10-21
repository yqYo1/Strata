# Native 16-bit long prefill

## Current measured state

Design, tune and select independent prefill and decode execution paths by
default. This includes kernels, scheduling and placement. If both
select the same implementation, that is an incidental result of their separate
decisions. Shared inference processing requires a matched measurement showing
that it is faster than independent paths for the recorded shapes/phases.
Existing sharing without that evidence remains a legacy arrangement to audit;
phase containment alone does not establish a sharing benefit.

The current priority is to restore the fastest recorded prefill path while
preserving the present decode implementation, settings and placement. Recover
the exact source/configuration and compare matched inputs, separating first-use
and later requests. A decode impact is a separation defect: check state handoff
and repeated decode performance, resolve the coupling, and keep the restoration
unqualified until it is fixed. New IQ2 and MTP model cohorts remain deferred
during this restoration. Controller `acc1abd4b` prepares the matched117K
baseline on the current phase-scoped build; its four host launch/evidence-refusal
tests and the same-build packed smoke/validate gates (two/three cases) passed.
The checked117K process is next; restoration and decode preservation remain
unqualified.

The current B570 speed reference uses fresh inputs of117489 tokens and separates
the first long request from later requests in each independent process.
Three packed-bank process medians for the later requests are
1002.056/1002.106/1002.149 tok/s; their first-long readings are
929.917/928.963/927.958 tok/s. This meets the recorded1000 tok/s criterion for
later-request medians only. The packed-bank layout's0.223% gain misses its
prospective2% screen and remains opt-in. See the
[individual samples and frozen comparison](../bench/results/2026-10-15-pr6-recovery-audit/prefill-packed-bank-speed-current.json).

The next selected component is the default-off CPU IQ2_S NT1 gate/up pair
producer. Its comparison matches the actual model baseline and uses
`STRATA_IQ_SINGLE_EXACT=0` in both arms:288 samples across six independent
processes preserve all output bits and reduce the two tested format
combinations' pool latency by8.85%/7.29%.
These are component results, with actual dispatch and work conservation, not
model speed gains. See the
[component decision](../bench/results/2026-10-15-pr6-recovery-audit/cpu-iq2-current.json)
and [current model comparison](../bench/results/2026-10-15-pr6-recovery-audit/iq2-pair-model-current.json).
The original observed pilot remains at commit`bedfdf312`; the repeated
comparison's controller pins and validates that Git record directly.

All six planned same-binary processes closed normally, with no fresh driver
fault, and the prospective reducer agrees with the retained individual samples.
Both decode workloads pass the faster-screen rule: paired candidate/base
throughput ratios are 1.016474/1.016684/1.015472 for code and
1.015337/1.010841/1.019018 for prose, with geometric means 1.016210/1.015060.
These are observer-charged screens, not clean throughput or causal hardware
attribution. MTP acceptance classes are confirmed in both arms at 83.11%/39.06%.

The candidate does not qualify under the original whole-process advancement
rule. This does not reject a separately scoped decode implementation. The
first code prefill fell from 637.4 to 484.7 tok/s (52104.8 to 68521.9 ms for
33214 fresh tokens). All three code prefill ratios are
0.760411/1.001320/0.999605: their geometric mean 0.913026 fails the 0.98
minimum, and the first pair fails the every-pair minimum of 0.95. Keep this
failed outcome; later samples do not erase it. Prose prefill's geometric mean
0.992706 passes its separate nonregression gate. A successful process/overall
`passed` flag denotes execution gates, not performance eligibility. The IQ2
option remains off by default. New phase-specific adoption criteria and
evidence must be committed prospectively; keep this original failed outcome.

The long layer-major GPU prefill does not call the CPU IQ2 producer. Its server
prompt clock ends after cache refill, before the last-token verification and
decode window loop. The source therefore does not establish that IQ2 pairing
caused the first prefill slowdown. The shared CPU verifier path can also read
short/window prompts. Source `6e40afc1b` now copies an explicit phase into each
synchronous CPU batch: prefill keeps the reference and only decode can select
the default-off IQ2 candidate. Request/token scopes restore the prior phase on
exit, including errors, without changing process environment variables.
The new engine and twelve fixtures compiled. Separate pair-off/on CPU children
passed twelve cases each: two formats, NT1/NT2, prefill/decode/prefill sequencing,
actual pair-row dispatch, identical output bits, guards and scope restoration.
The fifteen-case CPU census host contract and eighteen same-build engine
groups also passed, with unchanged boot and no fresh driver faults. These are
component checks, not new model speed, model-route or adoption evidence. The
legacy shared CPU dispatcher is contained, not justified as the final architecture.

The final baseline/candidate pair measured code PP 636.6/636.3 and TG
14.5/14.8, and prose PP 761.0/745.1 and TG 9.3/9.5 tok/s. Each code request
read/generated 33214/3611 tokens, and each prose request 33288/8638 tokens.
Original samples and all rejected prefill measurements remain in the linked
comparison. Source/metadata research proceeded during this diagnostic;
independent background CPU/I/O interference was not separately quantified.

Main's MTP observer source at `0b39c726f` is integrated into `feature/sycl`.
The new engine and eleven fixtures compiled at `05223b559`; 25 host-only
controller/refusal tests and the 15-case CPU census build contract passed.
The bounded profile-off/on graph fixture passed numerical/guard, event-provenance,
two-replay clock and checked teardown gates with no fresh faults. Same-build
eighteen-group engine checks also passed. Model execution and timing remain
unqualified.
The validated6c engine and ten fixtures are preserved and hash-checked after
recovering 9845 original inputs from Git. The unmeasured child branch/worktree
is retired. The prepared three-process weak-prose plan was closed before
any model launch after the phase-specific adoption clarification. Its packed
smoke/validate gates passed; they remain component evidence. Next, qualify the
new decode selection and compare it under a prospective decode-specific plan,
then resume MTP attribution with a newly qualified plan.
Graph clocks are nonadditive first-to-last-node
envelopes; hardware service and saturation remain unknown.

A post-hoc mechanism check limits what this IQ2 component can explain. In the
closed baseline, its two format bins account for 18.69%/17.81% of total decode
wall time for code/prose. Apply the component latency ratios 0.911505/0.927087
to those bins while holding other work and overlap fixed: with total decode
time D and per-format pool time P, `D / (D - sum(P * (1 - ratio))) - 1` gives
only about 1.61%/1.53% throughput improvement. This is a consistency calculation
using rounded representative component mixtures and observed pool wall,
including scheduling/waits; it is not a causal result, physical resource
ceiling or new acceptance rule. It motivates wider work reduction and MTP
investigation rather than extrapolating a component's 7–9% to the whole model.

The observed IQ2 pilot completed both33K thinking-generation cases at14.8/9.5
tok/s, with3611/8638 generated tokens. Its input, output/thinking hashes and
work census match the prior14.5/9.3 tok/s diagnostic. These are single observed
processes on different native builds, not a causal speed comparison.
Current CPU-pool wall intervals account for45.73%/43.19% of decode, including
scheduling and waits; they do not establish CPU or RAM saturation. GPU-reach
waits are dependency intervals, not arithmetic service. The
[closed work census](../bench/results/2026-10-15-pr6-recovery-audit/cpu-work-census-current.json)
records the remaining hardware questions. Sustained70 tok/s, clean repeated
candidate generation, agent-session, broad task quality and physical262144
context/session/RESTORE qualification remain open. Component or source checks
below do not close those model gates.

The closed pilot also constrains the next decode architecture decision. Combine
its measured per-format job counts with the blob sizes in the
[closed census](../bench/results/2026-10-15-pr6-recovery-audit/cpu-work-census-current.json):
`sum(blob_bytes * sum(jobs_NT)) / generated_tokens` counts one logical expert
blob per grouped CPU job, including a multirow job only once. The pilot at
`bedfdf312` has the same work counts and static formats. This covers target
experts, excluding separate MTP work:

| Workload | Generated tokens | Logical CPU expert GB/output token | Logical GB/s at70 tok/s |
| --- | ---: | ---: | ---: |
| regular-code | 3611 | 0.774322 | 54.203 |
| varied-prose | 8638 | 1.086338 | 76.044 |

If every such footprint required one DRAM read and the available read budget
equalled the measured six-thread256MiB scan's39.35GB/s, these workloads would
allow50.82/36.22tok/s before arithmetic and other costs. Fitting70tok/s into that
budget would require27.4%/48.3% fewer DRAM bytes per output token. Alternatively,
copying every current CPU job's full blob to the GPU over a hypothetical6GB/s
payload link would allow only7.75/5.52tok/s under unchanged work and placement.
These are conditional budgets, not measured traffic or hardware ceilings:
cache reuse, repeated kernel reads, overlap and the scan's access pattern differ.
IQ2 pairing preserves this logical weight footprint. Its instruction savings
therefore need the current repeated model comparison; the larger target also
requires investigating reuse, placement, verification work and MTP costs rather
than assuming that faster CPU arithmetic or complete GPU miss offload suffices.

The repeated comparison's closed baseline process also supplies a separate MTP
budget. In `src/program/generate.cpp`, the request runs target verification,
commit/emit and then `MtpDrafter::draft` before constructing the next window.
The outer draft clock includes host work, graph submission, completion waits
and charged observers; it is not GPU arithmetic service time.

| Workload | Emitted tokens / windows | Draft ms/window | Whole-window budget at 70 tok/s | Draft-only conditional tok/s |
| --- | ---: | ---: | ---: | ---: |
| regular-code | 3611 / 1035 | 33.14 | 49.841 | 105.28 |
| varied-prose | 8638 / 3978 | 33.05 | 31.021 | 65.70 |

The last column is `emitted_tokens * 1000 / (windows * draft_ms)`. Holding
the observed draft cost and emitted tokens per window fixed, CPU expert
improvements alone cannot reach 70 tok/s on the prose case, even if target
verification became free. This is a conditional serial budget, not a hardware
ceiling or a clean throughput measurement. The printed draft mean has 0.01 ms
resolution and includes the request's terminal window, which can skip drafting.
After the frozen cohort, separate draft host submission/waits from attention,
projections/MoE and head service before selecting a kernel or scheduling change.
`MtpDrafter::draft` currently waits after the round and each chain step; removing
those waits also requires preserving mapped inputs, NaN/probability stopping,
KV state and checked owner retirement. The indexed chain/window reviews retain
those holds; this budget clears none of them.

The current single-GPU residual handoff is a device-to-device path: the caller
binds `ver.final_R_all()` to MTP, `Verifier::R_` is carved from a `cudaMalloc`
arena, and the SYCL compatibility allocator uses `sycl::malloc_device`.
MTP's `Rin_` is another device-arena slice. `copy_from_mapped` is an elementwise
GPU copy here; its name does not establish a host source. At T=4, HC=4 and
N=2560 this moves 163840 logical bytes within device memory, rather than a
hidden-state round trip through RAM. Mapped token/step/output records and any
KV host backup remain separate paths; this source audit is not a measurement
of total PCIe traffic. The residual owner remains valid through the serial
draft completion before the next verifier run.

## MTP graph observer: tiny support fixture passed, model checks pending

The default-off SYCL observer separates host wall and calling-thread CPU time
for the draft round and chain steps. `STRATA_SYCL_MTP_OBSERVER=host` retains
ordinary graph finalization; `graph` explicitly requests executable-graph
profiling. Both use the same existing per-replay stream synchronization and
retain the returned completion event before another replay can overwrite it.
Neither mode removes the probability/NaN stopping check, mapped-record fence,
KV writes or existing round/step order. Profiling can change graph optimization
throughout the process, including outside the sampled windows. This is not a
clean speed measurement or an optimization adoption.

The server request selector defaults to 5 (the first long-generation request
after the four benchmark warm-ups); the single-request CLI defaults to 1.
`STRATA_SYCL_MTP_OBSERVER_REQUEST` explicitly overrides either default,
independent of CPU census/audit settings, without mutating the environment.
`STRATA_SYCL_MTP_OBSERVER_SKIP` defaults to 8 draft calls and
`STRATA_SYCL_MTP_OBSERVER_WINDOWS` defaults to 64, with a hard maximum of 64.
One selected request per process may emit at most 41 observer records: a summary, eight
phase aggregates and the first 32 raw graph-clock tuples. The omitted-tuple
count is explicit. This bounds query/output work; the external job deadline
must still supervise a driver call that does not return.

Capture, launch-wrapper, stream-drain-wrapper and timestamp-query intervals are
disjoint host regions, split between round and chain. Remaining time includes
mapped input preparation, output/probability reads, control and observer work;
it is not automatically classified as CPU arithmetic. The outer scope is each
`draft()` invocation, including capture and device switching. It excludes
`draft_first()` residual staging and differs from the legacy `ms_draft` clock,
which starts after round capture. Invalid clocks, overlapping regions,
exceptions, incomplete requests and insufficient windows are not a complete
profile. Calling-thread CPU time excludes runtime/helper threads and is not CPU
saturation evidence. Non-Linux thread CPU attribution is unsupported; enabling
the diagnostic there is rejected during load, before MTP GPU allocations.

Graph timestamps are queried only after checked completion, with no additional
marker or wait. Partial tuples and status are retained; unsupported/invalid
clocks or a query exceeding one second stop the diagnostic without a retry.
Start-to-end is the first-to-last-node envelope, including possible idle gaps
and dependencies. It is nonadditive with the host intervals and does not measure
arithmetic service, copy-engine service, occupancy, physical traffic or hardware
saturation. A submit/start inversion invalidates queue delay alone when the
positive envelope remains valid. Round/chain envelope sums are reported
separately; internal captured event aliases are never used as node clocks.

Before any model observer run, preserve the current validated engine and
fixtures, pin the new recipe and
build, then run the bounded graph-envelope fixture in separate profile-off/on
processes using the qualified runtime safety settings. Unknown graph profiling
support is not a pass. Applicable engine safety/numerical checks and a committed
bounded model plan follow; compare fresh observer-off, host-only and profiled
processes before drawing timing conclusions. Compilation and the tiny profile-off/on support fixture have passed. This
does not qualify the engine, model timing, observer overhead or performance.

## Tensor storage and precision

`STRATA_SYCL_PREFILL_NATIVE16=bf16` selects an experimental kernel path only
inside long layer-major prefill. Short prefill and decode keep their existing
FP32 route. Leave `STRATA_PREFILL_LAYER_MAJOR_R_DTYPE` unset, or set it to
`bf16`; another saved-residual format is rejected for this path.

The first implementation keeps the HC4 residual in BF16 between normalization,
projection, mixing, residual addition and interlayer copies. HC projections and
MoE gate/up, down and shared-expert projections produce BF16 tensors directly.
Their consumers accept those tensors without a global FP32 restoration.
Expert hidden activations remain FP16, matching the existing dequantized FP16
weight path. XMX GEMM has FP32 accumulators and an FP32 epilogue, then a single
16-bit output store. Normalization, sigmoid and expert summation use FP32 local
values. HC injection coefficients, router decisions/weights and GDN recurrent
state remain FP32.

There are explicit remaining boundaries:

- The PLE block at layer 1 restores and repacks one chunk through its existing
  FP32 interface.
- The final residual expands once for existing MTP, decode and diagnostic
  callbacks. Intermediate layers use a typed BF16 callback and direct copies.
- GDN internal projection and recurrent tensors use the existing FP32 kernels
  unless the separate experimental chunk64 route described below is enabled.
  QSA internals remain FP32 unless the separate typed-QSA option below is enabled.
- The existing scratch arena remains reserved to preserve the short-input
  layout. Native views share its unused ranges, including separate BF16
  residual/gated halves of the old FP32 gated range. This does not yet establish
  a smaller scratch allocation. The interlayer BF16 store uses two bytes per
  value and needs no separate pack/unpack staging slab.

Old float tensor audits, residual dumps, BF16x2 HC and control vectors are
rejected before native16 layer submission. They remain available through the
FP32 reference. Unsupported matrix tuples/geometry fail before GEMM submission;
there is no CPU arithmetic fallback. The first GEMM supports N and K divisible
by 16, with arbitrary positive token counts and strided output. Existing model
matrices on this path meet those dimensions; scalar injection still uses the
FP32-output GEMM.

On Arc B570, the fresh v2 build passed 68 independent GEMM cases, HC4/MoE
post-operations, runtime/compatibility checks, the existing single-queue FP32
GEMM/payload cases, and early/late allocation teardown. Both FP32 and BF16
completed a 129-token, 48-layer model check normally. All 128 target NLLs were
finite; three full-vocabulary rows also matched an independent FP64 NLL oracle.
This establishes mechanics, not model quality: the synthetic comparison had
mean absolute target-NLL difference 0.596 and maximum 6.56. These differences
have not been accepted as a quality tolerance.

The original full FP32 regression timed out in its separate-copy-queue pipeline.
It remains a failure; the narrower single-queue continuation does not turn it
into a pass. This experimental production route requires the qualified single
queue. The timeout mechanism remains unresolved, with no fresh kernel GPU fault
or driver dump. See the
[diagnostic](../bench/results/2026-10-15-pr6-recovery-audit/own-native16-separate-queue-timeout-v2.ja.md)
and [paired model receipt](https://github.com/yqYo1/Strata/blob/8063aacc6749d18089f7a53db5d3e47b745f348c/bench/results/2026-10-15-pr6-recovery-audit/own-native16-model-smoke-v2.json).
An earlier new allocation-registry destruction-order error was reproduced and
fixed separately; the fixed model processes exit normally.

The first official `bench.sh` comparison on the B570 used the same binary and
78,453-token request, thinking on, chunk 8192 and 32,768 GPU residual rows.
FP32 measured 346.5 tok/s; BF16 measured 123.8 tok/s (one process per arm).
The BF16 GPU residual fell from 1,280 to 640 MiB, but this implementation's
throughput fell 64.3%. It is rejected for default/performance adoption pending
kernel improvement. See the
[individual-sample comparison](https://github.com/yqYo1/Strata/blob/8063aacc6749d18089f7a53db5d3e47b745f348c/bench/results/2026-10-15-pr6-recovery-audit/own-native16-first-long-comparison-v1.json).
This is a first screening pair, not a repeated variance study or evidence that
the datatype alone caused the slowdown.

The matched operator check now identifies the new GEMM as a major cost. On
three model geometries, FP16 or BF16 inputs with native BF16 output took
5.57–30.66 times the wall time of the existing oneMKL FP32-output route. Each
geometry/format used one process and three timed submits per route after
warmup, with complete output checks outside timing. This does not identify
the instruction/layout cause or establish a datatype disadvantage. See the
[operator samples](https://github.com/yqYo1/Strata/blob/8063aacc6749d18089f7a53db5d3e47b745f348c/bench/results/2026-10-15-pr6-recovery-audit/own-native16-paired-gemm-summary-v1.json).

A natural 32,863-token diagnostic completed FP32 twice and BF16 once. At 64
stratified positions, every full-vocabulary logit row was finite; the two FP32
controls were byte-identical. BF16 agreed on top-1 at 59/64 positions, with
mean absolute target-NLL difference 0.556. Its sampled mean NLL was lower
(5.05747 to 4.88628), but a sparse sample from one document corpus does not
establish general quality or a permitted degradation. No quality tolerance
has been accepted. See the
[quality review](https://github.com/yqYo1/Strata/blob/8063aacc6749d18089f7a53db5d3e47b745f348c/bench/results/2026-10-15-pr6-recovery-audit/own-native16-natural-quality-review-v1.ja.md).

The original joint-matrix backend remains rejected for speed. The later vLLM
backend below improves measured long-prefill speed. Broader model quality,
physical 262144-context and RESTORE qualification remain required. Keep the
default FP32 route while those gates are open.

The architecture references are the
[vLLM XPU MoE dtype policy](https://github.com/vllm-project/vllm-xpu-kernels/blob/6b30c9b92cc525c32729df988b5f918e7c5661a5/csrc/xpu/grouped_gemm/collective/gemm/moe_dtype_policy.hpp)
and the pinned SGLang/vLLM HC and recurrent dtype contracts indexed in
`bench/results/2026-10-15-pr6-recovery-audit/research-scope-review-round785-v1.json`.
These distinguish 16-bit tensor storage from FP32 accumulation and state.
The first GEMM is authored here using the existing Strata/SYCL matrix API;
the proposal's `provenance.json` records all design and API sources. Framework
wrappers, CUTLASS templates and third-party kernel code were not imported.
Their per-file licenses were checked separately; newly authored project files
use the repository MIT license. XeStrata is excluded as an implementation source.

## Optional vLLM GEMM and typed QSA

The optional `STRATA_SYCL_VLLM_XE2_NATIVE16` build adds the actual Intel BSD-3 vLLM Xe2 direct16-output core. See [SYCL_VLLM_XE2_NATIVE16.md](SYCL_VLLM_XE2_NATIVE16.md) for its explicit runtime selection, isolated AOT object, stricter shape/lifetime contract and repeated integrated measurements. On the B570 at source `6354f0b4`, the matched 39,262-input official workload measured median 394.9 tok/s with vLLM BF16 versus 372.5 FP32 (three independent processes each). The earlier 78K custom-kernel experiment used different context/residency settings and is not the denominator for this gain.

`STRATA_SYCL_PREFILL_NATIVE16_QSA=1` additionally keeps QSA projection/norm/RoPE/append/attention/gate boundaries typed16 for resident unrotated INT8 K/V with session RoPE tables. Indexer/local accumulation and GDN's existing internal FP32 boundaries remain explicit. The shared scratch arena reservation is unchanged. Streaming/host mirrors and rotated/Q4/hybrid KV are currently refused; do not infer full-context adoption from the component tests.


The clean same-binary attention screening at source `d72c8ac3` used four
78,451-token fresh requests per process, thinking on, chunk8192 and no GDN
chunk64. Prefix33 PP was663.9 on first use, then725.2/720.3/725.0 tok/s;
direct INT8 attention was566.9, then592.9/589.0/592.5. Subsequent medians were
725.0 versus592.5, an18.28% direct-path regression. Both processes exited
normally without fresh GPU faults. Keep the existing prefix route for this
configuration. The comparison also changes score pitch33 to32, so it does not
isolate dequantization, physical memory traffic or SLM conflicts. These are
sequential samples within one process per arm;24-token generation does not
qualify sustained decode. See the [individual samples and comparison limits](../bench/results/2026-10-15-pr6-recovery-audit/qsa-attention-model-current.json).

The fixed-score33 K36/V260 experiment passed all bit, oracle and boundary
checks. Six rotated samples per arm on8192 queries changed median latency
by+0.60% (cap33 gathered),-0.51% (cap2051 gathered), and+0.23% (cap2051
clustered). These mixed subpercent results do not justify selection. Source
and individual samples remain at commit`ee14c50d`; the experimental pitch code
is retired from the active implementation.

The closed `qsa_bf16_prefix_rows_batch` experiment retained score33,
K32/V256 and28224 requested local bytes. It publishes first IDs with Q/state
initialization, and next IDs during PV, using the existing final PV barrier.
All current row readers finish before V/probability publication; PV reads no
row IDs. The final barrier still protects V-to-K overwrite. This removes one
workgroup barrier per32-cell tile without changing arithmetic or output order.
Its source and tests remain in Git at `51a6f70a`; the callable is retired after
the negative timing result and emitted-resource comparison. Production dispatch
is unchanged. Main's serialized qualification sequence for the current candidate is
`python3 tools/sycl/build_engine.py rebuild-attention`, then
`python3 tools/sycl/check_attention.py smoke`, `component`, and `perf`.
The oracle fixture compares all output bits with prefix33, including diagnostics
and physical262144 component extent. The timing fixture rotates six samples
per arm on three8192-query shapes, outside uploads/readbacks. These checks
do not qualify full-model context, RESTORE, quality or model throughput.

This schedule passed all those correctness checks, including explicit duplicate
IDs, aliased physical pages and first-/later-tile K/V diagnostics. Six rotated
samples per arm changed median latency by-0.60%,+0.10%,+0.24% for the same three
shapes above. It is not selected. At cap2051, baseline gathered versus clustered
reads took419.8 versus299.4ms with the same matrix work. This is locality
sensitivity, not a physical-bandwidth or saturation measurement.

The current-path MoE census used `python3 tools/sycl/bench_engine.py
prepare qsa-census`, commits its plan, then uses `run qsa-census`. This runs one
official78K request with parameter validation, trace and existing coarse/QSA/CPU
observers. The optional profile records exact planned job-size and jobs/expert
histograms from already available host routes; final issuer totals must match.
It adds no tensor readback or GPU submission. Requested DMA bytes and diagnostic
timings must stay distinct from physical traffic and clean speed.

The installed VTune2026.4 rejects GPU-hotspots on this Ryzen5600X host before
starting the target; this is not an observed permission denial. The maintained
`python3 tools/sycl/profile_attention.py resources` alternative ran the exact
qualified fixture with `IGC_DumpZEInfoToConsole=1`. Traced launch identities map
the executed prefix33 and row-publication kernels to128GRF/SIMD16, raw static `slm_size=32`
plus28224B of dynamic-local arguments, and no emitted scratch descriptor.
These are compiler/launch properties, not hardware utilization or bandwidth.
The static field's unit/schema and physical allocation rounding remain unverified;
the traced dynamic-local argument sizes are explicitly bytes.
The unlaunched compiled score32 variant has1920B spill metadata and2048B scratch;
it must not be attributed to the executed production prefix33.

The next independently authored `qsa_bf16_prefix_qcache_batch` candidate loads
the same16 BF16 Q fragments per QK subgroup once, preserves their DPAS order,
and reuses their initial shared arena for K/V after a uniform workgroup barrier.
It budgets20036 local bytes, including a nonzero four-byte unused accessor.
The installed compiler removes that unused accessor: traced dynamic arguments
total20032B. Default128GRF compilation adds2112B spill metadata and4096B scratch,
yet six rotated samples per arm reduced median latency by11.81%,13.24%,11.83%
for cap33 gather, cap2051 gather and cap2051 clustered. Exact-bit/oracle,
diagnostic/alias and physical262144 component gates passed; no new GPU faults.
Source`0ddd14719211` and individual results at`13976170f` preserve this variant.

The separate comparison applied the installed compiler's
[`grf_size_automatic` property](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_intel_grf_size.asciidoc)
only to the cached-Q kernel through the SDK's functor property API. It selected
256GRF, removed scratch and changed compiler thread-count metadata from8 to4.
Its matched median latency nevertheless increased16.65%,16.20%,12.65%; reject
this variant (`b82207709`). The comparison also changes the SDK functor form,
so it does not isolate GRF mode as the sole cause. Occupancy remains unmeasured.

The original default128GRF kernel source is restored byte-for-byte. The new
explicit `STRATA_SYCL_PREFILL_QSA_MATH=bf16-prefix-qcache` model screening route
uses the existing prefix allocation/admission and leaves default dispatch alone.
Main prepares/commits/runs `qsa-qcache-check`, then matched `qsa-qcache-base` and
`qsa-qcache` with `tools/sycl/bench_engine.py`. The check uses validation/trace;
the two speed arms each use four fresh78K requests, thinking on and the same
binary/configuration. These24-output screens cannot qualify sustained decode,
full-model262144 context, RESTORE or task quality.

The matched78K screen completed with78451 actual input tokens per request and
no reused tokens. Prefix33 PP was676.6 on first use, then725.0/718.5/724.5 tok/s;
cached-Q was708.2, then756.6/751.2/755.7. Subsequent medians724.5 versus755.7
show a4.31% model gain; sample SDs were3.62 and2.89 tok/s. This is one process
per arm, not repeated independent processes. Both exited normally without new
GPU faults. Actual request hashes, binary and configuration match. Retain the
opt-in candidate for further tuning; default adoption and the broader gates
above remain open. Short-output decode medians14.2 versus14.0 tok/s do not
decide sustained decode performance. See the [individual samples and comparison](../bench/results/2026-10-15-pr6-recovery-audit/qsa-cached-query-model-current.json).

The component-only row-coalesced K experiment passed all numerical, guard and
physical262144 component gates, but increased median latency12.44%,27.98%,7.75%
for cap33 gather, cap2051 gather and cap2051 clustered (six rotated samples per
arm; result`eb9b0469c`, source`2d4f24484`). Both attention kernels emitted128GRF,
2112B spill metadata,4096B scratch and20032B dynamic-local arguments. Reject it
and retire its active API/fixtures. Changed local stores and memory-level
parallelism are possible explanations; no physical traffic or bank counters
were collected. The current cached-Q model route keeps its original K gather.

The four-value K candidate preserves parallel row ownership and contiguous
cell stores, grouping four adjacent K values per lane. With thread`t=0..127`,
loop`r=0..15` and element`j=0..3`, its map is
`cell=t%32,d=4*(t/32)+16*r+j`; all8192 local destinations have one writer.
Typed `vec<uint16_t,4>::load` avoids wider-pointer aliasing and needs only the
existing two-byte alignment per [SYCL2020 vec members](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html).
The installed SDK implements this API with typed element loads. Captured
executed ISA lowers these four values to two32-bit A64 loads at offsets0/4,
with align2 retained in LLVM IR; this is not one64-bit message. The oracle also
uses a17-element scratch guard to exercise a pointer unaligned to eight bytes.
BF16 bits, matrix arithmetic/strides and all barriers stay unchanged. Main
authors this candidate independently. Six rotated samples per arm reduced
component median latency9.16%,10.15%,15.72% for cap33 gather, cap2051 gather
and cap2051 clustered (`d39258f6e`, source`112071c5c`). Bitwise comparisons,
guards, two-byte scratch alignment and physical262144 component cases passed.
This does not qualify model speed, task quality, physical262144 model/RESTORE
or sustained decode. Build pins also cover project `third_party` and maintained
`bench/sycl` inputs.

The model screening selector is
`STRATA_SYCL_PREFILL_QSA_MATH=bf16-prefix-kvec4`; defaults stay unchanged.
It uses the same prefix admission, scratch lease and cached-Q arithmetic as
`bf16-prefix-qcache`. Main prepares/commits/runs `qsa-kvec4-check`, then
`qsa-kvec4-base` and `qsa-kvec4` with `tools/sycl/bench_engine.py` for a fresh
same-binary comparison. The first run enables validation and trace; timing arms
are clean processes. A successful model screen alone is not default adoption.

The same-binary model screen (`23b0e7bb8`, engine source`52f047586`) read78,451
fresh tokens per request, thinking on, four requests per process per arm.
Cached-Q first PP was696.8 tok/s; subsequent samples756.4,750.7,756.3
(median756.3, sample SD3.26). Kvec4 first PP was737.2; subsequent
samples792.6,786.6,792.7 (median792.6, sample SD3.49), a4.80% median gain.
Both native engines exited0 without fresh GPU faults. Exact workload hashes,
engine/benchmark inputs, normalized configuration and private profiles matched;
only the QSA math selector differed. Three subsequent requests are not three
independent processes. The24-token output does not qualify sustained decode;
model quality, agent/long-generation, physical262144 and RESTORE gates remain
open, so this route stays opt-in.

## Resident MoE service reference

The maintained `sycl_native16_paired_gemm_bench --resident` fixture calls the
same vLLM Xe2 `beta0` implementation as production, with FP16 operands/weights,
BF16 outputs and FP32 accumulation. The shape sweep retains the original IDs
and adds Down8192, compact64 and exact M2334/M5963 row-mass quantiles from
`66d5c545` (histogram SHA`4aa422bcaddb00220f44ca5ad931328df36425f226cc5d49f21aca1e0f39d0b3`).
M8192 contains24,829,952 of37,656,000 routed rows; job counts alone are misleading.
Production selects compact16x64 below128 rows and wide128x128 otherwise.

Each resident sample submits eight identical beta0 kernels on one in-order
profiling queue, with all owners alive and no intervening transfer. It checks
the final full output against the independent closed-form oracle, all guards
and input immutability. Three warmups precede three timed bursts per route;
Main uses three independent processes per shape. Reset/readback/checking are
outside the reported burst interval. Record GPU event submit/start/end,
CPU submit/wait and host submit-to-completion wall time separately. Event
nanoseconds share a backend timebase per the [SYCL profiling contract](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:event-profiling);
do not subtract them from host clocks or claim exact hardware-cycle accuracy.

Build committed inputs with `tools/sycl/build_engine.py rebuild-gemm`, commit
the build receipt, then use Main-owned `tools/sycl/measure_gemm.py smoke` and
`perf`, committing the smoke before clean timing. The small first checks use
trace/parameter validation, and the clean sweep uses separate untraced children.
The source and all planned inputs stay frozen through the sweep.

The original sweep stopped at a profiling assertion, not a GPU fault
(`543ab5e28`). A bounded logged repeat (`0f6356d12`) found submit331ns after
the first GPU start while all GPU start/end intervals were positive and ordered.
The compiler's reported source revision, Intel LLVM`504366f4`, documents an
estimated submit clock in `sycl/source/detail/event_impl.cpp`; device start/end
are separate queries. This source corroborates a clock-domain explanation, but
does not prove every installed runtime byte came from that revision. The new
prospective service metric uses only positive ordered GPU start/end pairs;
submit-clock inversions remain recorded and disqualify submit-to-start latency
claims. No timestamp is clamped or sample silently excluded. Original failed
receipts stay failed in Git; all numerical, guard and lifetime checks remain.

This is a resident, reused-operand implementation reference. It excludes IQ3
weight dequantization, routing, SwiGLU, scatter and model transfers. Cold/warm
GPU clocks and cache reuse can differ from model execution. It is neither a
hardware peak nor a weighted prediction of model latency. Initial pageable
fixture uploads are recorded separately and do not qualify the production
pinned-memory PCIe ceiling. No model route or default is changed by this fixture.

The complete sweep at source`5eb48ed4` passed all435 invocation validations
across smoke, traced diagnostics and30 clean timing processes. Selected M8192
GU service was52.210 TFLOP/s (process medians51.896/63.963/52.210); Down was
45.668 (44.489/45.668/46.013). Keep the GU spread rather than treating its
best process as the hardware limit. The individual timed bursts and shapes
remain in [the compact result](../bench/results/2026-10-15-pr6-recovery-audit/resident-gemm-current.json).
This evidence justifies comparing resident service with actual model commands;
it does not establish that GPU arithmetic is the whole-model bottleneck.

Main's next model observation reuses the existing bounded command sampler.
Version2 qualifies positive, ordered GPU start/end service independently of the
estimated submit clock. Every retained tuple contributes its signed
start-minus-submit diagnostic; inverted samples never contribute queue-delay
sums. Group counts state how many service and queue-delay samples qualified.
Query delays, incomplete events, invalid duration/order and capacity failures
still invalidate service. Original version1 rejected receipts remain failed.
The sampler adds no GPU commands or readbacks and queries only completed events
after the existing checked drain; it samples first windows in layers0/15/31/47,
not all device work. Timestamp gaps alone do not prove device idle.

The first model check at`d9f3e91d9` exited1 before long-model arithmetic: the
legacy collector admitted only window2 and explicitly refused the production
window10 accumulator path. No fresh GPU fault occurred. The corrected route
tags its map upload, job-weight preparation, dequant, gather, GU, SwiGLU, Down,
FP32 accumulation and existing status readback, retiring the sampled events
at that existing status drain. It excludes initial router preparation and final
shared-expert/writes. Accumulation adds a fifth sampled command per selected
job; the finite cap is192 events/frame (maximum172 from32 selected jobs,
four experts and four non-job commands), with total host storage checked
against128KiB and included in admission. Requested window8/10 must really be
admitted; the observer cannot silently measure a window2 fallback. The existing
window2 collector remains separate. No arithmetic or synchronization is added.

With source and recipe committed, build using`tools/sycl/build_engine.py rebuild`.
Prepare/commit/run`qsa-kvec4-check` for the traced/validated32K-class input,
then prepare/commit/run`qsa-kvec4-census` for one official78K request with the
existing coarse, CPU, route-histogram and command observers. These profiled
requests support resource attribution; use separate clean processes for speed
comparisons. Neither run qualifies sustained generation, model task quality or
the full physical262144-context/RESTORE paths.

Both corrected model checks passed at`f111b5ff6`:39,260 and78,451 actual read
tokens, all four frames complete, peaks172/192 retained events, no fresh GPU
faults. The78K request retained678 qualified command samples and reconciled
21,514 jobs/37,656,000 routed rows with the exact historical histogram hash.
Its five matched8192-row jobs had median GU+Down service0.989792ms and
gather+FP32 accumulation1.240625ms. These are selected jobs, not a whole-model
time partition. Sampled model GU service is faster than the reused-operand
fixture, so that fixture cannot serve as a warmed hardware ceiling. Clock/cache
or duty-cycle attribution needs a separate controlled test.

The same diagnostic's nonexclusive queue intervals were QSA37.733s
(attention29.933s), MoE20.976s and weight upload8.412s. CPU time was99.944s,
of which67.579s was inside explicit waits; host grouping took0.161s. Do not
equate one busy host thread with a CPU arithmetic bottleneck. Neither queue
intervals nor requested transfer bytes establish physical saturation. Remaining
QSA attention and MoE intermediate traffic are the next tuning targets; the
actual Down epilogue and indexed-input feasibility reports are separately
reviewed candidates. Keep the [structured resource evidence](../bench/results/2026-10-15-pr6-recovery-audit/moe-window-census-current.json)
and the prior clean792.6 tok/s result distinct: this first profiled request's
735.2 tok/s is not a new clean speed comparison or inference adoption.

The opt-in QSA K/Vvec4 path keeps the qualified cached-Q/Kvec4 path fixed
and loads four adjacent BF16 V words per lane through typed`vec<uint16_t,4>`.
The existing two-byte scratch alignment is sufficient. Each vector stays in
one256-value row; missing cells write four zero words, and each original V SLM
destination retains one writer. The row-major local layout, payload, barriers,
DPAS order and FP32 online arithmetic remain the reference contract. This tests
load-instruction efficiency, not fewer useful tensor bytes or physical bandwidth.
Use the shared oracle fixture with guard offset17 (align2), tails, alias/diagnostic
cases and physical262144 component extent. Main's existing attention controller
now compares Kvec4 (variant0) with K/Vvec4 (variant1), six rotated samples per
arm on the same three8192-query shapes. Its model screening selector is separate
from the default dispatch and uses the qualified component export.

Source`4b9618071` passed those bit/oracle/diagnostic/alias/align2/boundary
checks, including the physical262144 component extent. Six rotated samples
per arm reduced median complete-call latency from11.0624 to8.7701ms
(cap33 gathered),327.2740 to230.7490ms (cap2051 gathered), and223.0425 to
174.0250ms (cap2051 clustered):20.72%,29.49%,21.98% reductions. Inputs were
immutable, outputs/guards matched, and no fresh GPU faults occurred. Timing
includes admission/prefix refresh/checked drain; it is not isolated attention
service, a bandwidth measurement or a model speed result.

For the separate model screen, startup selector
`STRATA_SYCL_PREFILL_QSA_MATH=bf16-prefix-kvvec4` reaches that exact qualified
export with the same prefix admission and scratch lifetime. Main prepares,
commits and runs`qsa-kvvec4-check`, then`qsa-kvvec4-base` and`qsa-kvvec4`
through the existing official benchmark controller. The base uses Kvec4;
both clean arms use four78K-class fresh requests, thinking on. V-route logging
must match the chosen arm. This remains opt-in pending long-model evidence
and the existing task/generation/full-context/RESTORE gates.

The engine from`5b35bee71` passed the validated/traced39,260-token check.
The clean same-binary screen then read78,451 new tokens per request, four
requests in each arm, thinking on. Kvec4 first PP was733.7 tok/s; subsequent
samples792.6,785.5,792.4 (median792.4, sample SD4.04). K/Vvec4 first PP
was786.6; subsequent853.2,847.3,853.9 (median853.2, sample SD3.63), a7.67%
median gain. Source/engine/workload/private-profile identities matched; only
the startup QSA math selector differed after normalizing private paths/ports.
Both processes exited0 with no fresh GPU faults. The
[individual samples and comparison](../bench/results/2026-10-15-pr6-recovery-audit/qsa-kvvec4-model-current.json)
retain the original baseline at`c728adad4`. One process per arm does not
establish independent-process variability;24 generated tokens do not qualify
sustained decode, task quality or physical262144/RESTORE. Keep this faster
screening route opt-in as the next prefill tuning baseline.

Before selecting a GDN scheduling change, Main prepares/commits/runs
`qsa-kvvec4-timing-check`, then `qsa-kvvec4-census` with the existing model
controller. They use the K/Vvec4 engine and existing GDN path
(`GDN_CHUNK64=0`), first32K-class then78K input, thinking on. Both enable
parameter validation, existing coarse/CPU/MoE observers and
`STRATA_PREFILL_LAYER_MAJOR_GDN_PROFILE=1`. Coarse observer v3 adds four fixed
event pairs for input projections, conv/gates, recurrence and output projection.
They retire only after the existing checked chunk drain. Every event status and
expected per-layer count is checked (36GDN/12QSA); observation failure remains
incomplete while the existing lease/Restore still owns retirement. The original
coarse v2 output and twelve event owners remain when GDN detail is off.
These nested intervals include host submission gaps and queue dependencies;
host-submit wall time is separate from thread CPU time. Neither establishes
pure GPU service or hardware occupancy. Keep sampled MoE service and clean853.2
tok/s evidence separate; observer-enabled PP is not a new speed comparison.

The first attempt (`95aec2914`) enabled legacy `STRATA_PREFILL_TIMING` and was
rejected by long-prefill admission before long-model arithmetic, exit1 with no
new GPU fault. Preserve that failed receipt. Its prohibition remains in place;
the supported coarse observer supplies the new substages instead.

Both supported observer checks passed. The full check read78,451 fresh tokens,
thinking on, with360 checked intervals per new GDN section across36 layers;
all owners closed, boot unchanged and no fresh GPU faults. On the Arc B570,
the GDN input projections, conv/gates, recurrence and output projection intervals
were6.27625,0.879437,7.99004 and2.94327 seconds. Recurrence host-submit wall
was15.4556ms. QSA and MoE intervals were31.1534 and20.9776 seconds. These
nonexclusive queue intervals identify a recurrence scheduling experiment,
not measured occupancy, a hardware ceiling or an additive wall-time breakdown.
The profiled785.0PP sample does not replace the clean853.2 median.

The next candidate partitions the independent value columns of each persistent
GDN state across2/4/8 workgroups, then performs the existing RMS/gating on the
GPU. Keep the FP32 per-column recurrence and reduction order, FP16 consumer,
shared state layout and decode/verify paths. Use the existing FP32 y buffer as
an explicitly owned intermediate, with ordered kernels and no host readback.
One partition with the separate epilogue is the control for its traffic/launch
cost. Require exact output/state/FP16 images, guards, chunk boundaries and
streamed262144/restore continuity before model speed comparisons. Recurrence
is only part of the remaining cost; this candidate alone cannot qualify1000PP.

`STRATA_SYCL_PREFILL_GDN_COLUMNS=1|2|4|8` selects this experiment only during
long layer-major prefill; unset/0 keeps the original. It cannot combine with
chunk64. For2/4/8 partitions the scan launches96/192/384 groups of256/128/64
lanes; each lane still owns32 FP32 state elements. No occupancy gain is assumed.
The intermediate adds two FP32 reads/writes per output element compared with
the fused path, before considering cache behavior. Public admission checks
registered device spans and the queue's context/device before submission.

Main builds the engine and the same-library manual fixture with
`tools/sycl/build_engine.py rebuild-gdn`, then runs
`tools/sycl/measure_gdn_columns.py smoke`, `validate`, `perf` and `context`,
committing each result before the next phase. The continuous resident timing
train rotates all five arms over three independent processes; full comparisons
run outside timing. It measures both kernels and launch gaps and excludes
initial transfer. The separate physical-extent recurrence check includes saved
FP32 carry restoration and the unchanged single-token decode consumer at the
last position; it does not qualify a full model conversation RESTORE.
Advance from resident timing only if the pooled median is at least5% faster
and each of the three independent-process medians beats the original. This is
a screening threshold, not model adoption. The first fixture compile failed
on a mixed-type`auto` declaration at`4720519ca`; the corrected build must retain
that failed identity rather than relabel it.

For the78,450 long-prefill rows over36GDN layers, the explicit recurrence
dot/state/output/delta arithmetic is15.5993444352TFLOP, excluding exponentials,
normalization and reductions. Dividing by the observed7.99004s interval gives
1.95235TFLOP/s of this logical work, not measured FP32 saturation. Separating
the RMS adds138,815,078,400 logical FP32 intermediate bytes over these layers;
actual DRAM traffic depends on caching. Column partitioning also repeats Q/K
loads, so the unchanged arithmetic count does not imply an unchanged memory cost.

The corrected same-library fixture passed traced17-token/split cases,24 cases
at1/33/65/129/8191/8192 tokens across all four partitions, and the selected
4-partition262144-token recurrence/state-restore/decode continuity check. FP32
output/carry and FP16 images matched the existing kernel bit for bit; guards,
input immutability and invalid admission checks passed with no fresh GPU fault.
This is recurrence-component evidence, not full model context/RESTORE coverage.

The three clean resident processes each measured six rotating8192-token samples
per arm after a first-use round. Pooled queue-interval medians were:

| Schedule | Median ms |
| --- | ---: |
| Existing fused recurrence | 23.693646 |
| Separate RMS,1 partition | 17.994375 |
| Separate RMS,2 partitions | 19.077656 |
| Separate RMS,4 partitions | 17.675782 |
| Separate RMS,8 partitions | 19.388281 |

The4-partition candidate reduced this interval25.40%; the1-partition control
already reduced it24.05%, so most of the gain is attributable to separating
the RMS rather than increased column parallelism. All three4-partition process
medians beat their original controls. The retained
[individual component samples](../bench/results/2026-10-15-pr6-recovery-audit/gdn-columns-current.json)
include first-use results and spread. The parameter-validated39,260-token model
check passed. The clean official benchmark then read78,451 fresh tokens per
request, thinking on, four requests per process. Existing versus4-partition
first PP was786.9 versus795.7 tok/s. Subsequent samples were854.3,845.7,853.7
(median853.7, sample SD4.80) versus867.2,858.7,867.1 (median867.1, sample
SD4.88): a1.57% median gain, or1415.7ms less median prompt time. The same
engine, helper hashes, original profile and request hashes matched; only the
column selector differed after normalizing private roots/ports. Both processes
closed without fresh GPU faults. The
[individual model samples and comparison](../bench/results/2026-10-15-pr6-recovery-audit/gdn-columns-model-current.json)
retain the original receipts in Git. Use4 partitions as the next opt-in profiling
baseline, keeping default0. One process per arm does not establish independent
process variability;24 generated tokens do not qualify sustained decode, task
quality, agent workloads, full-model262144 or conversation RESTORE. The component
gain therefore translates to a small model gain and does not qualify1000PP.

The next resource check uses the same qualified engine through
`bench_engine.py prepare|run qsa-gdncols4-warm-census`: two fresh78K requests,
thinking on, with the existing parameter-validated coarse/GDN/QSA/CPU/command
observers enabled before process startup. Exact long-request route markers
separate the records; each independently must pass all48-layer census and
observer completion checks. Retain first-use aggregate/CPU observations and
the subsequent request's detailed service samples. This tests whether first-use
CPU cost distorted the previous attribution; both speeds remain profiled,
and sampled service is not occupancy or full critical-path coverage. CPU parser
checks reproduced retained48-layer GDN and678 command observations and rejected
merged requests, missing windows/samples, incomplete or duplicate CPU headers,
and a missing GDN aggregate. The reconstructed check used synthetic census
distributions and absolute clock origins for already-retired raw logs.

The two-request check passed at78,451 fresh tokens each, with no new GPU fault
and every owner closed. Profiled PP was778.9 first-use and866.6 subsequently;
the clean867.1 median remains the speed reference. Warm QSA attention, MoE,
GDN recurrence and upload queue intervals were20.8568,20.9974,6.52363 and8.38669s.
Main-thread CPU was88.806s over89.508s wall:59.184s inside GPU waits and29.622s
outside them. The latter includes4.720s source copy,4.823s upload submission,
3.277s GEMM submission and16.630s unclassified CPU. These overlap device work
and are not an additive hardware-bottleneck breakdown. First-use unclassified
CPU/wall was19.490/26.745s versus warm16.630/16.971s; the cause remains unknown.
Next check software call-stack collection for the remaining host cost. The
[compact profile](../bench/results/2026-10-15-pr6-recovery-audit/moe-window-census-current.json)
keeps all673 selected warm command intervals. Review added exact frame/stratum
and zero-throw guards; both original log intervals passed those CPU-only checks.

Before another model run, Main checks unprivileged VTune CPU stack collection
with `profile_attention.py cpu-smoke`. It runs the installed OpenSSL SHA256
workload for3s under software sampling, with hardware characterization disabled,
then requires positive target-module CPU samples in a finite hotspot table.
Source/helpers and installed tool hashes are pinned; collection/report processes
have180s deadlines,64MiB raw-text and256MiB total budgets. This is a capability
probe, without model execution or a GPU-counter/compute-ceiling claim. No sysctl,
driver or privilege changes are part of it.

The CPU software probe was refused at collector startup by the current ptrace
policy (`ptrace_scope=1`); it produced no qualified CPU samples. Preserve the
[failed capability receipt](../bench/results/2026-10-15-pr6-recovery-audit/cpu-profiler-current.json).
The reviewed next path is disjoint non-GEMM host timing around existing launch
segments, preserving internal GEMM/transfer scopes. The current Host accounting
rejects nested Host regions, so a whole-job outer timer is invalid. That review
requires CPU-only lifetime/exception/accounting cases before separate traced
and untraced bounded model diagnostics. No speedup follows
from this source review.

The next diagnostic revision adds version4 CPU categories only to the mode10
expert window: map upload, job-weight preparation, gate/up and down dequantization,
gather, SwiGLU, accumulation and status readback. Each wraps an existing disjoint
host segment; internal GEMM accounting remains unchanged. Explicit GPU waits
inside a category are subtracted. These regions cover all layers with command
tracing off or on; no device kernel, transfer, synchronization or inference order
is added. Pre-job span/layout checks and other engine work remain unclassified.
Version3 and version4 unclassified time therefore describe different partitions.

Before the engine rebuild, `tools/sycl/check_host_profile.py` compiles and runs
the CPU-only fixture from committed inputs. It checks disabled/TLS ownership,
all17 region counters, nested wait subtraction, idempotent closure, exceptions
and invalid nesting. The parser must reject incomplete, duplicate, wrong-version,
nonzero-throw and inconsistent-partition receipts. Preserve the known-good engine
and same-library fixture outside the reused build root; the builder now requires
that hashed fallback before overwriting outputs.

The model sequence is `bench_engine.py prepare|run` with
`qsa-gdncols4-host-check`, then `qsa-gdncols4-host-traced-check`, then
`qsa-gdncols4-host-census`. The first two each use one32K nominal input
(actual token count recorded), thinking on, parameter validation and CPU/coarse
observers, with command tracing off then on in separate processes. The last
uses two fresh78K requests to distinguish first-use and subsequent CPU costs.
Every version4 category count must match the48-layer planned-work census:
one map/job-weight/status region per window, one dequant pair per active expert,
and one gather/SwiGLU/accumulation per job. Retain compact qualification receipts
and retire successful full logs/CPU fixture binaries once their consumer closes.
The model counters have not yet run; this is not a speedup, hardware saturation,
task-quality, physical262144 or RESTORE qualification. Observer overhead remains
unqualified until compared with a matched clean cohort.

The CPU-only gates passed at`fd0c38e71`: six lifetime/accounting cases and eight
invalid-case/parser rejections, with all owned processes closed. This does not
execute the model. The current qualified engine and GDN fixture were hash-checked
and copied before reusing the build root; all488 project,3608ggml and5703TLA
original source inputs were recovered and verified from their recorded commits.
The global inventory also still contains legacy artifacts under an unfinished
provenance audit. They are not qualified comparison builds; the current build
receipt records that exception and its next review after bounded model admission.
No global storage-cleanup completion is claimed.

Both command-tracing off/on model gates passed at39,260 fresh tokens, thinking
on,24 generated tokens, with no fresh fault and every owner closed. Per request,
48 map/job-weight/status regions,17,580 dequant pairs and18,796 gather/SwiGLU/
accumulation jobs matched the actual census. Off/on status-region CPU outside
explicit waits was3.884881/3.886876s; unclassified CPU was5.282371/5.341875s.
The status segment includes SYCL memcpy submission and runtime checking, so
this does not establish3.9s of host arithmetic: an API can wait internally.
First-use profiled PP was575.2/614.9; differing cold wall gaps prevent a speed
claim. The traced run qualified668 service intervals, but17 submit-clock
inversions prevent an absolute submit-to-start timing claim. Full per-layer
histograms are verified then reduced to conserved work totals for these admission
checks. Next use the committed two-request78K plan for warm attribution.

The final two-request check passed at78,451 fresh tokens per request, with
profiled PP796.0 first-use and865.1 subsequently, zero fresh GPU faults and
all ownership closed. The workload hashes match the earlier warm resource run.
Warm main-thread CPU was88.805216s over89.659797s wall, with59.173520s inside
explicit waits and29.631696s outside. Version4 attributes12.441650s to the new
MoE categories, leaving4.204023s unclassified; the same request's version3
definition would leave16.645674s. This is attribution, not time saved.

| Warm CPU outside explicit waits | Seconds |
| --- | ---: |
| Source copy | 4.769362 |
| Upload submission | 4.803936 |
| GEMM submission | 3.240548 |
| Dequant GU/down, gather, SwiGLU and accumulation submission | 6.845040 |
| Status wrapper | 5.567415 |
| Unclassified | 4.204023 |

The table omits smaller categories and overlaps device execution. The status
wrapper includes `attempt()` preflight runtime checks before both memcpy and
synchronize, queue/context lookup, submission, observer and post-check work.
Its5.567s cannot be attributed solely to memcpy or to host arithmetic. Ordinary
host pointers are legal SYCL memcpy arguments; the contract alone does not prove
whether the loaded adapter blocks internally. A future discriminating probe
must time those pieces separately and preserve all asynchronous error checks.

Warm QSA attention, MoE, GDN recurrence and upload queue intervals were20.8553,
21.0035,6.52502 and8.40041s. Requested layer-weight upload payload was50,292,326,400
bytes; payload divided by this nonexclusive upload interval is5.987GB/s,
not a physical PCIe counter measurement. GPU saturation and full critical-path
coverage remain unproven. The clean867.1 tok/s median remains the speed reference.

For the four sampled expert windows only, expanding each row-bin sample mean
by its eligible command count gives0.149864s Down and0.265763s accumulation
service, with0.205840s gather and0.278878s GU. These are descriptive screening
estimates, not full-model totals or statistical bounds. The next component
candidate is a Down scatter/FP32 accumulation epilogue using the existing
reviewed contract: retain the actual BF16 Down rounding before FP32 FMA, unique
writers, legal element/tail ownership and checked event/owner retirement. Do not
change inference order or treat it as adopted before same-library correctness
and matched component/model timing pass. Existing source-specific fusion and
indexed-A references are available; no repeat generic survey is needed.

## Ordered parallel exponentials for QSA

A separate component candidate adds `qsa_bf16_prefix_ordered_exp_batch` beside
KVvec4, without a model dispatch hook. It keeps cached BF16 Q, typed K/V loads,
score pitch33, probability rounding and FP32 recurrent updates. SIMD16 lanes
compute two exponentials each. Every lane executes32 uniform indexed exchanges;
lane0 sums cells0..31 in the original ascending FP32 order. The existing subgroup
max/invalid-score reduction supplies the maximum. No additional workgroup
barrier, SLM array, host arithmetic or transfer is added to the normal route.
Shuffle latency, register effects and speed remain unmeasured.

The implementation follows the independently authored existing Strata kernel
and the reviewed order-preserving QSA design indexed under
`qsa_ordered_parallel_softmax_scope`; no third-party implementation is copied.
The pinned compiler's public `select_from_group` API supplies lane exchange.
The diagnostic-only `QsaPrefixStateProbe` records one query/KV group/head's
maximum, alpha, normalizer and tile sum per32-cell tile, including padded heads.
Its full device extent, alignment and disjointness are checked before either
kernel submits. Non-probe entrypoints have no capture writes. Captured outputs
must match normal outputs before interpreting state differences.

Main first closes the frozen Down comparison; this source was prepared in an
isolated child of `feature/sycl`, without another build/GPU job. After integration,
reuse the closed candidate cache with `build_engine.py rebuild-ordered`, preserve
the validated fallback, and run `check_attention.py smoke|component|perf ordered-exp`
one phase at a time, committing each result before advancing. The first small
GPU phase uses tracing and parameter validation. Require complete BF16 output,
FP32 state/guard and input-preservation comparisons, invalid state-span/coordinate
rejections, width/tail/missing/duplicate/page-alias diagnostics and physical262144
component addressing. A component check does not qualify full-context model,
RESTORE, numerical quality or sustained generation.

The prospective clean timing compares actual KVvec4 and ordered-exp exports in
three independent processes, six rotated measured samples per arm/shape, at
78453 cells/8192 queries with cap33 gather and cap2051 gather/clustered patterns.
Timing includes admission, pack, attention and checked drain, with checks/copies
outside. Main's finite budget is180seconds,360CPU-seconds,4GiB RSS and64MiB output
per child. Advance only for geometric-mean candidate/base ratio<=0.95, no shape
ratio>1.05, and every process faster in both cap2051 patterns. Passing permits a
later matched official long-input model A/B; it is not inference adoption or a
hardware saturation claim.

Source`9d000172f53c399453a0662c3eac2b0771114e7e` passed the traced smoke and
all11 numerical/boundary cases per arm. Ordered-exp also matched the serial
FP32 maximum, alpha, normalizer and tile-sum bits, including padded heads and
unused capture guards; all9 invalid probe admissions rejected. No fresh GPU
fault or boot change was detected. The initial failed smoke remains archived
at`963d387698f8b30b8fff9229449775000bdef4b8`: its invalid-overlap test pointed
at the legal leading guard, and the corrected test points inside the output.
This was a fixture error, not a relaxed kernel admission or numerical criterion.

Three independent clean processes then completed all108 individual samples:

| Cap / access | KVvec4 median ms | Ordered-exp median ms | Latency change |
| --- | ---: | ---: | ---: |
| 33 / gathered | 8.792280 | 8.618858 | -1.97% |
| 2051 / gathered | 231.586801 | 224.028323 | -3.26% |
| 2051 / clustered | 174.369835 | 139.799954 | -19.83% |

Every process improved both cap2051 patterns. The geometric-mean latency ratio
is0.912692, satisfying the prospective component gate. The
[individual results](../bench/results/2026-10-15-pr6-recovery-audit/qsa-ordered-exp-current.json)
retain spread and process medians. Proceed to an explicit opt-in long-prefill
model route and a matched official78K comparison; no new model throughput or
sustained decode, task-quality, full-context or RESTORE qualification is claimed.

`STRATA_SYCL_PREFILL_QSA_MATH=bf16-prefix-ordered-exp` explicitly selects the
qualified kernel in dedicated long prefill. It uses the same prefix allocation,
resident-pool admission, strides and owner lifetime as KVvec4; the component
kernel and default selector are unchanged. Startup logs identify parallel exp
with ascending sum, cached Q and typed4 K/V. Short prefill and decode keep their
existing route.

Main rebuilds from committed inputs, then prepares/commits/runs
`qsa-ordered-exp-check`, `qsa-ordered-exp-base` and
`qsa-ordered-exp-candidate` using `tools/sycl/bench_engine.py`. Preparation
requires the recorded component gate and unchanged kernel source. The first
model gate uses validation/trace on a32K nominal input. Both clean speed arms
use the same binary, GDN columns4, thinking enabled and four fresh78K requests;
only the startup QSA selector differs after normalizing private paths/ports.
Each later plan requires its matching completed predecessor committed. Retain
all requests, separate first-use from subsequent speed and archive prior gates
by immutable commit. This is a model screen, not independent-process variance,
sustained generation or default-adoption qualification.

The opt-in source`ce1e950a843953c7856382978f7993bcc848e366` passed its
39,260-input validated model admission, thinking enabled, without fresh GPU
faults. The subsequent same-binary clean screen used78,451 fresh tokens on
every request and matched benchmark source, request hashes, profile seed and
normalized environment/configuration. Both processes exited normally.

| QSA route, GDN columns4 | First-use tok/s | Subsequent tok/s | Subsequent median / sample SD |
| --- | ---: | --- | ---: |
| KVvec4 | 789.4 | 867.8, 859.5, 867.4 | 867.4 / 4.68 |
| Ordered-exp | 816.9 | 902.9, 893.5, 900.6 | 900.6 / 4.90 |

The matched subsequent-median gain is3.83%. Keep ordered-exp as an explicit
long-prefill tuning candidate. This is one process per arm with three subsequent
requests, not independent-process variance. All individual requests and the
immutable baseline reference are in the
[current model comparison](../bench/results/2026-10-15-pr6-recovery-audit/qsa-ordered-exp-model-current.json).
The24 generated tokens do not qualify sustained decode; quality, full-model
262144 context and RESTORE remain open. Neither1000 PP nor70 TG is achieved.
The next source-backed candidate is UR V2 submission batching, conditional on
bounded queue/event/profiling acceptance and preservation of the existing
copy-offload avoidance path; the indexed review does not qualify running it.

At comparison closure, all component samples and nine official model requests
were checked against raw output, and the original build inputs were recovered
from Git before retirement. The current ce1 engine and three qualified QSA
fixtures remain. The older bb8 fallback/GDN fixture, closed QSA logs/caches and
four completed audit report/review payloads were retired under the
[deletion manifest](../bench/results/2026-10-15-pr6-recovery-audit/build-retirement-20261018.json).
Before another rebuild overwrites this selected engine, preserve and hash-check
it outside the reused cache and commit a new fallback reference. The six legacy
provenance holds remain unresolved; this is not global storage cleanup.

## Submission batching acceptance

The pinned UR Level Zero V2 provider exposes the startup-only experimental
`UR_L0_V2_FORCE_BATCHED` selector. The source audit in the research index confirms
that common command-list creation still honors the existing copy-offload
disable flag, but regular batched-event profiling is an unresolved compatibility
question. Current engine streams explicitly request in-order execution and
profiling. Do not infer acceptance from the environment variable or ELF strings.

The maintained compatibility fixture's explicit `--submission` mode stops before
graph recording. It checks two-stream copies/order, explicit/default-stream
elapsed times, direct kernel profiling, integer/guard results for1/119/120/121/241
queued kernels, ordinary-RAM copy lifetime and teardown. `sycl_submission_test`
is a manual target; the default/native compatibility tests keep their graph
coverage. No inference math or production queue property changes here.

Main first preserves the selected engine, commits source/recipe/controller and
runs `build_engine.py rebuild-submission`. Then run/commit
`check_submission.py control` and only after it passes run
`check_submission.py candidate`. Separate owned processes keep the selector
unset versus1, set all runtime variables before initialization, and retain
copy-offload disable1. Both phases use tracing/parameter validation and finite
90-second/120CPU-second/2GiB-RSS/64MiB-output budgets. All10 fixture cases and
normal ownership/fault closure must pass before a forced-batched model run.
This gate measures compatibility, not speed or hardware saturation. A failed
profiling/synchronization contract prevents model advancement without changing
the900.6tok/s route or weakening its existing API.

At source`dfba48ef52e1bd8df596ecda06ab4eae0c882359`, the ordinary control
passed all10 cases. The forced-batched candidate passed the initial copy/order
case, then returned`UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE` for the first
explicit-stream elapsed-time query. The remaining candidate cases did not run.
Both processes closed without a fresh GPU fault or boot change. Reject this
setting for the current profiling-enabled runtime; no forced-batched model or
speed measurement follows. The engine executable is byte-identical to the
900.6tok/s reference. Keep the
[failed gate and compact diagnostic](../bench/results/2026-10-15-pr6-recovery-audit/submission-acceptance-current.json).
The selected next implementation is the interleaved GU/SwiGLU epilogue seam;
prove actual fragment partner ownership before running its GPU kernel.

## MoE GU/SwiGLU component

The first gate is host-only inspection of the actual pinned TLA matrix and
output-copy layouts. `build_engine.py rebuild-gu` builds the manual
`sycl_moe_gu_swiglu_bench` target. `measure_gu_swiglu.py ownership` checks both
compact16x64 and wide128x128 policies at1/3/17/33/129/2334/5963/8191/8192 rows,
including masked tails. Each logical GU value and hidden output must have one
owner. Adjacent interleaved gate/up columns may use same-index lane-XOR1 exchange
only if every pair in that layout passes. This mode creates no Runtime, queue or
USM allocation and submits no GPU work; its90s/120CPU-second/2GiB-RSS/64MiB-output
budget is independent of later numerical and performance gates. No model path
selects a fused GU kernel at this stage.

The original ownership gate at source`b3e982d93` passed all36 cases, with zero
adjacent-pair mismatches for both layouts/policies (receipt at`3ea300e5d`). The
next explicit component API uses the matrix layout and packs two already-rounded
BF16 fragment values into one32-bit subgroup exchange. Adjacent lanes compute
different fragment indices, balancing the hidden writers. A new host gate must
prove this packed mapping before any GPU test, followed by a384-work-item check
of the actual local/subgroup index relationship. This is newly authored Strata
code using the existing licensed DPAS helper and public TLA/SYCL APIs.

Run/commit `measure_gu_swiglu.py ownership`, then `smoke`, then `validate`, then
`perf`. Smoke uses tracing/parameter validation and checks both tile policies,
finite/rounding/NaN/saturation inputs and13 rejected admissions. Full validation
adds1/17/33/129/2334/5963/8191/8192-row cases for both policies. BF16 diagnostic
capture must match dense DPAS output exactly; hidden output preserves finite
FP16 saturation and NaN classification, with a prospective maximum1 FP16 ULP
bound across the existing JIT activation and new AOT epilogue. Small finite
cases also use an independent integer dot-product oracle. Check guards and input
immutability for both capture and no-capture specializations.

Clean timing uses three independent processes, rotated separate/fused arms,
five measured rounds plus warmup,16 submissions per burst, at64/160/2334/5963/8192
rows. Keep every measured sample and both host wall time and queue intervals;
an interval includes launch bubbles and does not establish GPU busy time.
The prospective component gate is wall-ratio geomean<0.98 with every large
shape/process ratio<=1.02. Any model integration still requires long-input
matched measurements and the broader correctness gates. The eliminated logical
GU write/read is5120bytes per row,40MiB for8192 rows; this is an operation count,
not a measurement of physical VRAM traffic or a proof of a bandwidth bottleneck.

At source`5ca094afc`, the packed ownership36 cases, traced smoke4 cases and13
admissions, full18 cases and all timed outputs passed without a fresh GPU fault.
BF16 captures were exact and the observed FP16 difference was0 ULP, including
the no-capture specialization. NaN payload bits are not an adoption criterion;
the finite hidden output must remain saturated and within the stated bound.
The native Luna-max review corrected a research false positive: the all-slot
partner loop, XOR involution, parity and packed writer census also prove sloti+1.

| Rows | Fused/separate wall ratio, median of3 process ratios | Decision for next model trial |
| --- | ---: | --- |
| 64 | 0.85950 (range0.67026–0.92128) | Exclude because of wide timing spread |
| 160 | 1.07211 | Retain separate GU/SwiGLU |
| 2334 | 0.97321 | Eligible large-job candidate |
| 5963 | 0.94814 | Eligible large-job candidate |
| 8192 | 0.92754 | Eligible large-job candidate |

The component wall-ratio geometric mean was0.94205. This is a component result,
not model throughput. Preserve the samples in the
[current component result](../bench/results/2026-10-15-pr6-recovery-audit/moe-gu-swiglu-current.json).
The explicit startup selector`STRATA_SYCL_PREFILL_GU_SWIGLU=large` is a model
experiment: use the wide fused kernel only for jobs>=2334 rows in the existing
long-input accumulator window. Smaller jobs keep the separate operators.
Speeds between the sampled sizes remain unmeasured; actual model timing decides
this threshold. Default, short-input prefill and decode do not select this API.
Existing command-service census assumes a separate activation command, so the
new selector rejects that observer until fused-command coverage is qualified;
the ordinary parameter validation and trace path remains available.

Run/commit `bench_engine.py prepare qsa-gu-fusion-check`, then its matching
`run`. Only a passed>=32K actual-input admission permits the corresponding
`base` then`candidate` phases. Both clean arms use the same engine, ordered-exp
QSA, GDN columns4, matched official`bench.sh` requests and thinking enabled;
only the GU selector changes. Each arm records first-use plus3 subsequent
fresh>=64K requests and actual fused/separate job counts. Repeated independent
cohorts are still required if variability prevents an adoption decision.

The matched78,451-token comparison used engine
`89895909e04a811b2b34ba2f9eca957718086c13df2f669726b8182368482e02`
at source`3d8708758`. Six independent native processes ran in the committed
order A0/B0/B1/A1/A2/B2, with one first-use and three subsequent fresh inputs
per process, thinking enabled. Every input/profile/configuration/runtime and
executable matched after normalizing private output paths, ports and the GU
selector. Recovering each original source snapshot also verified that the
executed benchmark, server and workload helpers matched; the cohort0-to1
extension changed only the owner/receipt controller.

| Pair | Separate first-use / subsequent median tok/s | Fusion first-use / subsequent median tok/s | Subsequent median gain |
| --- | ---: | ---: | ---: |
| 0 | 814.8 /898.2 | 819.2 /903.5 | +0.5901% |
| 1 (reversed) | 816.0 /899.8 | 819.0 /903.2 | +0.3779% |
| 2 | 816.4 /900.0 | 816.3 /902.9 | +0.3222% |

All nine matched subsequent inputs favored fusion. Their geometric-mean speed
ratio was1.00566723 (+0.5667%), passing the prospective requirement of all three
paired medians improving and geomean>1.002. Keep the original samples and spread:
for example, A1's subsequent samples were900.3/883.6/899.8 with sampleSD9.50.
First use is excluded from selection; the third pair did not improve first use.
All six processes passed eligibility/closure checks without a fresh GPU fault
or boot change. At78K, fusion handled about5,000 jobs and33.9million rows per
request; smaller jobs remained separate.

`python3 tools/sycl/bench_engine.py summarize-gu` verifies the six committed
receipts, recovered source hashes, request bytes, normalized settings, actual
route counts and independent native identities before recording the decision.
The [current compact comparison](../bench/results/2026-10-15-pr6-recovery-audit/moe-gu-swiglu-model-current.json)
keeps all24 prompt-speed samples and links the full original receipts in Git.
Retain the explicit large-job candidate for further qualification. The default
selector stays off:24-token generation does not qualify sustained decode,
broad task quality or physical262144/RESTORE. The1000tok/s prefill and70tok/s
decode goals remain unmet.

The historical runtime check covers the then-configured `runtime_elfs`, not
a complete loaded-library census. A later offline dependency check found the
four linked MKL libraries absent from that set; subsequent settings pin them
and the libdl/libpthread shims as well. This does not retroactively add per-launch
hash evidence to the six finished processes or change their original status.

## MoE Down/scatter component

`Gemm::down_scatter` is an explicit component API with no model dispatch hook.
Its separately named final-fragment sibling retains the licensed vLLM-XPU BSD
DPAS loop; the original dense helper definition is preserved verbatim from
`7829568c6`. FP16 X/weights have the fixed production Down shape2560x640.
The new epilogue uses the actual TLA `partition_C` coordinates, BF16 conversion
on the accumulator's unchanged thread/value layout, then explicit FP32 FMA.
Logical tails are masked before map reads. The published immutable expert plan
provides unique token rows per job; the original in-order queue serializes
contributions from different jobs/experts. Complete registered device spans,
non-aliasing and the original block2D input alignment are checked before enqueue.
A diagnostic-only BF16 capture is optional; the null-capture kernel materializes
no Down tensor. These contracts do not establish speed or model qualification.

Source`7fbe795ce` passed18 exhaustive CPU tile/element ownership cases for both
policies at1,3,17,33,129,2334,5963,8191,8192 rows; one traced seven-result small
GPU gate; and22 tail/split/irregular/rounding cases. BF16 captures and FP32
accumulation matched the separate reference bitwise, including exceptional
inputs and invalid-map status. The small finite cases also checked an independent
integer-dot/RNE oracle; nine invalid admissions rejected. All processes closed
normally without a fresh GPU fault or boot change. Main uses the maintained
[fixture](../bench/sycl/moe_down_scatter.cpp) and
[phase controller](../tools/sycl/measure_down_scatter.py).

Three independent resident processes measured six rotated rounds per arm/shape,
16 repetitions per sample, at64,160,2334,5963,8192 routed rows and both contiguous
and every-other-token layouts. The baseline uses the same tile as the candidate:
compact below128 rows, wide otherwise. Queue-marker intervals include host gaps;
they are not device busy, physical traffic or a hardware ceiling.

| Routed rows | Contiguous separate/fused ms | Every-other-token separate/fused ms |
| --- | ---: | ---: |
| 64 | 0.045553 /0.045326 | 0.022663 /0.022998 |
| 160 | 0.028639 /0.051579 | 0.028363 /0.051549 |
| 2334 | 0.337158 /0.397920 | 0.369938 /0.418913 |
| 5963 | 0.810121 /0.879896 | 0.929319 /0.855455 |
| 8192 | 1.112955 /1.189010 | 1.337500 /1.127850 |

The ten-shape geometric-mean latency ratio is1.148080, so the unchanged
prospective gate failed. That gate requires ratio<=0.95, no shape ratio>1.05,
and every process to improve at2334/5963/8192 for both layouts. Irregular8192
improves15.67%, but160 and2334 regress in every process. **Do not connect this
variant to the model.** Individual samples, process medians and limits are in
the [original raw-fragment result](https://github.com/yqYo1/Strata/blob/16b697d9c2c5a699c8e7250e7a04d2b5cfbf15b0/bench/results/2026-10-15-pr6-recovery-audit/moe-down-scatter-current.json).
The clean model reference remains867.1 tok/s; this component did not change it.
Physical262144, model quality, RESTORE and decode remain separate gates.

For the earlier7fbe795ce build, offline inspection of the retained known-good
and rebuilt engine native images
found identical instruction streams through EOT and identical complete per-kernel
`.ze_info` blocks for all eight existing dense entries. Nine bytes per entry
differ after EOT in null-destination immediates/padding; no branch targets enter
that region and no unresolved indirect transfer was found. The other eight
existing native text entries matched directly; four new scatter entries were
added. The [original build result](https://github.com/yqYo1/Strata/blob/16b697d9c2c5a699c8e7250e7a04d2b5cfbf15b0/bench/results/2026-10-15-pr6-recovery-audit/native-gdn-engine-build.json)
records hashes and exact comparison scope. This checks those native calculations
and recorded kernel argument information, not total host behavior or model speed.

The reviewed next discriminator is an explicitly constructed BF16 output-copy
TV layout. The current descriptor is an identity tensor, so the copy layout and
its paired coordinates need a new ownership/equality proof; `partition_C` tests
alone do not establish it. Keep the separate two-stage path as performance
anchor and raw scatter as diagnostic control. Do not change metadata hoisting
or alias declarations in the same experiment. Also predeclare a representative
working-set comparison: this fixture's accumulator contains T or2T token rows,
whereas the measured model window has78451. Row sparsity and VRAM footprint
must be matched before a model-relevance claim. Evaluate tile choice separately.
Logical byte removal and one fewer launch did not establish a speedup, and the
shape-dependent timings do not yet establish their hardware cause.

After recording those results, Main retired221 closed-run, CPU-fixture,
consumed-review and old-fallback files (61,391,969bytes). Original run helpers
and the fallback's9799 source inputs were retrieved and hash-verified from Git
before deletion. Keep the new runtime-qualified engine and its current GDN
fixture. Older legacy build provenance remains an explicit separate exception;
the six-root review has completed, with no exact final-build source proof for
any of those roots. Each binary matches its own receipt, but seed links,
similarly named proofs and declared commits do not establish recovery of the
final source, recipe and dependencies. Keep the six provenance holds until
their recovery steps are verified. The review identity and unresolved outcome
are recorded in the [current build result](../bench/results/2026-10-15-pr6-recovery-audit/native-gdn-engine-build.json);
this is not a global cleanup claim or authorization to retire those roots.

## Optional GDN chunk64

`STRATA_SYCL_PREFILL_GDN_CHUNK64=1` adds a separate experimental long-prefill
route with native BF16 projection/conv/recurrence activations, FP32 gate/beta,
carry and shared convolution history, and a genuine FP16 projection consumer.
It retains the loaded16/48 head order. The lifetime-shared8192 arena fits the
existing GDN reservation, preserving the MoE-window8192 admission. See the
[maintained contract and fixture](../bench/sycl/gdn_chunk64/README.md).

The same-binary clean official benchmark read78,451 fresh tokens per request,
thinking on, four requests per process. Existing versus chunk64 first PP was
682.9 versus677.4 tok/s; subsequent medians were725.3 versus725.8 tok/s. This
0.07% difference establishes no material model speed gain despite the earlier
22% resident component gain. Both processes exited without fresh GPU faults.
Keep the existing GDN selected for performance. The new route remains opt-in;
its full64 solve differs from the SGLang XPU block16 contract, and numerical/task
quality, sustained generation, physical262144 and RESTORE are still unqualified.

The next source candidate adds an explicit `ScatterLayout::output_copy` parameter
without a model hook. It constructs the pinned TLA public
`make_block_2d_copy_D<cutlass::bfloat16_t>(mma, stride)` layout without a global
output pointer. Values use `partition_sg_fragment_S`, paired coordinates use
`partition_S`, and the same TLA reorder as dense BF16 output precedes the
unchanged scalar FP32 scatter/FMA. The original matrix mapping remains a
diagnostic arm. No tile, metadata-hoisting or alias-policy change is combined.

Before measurement, the new plan requires36 host ownership cases,14 traced
small checks and52 GPU correctness cases across both layouts and tiles,
including fixed78451-token windows. Only after those pass, three independent
processes compare separate, raw-fragment and output-copy arms at64/160/2334/5963/8192
expert rows, contiguous or evenly distributed over the same78451-row accumulator.
Six rotated measured rounds and one warmup use16 repeats; every timed sample
gets a full bitwise reference comparison outside timing. Allocations, resets
and readback stay outside marker intervals. Main supervises each process for
900seconds (other phases600),6GiB host RSS and128MiB raw output. The prospective
advance gate remains geometric-mean latency ratio<=0.95 against the separate
reference, no shape>1.05, and each process improves at2334/5963/8192 in both
placements. Passing permits a later model experiment, not inference adoption.

Source7ad7aad7ad7399ba54ce3aaa5171b0298370e117 passed all36 CPU ownership,
14 traced small and52 GPU correctness cases. Captures and normal kernels
matched BF16 outputs and FP32 accumulation bitwise, including guards and invalid
maps. The three independent performance processes also completed normally,
with540 measured and90 warmup samples and no fresh GPU fault or boot change.

| Routed rows | Contiguous separate / raw / output-copy ms | Distributed separate / raw / output-copy ms |
| --- | ---: | ---: |
| 64 | 0.027161 / 0.025632 / 0.025791 | 0.026423 / 0.025879 / 0.025898 |
| 160 | 0.030964 / 0.054749 / 0.054603 | 0.030911 / 0.055638 / 0.055446 |
| 2334 | 0.338083 / 0.399388 / 0.397962 | 0.367198 / 0.419964 / 0.418617 |
| 5963 | 0.798512 / 0.848633 / 0.847256 | 0.847103 / 0.937369 / 0.934727 |
| 8192 | 1.087380 / 1.127760 / 1.125495 | 1.160900 / 1.225910 / 1.224640 |

The output-copy/separate geometric-mean latency ratio is1.175797. Every process
is slower at2334/5963/8192 in both placements, so the predeclared advance gate
fails. **Keep the separate engine route; do not connect either fusion variant.**
The [current individual results](../bench/results/2026-10-15-pr6-recovery-audit/moe-down-scatter-current.json)
preserve the raw control, spread and every sample. This repeated same-buffer
component regime does not identify physical traffic, saturation or model speed.
At the close of this Down comparison, the clean model reference was867.1 tok/s;
the later ordered-exp model screen above reaches900.6 tok/s.

Bounded offline ELF inspection of the actual7ad engine also found125504/126912
byte native wide no-capture scatter sections, versus8128 bytes for dense
FP16-to-BF16 wide GEMM. Compact scatter sections were15936/15680 versus3072
bytes dense. Raw and output-copy code hashes differ. This is compiled code-size
evidence, not proof of instruction-cache pressure, occupancy or a slowdown
mechanism. Before another fusion variant, investigate grouping repeated row
metadata/predicates while preserving static fragment access; a dynamic register
index or restricted pointer must not be assumed faster or legal without checks.
The QSA ordered-exp component and opt-in model screen subsequently passed their
gates as described above. Neither Down fusion variant is selected.

### Next Down discriminator: workgroup row metadata

The selected next component hypothesis stages each logical M row's validated
token and weight once per workgroup, independently of the TLA fragment layout.
It targets repeated global metadata loads, address calculations and predicates
in the rejected scalar epilogue; those source operations and the compiled code
size do not establish a measured resource bottleneck or predict a speedup.
The GU model comparison closed before this implementation; its original
source and individual samples remain reachable in Git.

Use the existing matrix mapping and explicit opt-in component API. Local IDs
cover the16 or128 M rows without overlap, including a masked final M tile.
Validate the slot/token before loading a weight, OR the existing error bit for
an invalid map, and publish an invalid-token sentinel. After one unconditional
workgroup local-memory barrier, the unchanged DPAS helper and BF16 conversion
feed the original per-cell FP32 FMA using cached token/weight. All work-items
must reach the barrier. No subgroup row-ownership assumption, dynamic fragment
index, global metadata slab, alias promise or barrier removal is included.
Two local arrays require128 or1024 bytes per workgroup; query the device limit
and retain all current span, plan lifetime and disjoint-writer checks.

Before implementation, an independent source-only review must assess the
barrier, tails, error semantics and numerical contract against the pinned SYCL
and TLA interfaces, and compare counterarguments such as static factoring or
already-commoned loads. Main owns implementation, codegen inspection and tests.
Keep the existing separate and scalar-fused controls. Require exact captures,
FP32 accumulation, guards, immutable inputs and invalid-map outcomes before
three clean independent processes measure the fixed78451-row window at
64/160/2334/5963/8192 expert rows in both placements. The existing advance gate
(geomean latency ratio<=0.95, no shape>1.05, each large shape improves in every
process) remains prospective. The extra barrier or SLM accesses may regress;
reject the variant if it fails. Component success alone permits only a later
model trial, not adoption or a hardware-saturation claim.

The independent source-contract review requires BF16 diagnostic capture to
remain before the invalid-token skip, including invalid-map cases. Status is
the final OR-ed bitmask, not an atomic-call count: a bad slot sets bit1 and
suppresses that row's token-error branch; bit2 requires a valid slot. Add a
combined bad-slot/bad-token case and require exactly bit1. This review found
the cache conditionally legal; it provides no new runtime or speed evidence.
Use the pinned TLA revision87f6850680a580654b9ea2c80dbc01aeb36ad231 when
resolving its interface references.

The component implementation exposes `ScatterLayout::matrix_row_cache`; no
engine route selects it. The original matrix and output-copy kernels remain
separate specializations. Its host ownership check also counts every local
metadata slot and checks each actual TLA coordinate against its workgroup's M
interval. The published plan, maps and weights stay immutable through the
checked drain. Typed metadata/output alignment is rejected before submission.

Before its first GPU run, source review found that this fixture's borrowed
queue was not drained by `Allocation` teardown: `Runtime::wait()` covers only
the two built-in queues. A synchronous submission/admission exception could
therefore unwind owners while earlier borrowed-queue work remained pending.
Explicit checked drains now precede Case and Buffer member destruction; a
failed drain exits without releasing those owners. This is a source-level
lifetime fix, not a demonstrated cause of an earlier GPU fault.

The new fixture requires54 host ownership cases,33 traced small-check events
(including16 admission rejects,7 invalid-map cases and synchronous-exception
retirement per policy), and78 full
correctness events across the three policies and both tiles. It verifies full
input bits/guards, diagnostic M tails, null-capture non-writes, combined
bad-slot/bad-token precedence and a mixed-row error mask of3. Clean timing
retains three arms: separate, scalar matrix fusion and row-cached matrix fusion;
output-copy remains a correctness diagnostic. All checking/readback is outside
the reported bursts. `rebuild-down` also rebuilds the GU fixture because the
component shares its adapter translation unit; the validated prior engine and
same-build GU fixture are preserved before overwrite. These are prospective
performance gates, not a new model performance result.

At source`c9ad511e4f3ec8387327e6c86d8e6a9e02988eac`, the54 CPU ownership,
33 traced small-check and78 full correctness events passed. All three policies,
tails, capture/non-capture, immutable inputs, invalid maps and checked exception
retirement completed without a fresh driver fault. The current
[component result](../bench/results/2026-10-15-pr6-recovery-audit/moe-down-scatter-current.json)
records these gates. Three independent clean timing processes subsequently
closed normally, with630 successful timing events (90 warmup,540 measured),
no fresh driver fault and unchanged boot. The original all-shape advance gate
failed: the geometric-mean latency ratio is0.987055, above the required0.95,
and160-row shapes exceed1.05. No model route selects the cache.

| Expert rows | Row-cache/separate ratio, contiguous | Distributed |
| --- | ---: | ---: |
| 64 | 0.99792 | 0.86405 |
| 160 | 1.37682 | 1.37339 |
| 2334 | 0.95136 | 0.98721 |
| 5963 | 0.87003 | 0.91436 |
| 8192 | 0.83246 | 0.86564 |

Every2334/5963/8192-row shape improves in each of the three processes. Their
exploratory six-shape geometric mean is0.901956; it does not turn the failed
all-shape gate into a pass. The next candidate should retain separate Down
and accumulation below2334 rows and use the cache only for larger jobs.
Commit that routing and its threshold/tail/diagnostic checks, then define a
fresh component/model qualification plan before execution. This is a new
candidate, not a retrospective threshold or a selected model route.
The compact result retains every measured queue and wall sample, spread,
original decision, checked outcomes and exact source/fixture identity; nominal
case payloads and unused warmup durations are retired after verified extraction.

Offline AOT inspection of engine`f004aabae84dbea46db59bdf98ec0413a8bf93b53c13e9b5b28e5c2ed045fc86`
and fixture`e28d402742cfceca3d238142e2de1065b8bd1d6c0ca553ad87feb3052b8186e3`
found identical native no-capture code between those two artifacts for both
policies/tiles. Scalar matrix versus row-cache text is15680/11648 bytes
(compact) and126912/67840 (wide). All selected metadata entries report
256GRF/SIMD16/eu_thread_count4, barrier_count2 and no per-thread memory-buffer
descriptor. Code size and static metadata do not establish JIT behavior,
complete ABI equivalence, occupancy, instruction-cache pressure or speed.
The timing cohort above decides the uniform candidate; code-size reduction
alone did not avoid its small-job regression. Review raw logs/JIT after this
owned closure. The candidate build has one immediate consumer: the forthcoming
selective-routing comparison; retain the validated8989 engine/fixture fallback
until replacement qualification or explicit comparison closure.

While this cohort is frozen, the benchmark client adds minimum actual-input
sizes98304 and131072 in source`10a18c2e7`; its source files are outside this
component build/run's pinned inputs. After closure, the maintained client,
evidence and fail-fast suites passed6+14+7 tests, including the new sizes'
context boundary and actual-count checks. The first longer
screen should use98304 with the current131072 context/resident-KV setting,
keeping the selected engine route, chunk and other placement settings fixed.
This asks whether amortizing fixed work over more input improves throughput;
it is not a same-workload kernel speedup. Keep first-use and subsequent requests
separate and record actual counts. A131072-minimum screen needs a larger
context allocation and its explicit capacity/boundary check first. Neither
screen establishes physical262144-context or sustained decode qualification.

The next Down prototype adds startup-only
`STRATA_SYCL_PREFILL_DOWN_SCATTER=large` (default0). Only the native-BF16,
vLLM auto/wide backend accepts it; jobs below2334 rows still execute separate
Down and accumulation. The selected wide row-cache kernel retains BF16 output
rounding before FP32 FMA and writes neither the Down nor diagnostic tensor.
The outer window keeps its existing immutable route owner, queue ordering,
status drain and exception lease. No router readback or transfer is removed.
Command-service profiling rejects this fusion until that observer is qualified.
Request-level counters distinguish fused and separate jobs/rows.

This is Main-authored routing and validation, with no new copied implementation.
The fresh component plan adds actual startup-selected flag0/large processes,
2333/2334/2335 boundaries, false-return non-writes, map masks1/2/3 and
8191/8192 tails. Three clean timing processes compare the selected hook plus
fallback against separate dense+accumulation at14 fixed-window shapes. The
prospective gate is geometric-mean latency ratio<=0.97, no shape>1.05 and
improvement at every>=2334-row shape in each process. A3% component gain warrants
only a subsequent matched model experiment; the prior uniform candidate's
failed5% gate is unchanged. Requalify the same-build GU fixture before the next
model comparison because its shared adapter translation unit changed during
the previous Down work. Source45c387d11 built successfully;54 ownership,45
traced smoke and110 full correctness events passed in closed owned processes,
including startup flag0/large, threshold/tail/fallback checks and mixed map
masks. No fresh driver fault occurred. Same-build GU requalification, including
its clean timing, was then completed:36 ownership cases,4 small correctness
cases with13 admission rejects,18 full cases and3 clean timing processes passed
without a fresh fault. The component wall-time geomean is0.952777; the existing
>=2334 selector still excludes the slower160-row shape. Selective Down timing
remains pending; no new model speed or adoption is claimed.

The current source-view build's Down fixture differs from the source45 fixture;
its old receipt does not qualify the new executable. A fresh plan at
`1dff76365` repeated54 ownership,45 traced smoke and110 full correctness events
successfully. All three clean timing processes then exited normally, with504
used queue/wall samples, complete ownership closure, unchanged boot and no
fresh driver fault. The selected/separate latency geomean was0.937848,
but64 contiguous rows regressed9.53% (ratio1.095345), exceeding the unchanged
1.05 per-shape limit. Every>=2334-row shape improved in each process.
**The selective candidate fails its advancement gate; do not run its gated
model comparison.** This neither changes source45's original unrun timing
status nor qualifies model speed, quality, sustained decode or full context.
The next source question is whether the low-level separate reference and
public-wrapper candidate measure the same host dispatch work at fallback
shapes. Any corrected comparison needs a separately committed plan and evidence;
the current failure and all individual timing samples remain in the
[current component result](../bench/results/2026-10-15-pr6-recovery-audit/moe-down-scatter-current.json).
The validated a576 engine and its same-build fixtures were hash-checked at
`known-good-before-selective-down-v1` before reusing the build cache. The old1222
fallback's9812 original source blobs and recipe were recovered before retirement.
A bounded lookup of3074 local source files and2307 Git blobs found both old
controllers and one old source override, but left original source/dependency
recipe gaps. Six obsolete unqualified native-product payloads were retired;
their original small receipts and unique source evidence remain separate from
qualified builds. This is no global-retirement or historical source-recovery claim.

### Production-matched Down comparison

The next fixture uses `--bench-public PROCESS ARM`. Both arms call the actual
public try hook and fallback: a fresh process starts with Down0 or Down-large,
which remains immutable through runtime initialization and all measurements.
There is no engine math change. Six processes run A0,B0,B1,A1,A2,B2; each covers
all14 fixed78451-window shapes, one warmup and six measured rounds. Use256
repeated calls for64/160-row timing batches and16 otherwise, to reduce fixed
marker/drain noise without removing small shapes. All504 used queue/wall
samples and each native process identity/actual selector must be retained.

The original v6 failure remains failed. This new comparison keeps the same
<=0.97 geomean, <=1.05 worst-shape and per-pair>=2334-row improvement gates.
A partial cohort cannot advance. Repeat54 ownership,45 traced small and110
full correctness events on the exact new fixture before timing. The controller
checks full process order, unique PID/start ticks, route markers, effective
startup selectors, timing granularity, shape coverage and closure. Its offline
negative cases cover partial/duplicate/reordered evidence, reused identities,
a low-level control, the old small-batch size and the wrong startup route.
Source`9b6db2661` compiled the fixture; the engine SHA matches the validated
a576 engine. Main's corrected source now records every timed hook return,
requires exactly repeat or zero fused dispatches, checks the14 preflight
row/placement cases and prospectively requires every marker bracket, including
warmups, to last at least2ms. This screens granularity, not timer accuracy or
noise. A shorter batch invalidates the comparison; retry requires a new
committed plan. Main passed all16 offline fault/evidence tests at source7626a2fc5,
including wrong timed-hook counts, duplicate preflight placement, warmup duration
and inconsistent marker arithmetic. Source396d0c91f built the corrected fixture;
the engine remains byte-identical570b65dd with native origin9b6db2661 preserved.
Compilation does not reuse qualification for the changed Down fixture.
The exact93248e27 fixture passed54 CPU ownership,45 traced smoke and110 full
numerical/admission events, all owners closed and no fresh GPU fault. All six
clean timing processes then closed normally, with504 eligible used samples,
checked actual dispatch counts and no fresh fault. The candidate/base latency
geomean is0.953898926 (4.61% shorter), the worst shape ratio is1.020361103,
and every selected large shape improves in each of the three process pairs.
The unchanged component gate permits the117K model comparison; it establishes
neither a whole-model gain nor adoption. The original v6 failure stays failed.
The a576 fallback protected that fixture/model comparison. Its six executable
copies were retired after the completed cohort selected the current qualified
generation and both original source inventories/recipes were recovered from Git.
Only its `source-build.json` remains for the current scoped bank/provider
admission proof; preserve the selected current generation before the next
native rebuild. The dedicated
`bench_engine.py summarize-down` analyzer preserves the frozen>1.002 model
criterion, while `summarize-source-view` keeps>1.005. It rechecks original
component qualification, exact117489-token requests, six-process order, paired
inputs and clean work conservation; first-use and the1000tok/s target remain
separate. Main passed all26 model statistics, work-counter and build-identity
tests; these are offline checks, not end-to-end model qualification. The later
bank/provider-scoped build-proof change at sourcec4583093b passed all32 model
statistics/build-proof tests after timing closed; all16 Down evidence tests also
passed again. Independent receipt-chain review found three admission gaps.
Main fixed the same-source shortcut, committed/full-receipt binding and launch
rechecks in93257a29a; all37 offline tests passed at0c1632b10. The original GPU
controller commit78a119cf7 contains the exact compact build receipt corresponding
to the retained full a576 record. Both9813-input source inventories and original
recipes were recovered, and current bank/provider bytes matched. This reuses
only those component gates, without changing their original status or claiming
a new GPU check. The Down fixture remains separately qualified. Independent
Orca source audit and separate native Luna-max review found no verified reducer
defect. Receipt-chain wiring tests remain optional strengthening; actual eligible
model evidence is still required before a cohort decision.

The first parameter-validated Down-large model request then passed at117489
fresh read tokens, thinking on and24 generated tokens. All96 mixed windows
retired; actual Down counters record7485 fused jobs/49290450 fused rows and
29868 separate jobs. Native and benchmark owners closed normally, with no fresh
fault. The profiled930.2tok/s is diagnostic; it is not a clean speed result.
The clean comparison kept source-view enabled and changed only Down0/large
across the frozen six-process order. Each process completed four fresh117489-token
requests, thinking on, with unchanged384-slot expert banking,117488 GPU residual
rows,81920+35568 MoE windows and8192 operator chunks. All owners closed normally
without fresh GPU faults. Each process's median of its three subsequent requests was:

| Process pair | Down0 tok/s | Down-large tok/s | First use Down0 / Down-large tok/s |
| --- | ---: | ---: | ---: |
| 0 | 998.6 | 1005.8 | 926.2 / 933.1 |
| 1 | 998.2 | 1005.8 | 925.9 / 931.0 |
| 2 | 998.3 | 1005.5 | 925.6 / 933.2 |

The three independent paired median ratios have geomean1.007345335
(+0.7345%), passing the prospective>1.002 criterion with every pair faster.
All three candidate medians exceed1000tok/s: this establishes the stated
subsequent-request117K prefill speed milestone, with first-use costs separate.
The committed reducer recovered9813 original native/dependency inputs and112
frozen benchmark inputs, checked the original component gate, actual model
dispatch/work conservation and matched request/configuration identities. See
[all24 speed samples and the decision](../bench/results/2026-10-15-pr6-recovery-audit/prefill-long-input-current.json).
Retain selective Down as an opt-in for further qualification. Default adoption,
broad quality,physical262144/session/RESTORE and sustained70tok/s decode remain
unqualified;24 output tokens do not establish sustained generation speed.
The selected engine completed the official thinking-enabled long-generation
check and three independent clean processes, with actual fresh inputs33211 and
33285 tokens. Regular-code generated2931 tokens at14.6/14.7/14.6tok/s;
varied-prose generated10805 at9.5/9.5/9.4tok/s. Observed MTP acceptance was
84.84% and39.98%, confirming the intended strong/weak pair. All owners closed
normally without fresh GPU faults. The existing reducer recovered114 original
benchmark inputs and the four-process result chain, excluding the traced check
from clean statistics. See the [clean samples at8fe6b1af6](https://github.com/yqYo1/Strata/blob/8fe6b1af6cd9a7c86411143dfc9f98d9209bc1e8/bench/results/2026-10-15-pr6-recovery-audit/decode-long-generation-current.json).
This is a baseline, with no measured gain or70tok/s qualification. Logical
target expert-cache hit rates were2.0% and1.0%; these do not establish physical
RAM/PCIe traffic or CPU arithmetic as the limiting resource. The next consumer
is qualification of the CPU-time observer before resource attribution. Its four
CPU clock/pool tests passed both standalone and in the native build, including
output parity and enabled/disabled accounting. The new engine passed two bounded
native requests with one valid CPU record per completed request, zero clock/
counter errors and normal ownership/fault closure at`69888d181`; the old engine
also passed two requests on the proposed assets at`d2ec0ec93`. These tiny checks
do not measure observer overhead or qualify long-generation attribution. The
clean speeds above used libze-intel-gpu1 package26.31.39395.14; the system subsequently installed
26.31.39395.16 and updated libudev. These assets need explicit qualification;
the shared qualified runtime pins remain unchanged. Six small bank upload/guard/
unwind cases passed with full API trace on the proposed assets at`db11be131`.
The earlier full-model check hit its64MiB API-trace budget and remains incomplete,
with no fresh driver fault. Both closed model checks used parameter validation
and stage progress.
The same-build provider and six full expert-bank cases then passed on the
proposed assets at`f5561c265`. The official long-generation CPU observation at
`26d0f4bd6` completed both workloads:33211/33285 fresh inputs and2931/10805
generated tokens. All six CPU records matched the four warmup/calibration and
two long requests, with zero clock/counter errors, normal native/benchmark
closure and no fresh GPU faults. See the [current diagnostic and two CPU observations](../bench/results/2026-10-15-pr6-recovery-audit/decode-long-generation-current.json).
Its13.0/7.2tok/s readings include CPU/wall observers and parameter validation;
they are not clean speed or a matched observer-overhead comparison. Worker CPU
time includes scheduled memory stalls and pool polling; GPU-reach wall time
does not isolate device service. No limiting hardware resource is established.
The matched observer-bundle comparison then completed six independent processes
in the committed off/on/on/off/off/on order on the same engine and proposed
runtime. Every request read33211 fresh tokens, generated2931 including thinking,
and accepted2105 of2481 drafts. The off samples were14.3,10.3,9.0tok/s; the
on samples were9.1,10.2,10.5. All owners closed normally without fresh GPU faults.
The three paired on/off per-token latency ratios were1.573758,1.005914,0.853985;
their geomean1.105731 fails the unchanged1.05 aggregate/1.10 maximum-pair gate.
The final pair reverses direction and the off arm itself varies widely. This
does not identify a constant observer penalty, a CPU-clock cost or a hardware
bottleneck. Do not compensate profiles by dividing by the observed geomean.
The reducer recovered all115 original benchmark inputs; all six predecessor
receipt hashes were also checked against their original Git blobs.

The isolated CPU-region calibration then passed all12 batches in six owned
off/on/on/off/off/on processes at source`227869de9`. Each process ran one and
six physical cores,500000 regions per thread and256 fixed integer operations
per region. Actual checks cover independent process identities,50ms batch
floors, exact integer results/counts, CPU clocks, affinity and restored masks.
Paired additional scheduled CPU time per region was1111.19/1143.01/1115.99ns
with one thread and1172.59/1178.44/1170.50ns with six; the medians are1115.99
and1172.59ns. This includes both CPU-clock reads and bookkeeping/control costs.
It is a local differential microbenchmark, not a production-overhead bound or
evidence of a syscall/vDSO path. Do not linearly correct model wall time from it.
No lighter-accounting rewrite is selected from these values. The same-runtime
old/new comparison and first bounded resource observation below are closed;
their results still do not identify a limiting hardware resource.
Independent source/interpretation review found no calibration reuse blocker.
The closed calibration export, standalone fixture and parser tests are retired;
restore original source`227869de9` and its committed plan for a concrete new
comparison. Individual batch timings/totals remain in the current index, with
original per-thread vectors at`1c35d2156`.
A separate read-only placement observation found the native leader on CPU0,
expert workers on CPUs1–5,19 earlier threads allowed on0–11 and five later
threads on0. Their roles/activity and any latency effect remain unknown.
The host pin occurs in `SessionLoopScratch::init`, not the pool constructor;
Linux inheritance is a hypothesis for those later masks, not a demonstrated
cause. Do not change affinity during an active comparison.
Explicit runtime-baseline adoption remains pending and the shared qualified
asset hashes remain unchanged. The observer-off old/new comparison used the
frozen old/new/new/old/old/new order, the unchanged official regular-code request
and four warmups. Both arms removed CPU/wall/validation observers before native
exec and preserved the safety settings. Original fallback inputs/recipe and
both runtime checks bound the selected binaries. The prospective diagnostic gate
was new/old per-token latency geomean<=1.05 and maximum pair<=1.10 after request/
work reconciliation. This single workload does not replace the strong/weak pair
required for adoption. The closed launcher/reducer is recoverable from original
source`77f1ad644`; `generation_engine.py` now contains shared admission helpers
only. No inference optimization, runtime-baseline adoption or resource saturation
follows from this comparison alone.
Independent source audit and native Luna-max review found no technical reuse
blocker. Main closed the missing automatic warmup assertion at`77f1ad644`;
all18 offline evidence tests passed. The final reducer checked all six original
hash-chained journals against their closed receipts, including the four
successful warmups, request identities and order. Residual
null-counter work is the intended old/new test subject, not proof of a defect.
The audit worker is completed and its terminal is closed. The original frozen
index remains recoverable at each measurement revision.
All six processes closed with the same33211 fresh input tokens,
2931 generated including thinking and2105 of2481 accepted drafts. All four
warmups passed and native/benchmark owners closed normally without fresh faults.

| Pair | Old tok/s | New tok/s |
| --- | ---: | ---: |
| 0 | 13.1 | 11.8 |
| 1 | 11.5 | 14.4 |
| 2 | 13.2 | 11.4 |

Original closed receipts remain in Git at`c1e33bdf0`,`b6e70064d`,`4de041ebb`,
`8a76c139e`,`12e0ac0a2` and`50315cbef`. New/old per-token latency ratios are
1.115807,0.798233,1.156885, with geomean1.010035. The geomean meets1.05, but
two pairs exceed the unchanged1.10 maximum; the diagnostic nonregression gate
fails. Old/new medians are13.1/11.8tok/s with sample SD0.954/1.629. Preserve
all samples and the failed decision. Pair1 reverses the faster arm; these data
do not establish a constant engine cost or a hardware bottleneck. Keep the old
engine as the speed reference. It also completed the passive diagnostic below.
The newer extra build has since been retired after source recovery and consumer
migration; see the current [build decision](../bench/results/2026-10-15-pr6-recovery-audit/native-gdn-engine-build.json).
No new inference optimization or runtime-baseline adoption follows.
After the CPU-capacity comparison closed, the unused internal CPU observer's
production hooks, header, two CPU fixtures and build mode were retired to
`d50d31347cc8505a40181db8308735bfa0058427`. Before independent copy-safety changes,
all maintained native/build files were restored byte-for-byte to the selected
`396d0c91f301` source; only that observer delta existed in those paths.
The external collector, original failed gates and compact observations remain
distinct. This removes an unqualified experiment from the next candidate's
baseline; it is not a new build, speed measurement or observer-overhead result.
During process0, a read-only capability check of the owned native client's
`/proc/<pid>/fdinfo` exposed xe `drm-cycles-ccs`, `drm-total-cycles-ccs` and
memory residency fields on kernel7.0.0-38-generic. This is field availability,
not an interval utilization result. A subsequent bounded diagnostic can test
client identity, duplicate descriptors, engine capacity, monotonic recovery and
aligned request intervals before interpreting cycle deltas. The kernel's
[generic contract](https://www.kernel.org/doc/html/latest/gpu/drm-usage-stats.html)
and [xe implementation contract](https://www.kernel.org/doc/html/next/gpu/xe/xe-drm-usage-stats.html)
describe these counters. Engine busy fraction still does not measure arithmetic
efficiency, VRAM bandwidth or physical PCIe bytes. Preserve unknowns rather
than converting the earlier GPU-reach wall percentage into device utilization.
The independent [client resource reader](../bench/sycl/client_resources.py)
at`3743a3bd4` passed all14 synthetic CPU-only tests with
`python3 -B tests/core/test_client_resources.py`. It binds the original receipt
to the current boot and rechecks PID/start/SID,
deduplicates descriptors, preserves raw cycle counters and recovery spans,
rejects impossible deltas and reads scheduled CPU ticks. Enumeration stops at
the descriptor/thread budget before materializing an unbounded directory.
Collector completion requires an explicit terminal and at least two samples
with eligible CCS intervals; empty or unsupported clients cannot pass. A
confirmed process exit can close this process-bounded collector, but planned
model-phase completion and coverage remain separate checks. The CLI has a900s/
32MiB bound and opens no GPU device.
The [resource controller](../tools/sycl/resource_engine.py) at`d9ff684f1`
prepares a bounded official regular-code diagnostic on the old engine, recovering
its9813 original native/dependency inputs and recipe. The original build path
differs from the retained path; admission checks the original committed relocation
record as well as source identity, byte size and actual executable SHA256.
Its relocated-source regression test passed, along with the native-wrapper
synthetic lifecycle test,7 HTTP/eligibility tests and18 evidence tests (41 total
with the reader tests). No new native engine was built.
The wrapper now records the boot ID and owns the passive reader beside the
native child. The diagnostic preserves parameter validation, stage progress
and the qualified safety environment; its observer-contaminated speed is not
a clean result. Exact client clocks divide observations before and after the
first content/reasoning output. These are client windows, not native prefill/
decode boundaries. The reducer replays raw cycles, checks the terminal and
contained spans, and records the unsampled final edge without extrapolation.
The committed900s/1s/32MiB collection plan ran inside the1800s/64MiB official
benchmark budget. The independent Orca Luna audit and native Luna-max review
are completed; Main owned all tests and GPU work.
The first live collection at`497311e01` then closed normally without a fresh GPU
fault:33211 fresh inputs,2931 generated tokens and2105/2481 accepted drafts.
Its8.5tok/s includes parameter validation, stage progress and the collector;
retain the earlier clean speeds separately. All523 snapshots and both CPU/client
windows passed replay/coverage checks, with no final unsampled response edge.
Orca audit plus native Luna-max review identified a result-to-journal clock
binding gap, with no actual mismatch. Main fixed it at`7d5a11afb`; both admission
tests passed (42 relevant CPU tests total). The committed post-closure analyzer
matched all work/clocks to the original hashed journal and reproduced every
resource value, preserving the original execution outcome.

| Observed client window | Contained CPU wall s | Process CPU core equivalents | CCS busy fraction |
| --- | ---: | ---: | ---: |
| Before first output | 50.160 | 0.943 | 0.830 |
| After first output | 342.636 | 4.319 | 0.325 |

The leading native thread stayed on CPU0 and used283.33 scheduled CPU seconds
after first output. Five other busy threads on CPUs1–5 used232.83,233.71,256.55,
240.39 and231.88 seconds. Scheduled time includes polling and memory stalls;
CCS busy cycles include device stalls and are associated-client measurements.
These observations do not show GPU arithmetic saturation or establish CPU
arithmetic versus RAM/submission/synchronization as the limiting resource.
The second, collector-only diagnostic at`a71b6214a` removed parameter validation,
stage progress and internal CPU/wall observers while preserving the selected
engine, runtime and safety settings. Its committed plan at`ca84a06c5` completed
with the same33211 fresh inputs,2931 generated tokens and2105/2481 accepted
drafts, normal ownership closure and no fresh GPU fault. All519 snapshots passed
journal-bound clock/work and resource replay checks. Before first output it
observed51.186 contained CPU-wall seconds,0.936 process core equivalents and
0.809 CCS busy fraction; after first output,333.541 seconds,4.380 core equivalents
and0.334 CCS busy fraction. These are client windows, not native phase boundaries.
The8.8tok/s reading still includes the external collector and is not a clean
speed sample or a paired overhead comparison. The post-closure analyzer at
`dc4acc8f8` reproduced the original`682d47f8b` result. Its333 contained reader
calls after first output totalled1.088 seconds; this synchronous call duration
does not bound indirect driver/cache/clock effects or total model slowdown.
See the [current compact resource result](../bench/results/2026-10-15-pr6-recovery-audit/client-resources-current.json).

An idle-device Sysman probe at`dc4acc8f8` successfully initialized the installed
loader1.32.0 with validation and bound PCI0000:05:00.0. The device reported
bandwidth-counter support, but the first`zesDevicePciGetStats` returned
`ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS` (`0x70010000`); no successful snapshot
was obtained. The owned probe closed normally with no fresh fault, but the
counter capability check failed. No permission setting changed. Keep physical
PCIe traffic unknown and do not repeat the same probe under unchanged access;
the [compact denial result](../bench/results/2026-10-15-pr6-recovery-audit/sysman-pcie-capability-current.json)
preserves its source, actual assets, command and outcome.
After PR141 made the original source/results default-reachable, Main verified
the closed captures, probe recipe/assets and original hashes, then retired85
raw/transaction files totalling15929917 bytes. Both closed raw roots are absent.
The consumed probe source/mode is retired to`dc4acc8f8`; the selected engine
and its five same-build fixtures remain. This changes no original outcome.

A static audit of the actual3304-byte`native_experts.txt` found Down weights
IQ4_NL in39 layers and Q2_0 in9, with no IQ3_S Down. Gate/up uses17 IQ3_XXS,
10 IQ3_S,20 IQ2_S and1 IQ4_XS layers. Thus the pack's IQ3_S filename is not a
per-projection format contract. Its H2560/FF640/512-expert geometry and packed
payload are recorded in the resource result; these do not measure physical
traffic or dynamic reuse. Do not optimize an unused IQ3_S Down path.
The [CPU capacity controller](../tools/sycl/cpu_capacity.py) freezes a clean
pool-workers5/1 comparison: six independent processes in5,1,1,5,5,1 order,
three pairs, with four official warmups and identical long-generation input/work.
Support directional sensitivity only if all three paired per-token latency
ratios agree and their geomean exceeds1.10 in that direction. Failed/partial or
mismatched-work cohorts remain ineligible. All7 offline negative-evidence tests
passed. Controller source`5e7943d80` then passed the bounded one-worker check:
two thinking-enabled small inputs,16 generated tokens each, actual one-worker
plus host marker, normal owned closure and no fresh GPU fault. The generator
uses Strata's actual tokenizer/template; an earlier optional-tokenizers import
failed during preparation before any native launch and was corrected.
The independent Orca audit and native Luna-max review are now closed, with no
execution safety blocker. They require model/expert/MTP byte identity,
cross-process boot/kernel/guarded-runtime equality and evidence of CPU-pool work
during the timed request before a performance interpretation. The unlaunched v1
clean plan was explicitly cancelled, preserving its successful small check.
V2 keeps the original six-process order, all original work keys and the10%
directional gate. It hashes model inputs before the cohort and after its last
request, with device/inode/size/mtime/ctime guards before and after each native
run. Hash I/O stays outside measured execution. Existing completion footers
provide decode-cache-miss evidence: original DONE cache counters start after
prefill, and at pcie-frac0 refused native rows enter the CPU pool. This adds no
timed observer and does not count exact routes or physical bytes. Ten offline
negative-evidence tests passed. V2 source`7d9dce79a` pinned all15 model inputs
(85989955827 bytes) and passed the revised one-worker check: both16-token
requests, unchanged model/platform identity, normal owned closure and no fresh
GPU fault. The first clean long process (five workers plus the host) then
completed all four warmups and the33211-fresh-token request:2931 generated
tokens including thinking,653.9 prefill tok/s and14.6 decode tok/s. Native and
benchmark owners closed normally, model/platform identities stayed unchanged,
and no fresh fault occurred. Its existing completion footer reports2.0% decode
expert-cache hits, establishing CPU-pool work under the configuredpcie-frac0.
The second process (one worker plus the host) also completed all four warmups,
with identical fresh input,2931 generated tokens and2105/2481 accepted drafts.
It measured647.6 prefill tok/s and8.7 decode tok/s, with normal native/benchmark
closure and no fresh fault. The first one/five per-token decode latency ratio
is1.684906196. The third process (one worker plus the host) also passed with
the same work and no fresh fault:651.9 prefill tok/s,8.3 decode tok/s and
352495.2 decode ms. The fourth process (five workers plus the host) passed with
the same work and no fresh fault:652.9 prefill tok/s,11.2 decode tok/s and
260639.1 decode ms. The fifth process (five workers plus the host) also passed
with unchanged work and no fresh fault:652.4 prefill tok/s,10.1 decode tok/s
and290587.5 decode ms. The final one-worker process passed at642.6 prefill
tok/s,8.1 decode tok/s and362645.0 decode ms, with the same work and no fresh
fault. All six owners closed normally; final hashes of all15 model inputs
(85989955827 bytes) match the initial model fingerprint. The frozen reducer
and an independent replay agree: one/five per-token latency ratios are
1.684906196,1.352426401 and1.247971781; geomean1.416765734. Every pair is slower
with reduced capacity, exceeding the unchanged1.10 directional gate.
The result supports CPU expert-pool capacity sensitivity for this request.
Keep the individual five-worker14.6/11.2/10.1 and one-worker8.7/8.3/8.1tok/s
samples: medians11.2/8.3 do not erase the substantial baseline spread.
This does not isolate arithmetic from RAM/scheduling or prove baseline saturation,
model quality, strong/weak adoption or70tok/s. A bounded explicit-copy candidate
can reduce both CPU arithmetic and RAM demand; it still needs the separate
source/submission/retirement gates below and measured total-path improvement.
After PR144 preserved the original controller and result commits on the fork
default, Main verified closure and useful evidence, recovered the temporary
native source views from396d0c91f301, and retired187 closed raw/audit files
totalling37592380 bytes. The deletion manifest is pinned in the result atf8a60975c.
The selected engine and five same-build fixtures remain for the comparison.
This diagnostic can test CPU-pool capacity sensitivity, not distinguish
arithmetic from RAM limits or prove saturation at the baseline. Small-check
success establishes neither task quality nor a speed gain. Keep earlier clean
speeds and all failed comparison gates separate; see
the [current plan/check](../bench/results/2026-10-15-pr6-recovery-audit/decode-cpu-capacity-current.json).
A separate original-source audit finds the selected SYCL ordinary Arena provider
cannot admit nonresident GPU experts through the existing alias/pin gate,
although SYCL forces explicit mode0 copy. A host-USM FileExpertSource complement
is a separate conditional route. Before a new copy capability, check every
enqueue result and require checked retirement or fail-stop with owners alive:
current fetch ignores submit errors, and a query error is not completion.
The [reviewed source requirements](../bench/results/2026-10-15-pr6-recovery-audit/research-index.json)
do not qualify enabling the route or predict a gain.

The independent verifier correction checks every copy submission, refuses
invalid staging extents and drains earlier copies even after a later submission
fails. An error from a stream query no longer counts as completion. All SYCL
verifier teardown now requires checked drains before releasing borrowed owners;
if sticky runtime errors prevent that proof, the process stops with owners
alive. The redundant queued flag callback is removed from the SYCL path.
This change still leaves the ordinary-Arena nonresident GPU route closed.
Before execution, the committed plan is to build with
`python3 tools/sycl/build_engine.py build-copy-safety`, commit its build result,
then run `python3 tools/sycl/check_verifier_retirement.py`. The prospective gate
requires all ten existing submission cases plus six isolated production-verifier
cases (valid copy, partial submission failure, bounds, query error, destructor
failure and copy-drain failure), normal owned closure, unchanged source/runtime
and no fresh GPU fault. The last two cases must exit1 before unwind or atexit;
they inject only a host-task exception, never an invalid GPU address or stalled
kernel. The small traced/validated check explicitly admits the already observed
driver.16/libudev.18 diagnostic delta without changing the shared runtime
baseline. Source changes, build receipt, small-check execution and any future
model-route/performance decision remain separate gates. The source396 known-good
engine and its five fixtures remain the fallback for this immediate comparison;
the second generation must be selected or retired at qualification closure.
Source`d20736c36290` compiled successfully. Both same-build fixtures passed all
ten submission and six verifier cases under the traced/validated environment;
the two fail-stop cases exited1 with only their pre-stop markers, without
unwind/atexit markers. Source/runtime hashes remained unchanged, all owned
processes closed, the boot stayed unchanged and no fresh GPU fault appeared.
See the [small-check result](../bench/results/2026-10-15-pr6-recovery-audit/verifier-copy-safety-current.json).
The independent delta audit identified a public failed-object reuse/output gap:
the server exits on run failure, but the object did not itself refuse another
window or old logits. It also noted early flag-B publication for an empty
batch, although SYCL's next graph segment already waits for checked retirement.
The follow-up latches failed copy submission as terminal and refuses subsequent
run/commit/logits/logprobs/residual access; it publishes flagB only after checked
retirement, including empty batches. The revised prospective gate adds a seventh
isolated empty-batch case and strengthens the partial-failure case to exercise
public reuse and output refusal. Build with
`python3 tools/sycl/build_engine.py rebuild-copy-safety`, commit the build result,
then run the same check controller with its fresh v2 root. All17 cases, source/
runtime/boot identity, ownership and fault checks must pass; the original16-case
pass remains a result of the earlier source only. Source`cd81dcd133e7` rebuilt
successfully and all17 revised cases passed, with normal owned closure,
unchanged source/runtime/boot and no fresh GPU fault. The separate native
Luna-max review confirms that existing callers abort after run failure and
earlyB cannot bypass the checked SYCL segment barrier; the public failed-object
reuse gap was not a demonstrated current-caller or GPU-hang mechanism. The
small follow-up closes that API gap. Its source and17-case result are distinct
from the independent audit ofd207. These component results do not enable the
ordinary-Arena route, establish model correctness/nonregression or add speed.
Return to the bounded explicit-copy/CPU-share comparison next.

The ordinary-Arena candidate is independently authored against this Strata
provider/planner and verifier. `STRATA_SYCL_VERIFY_ARENA_COPY=1`, set before
native exec, permits only explicit mode0 copies from the existing immutable
private-Arena layer view. It rechecks expert count, stride, full size and
address extent; shared/file/transient providers gain no new promise. Existing
pin and device-alias predicates retain their meaning. The default remains off.
The existing `pcie-frac` still selects a bounded share of distinct missed
experts; this changes neither prefill placement nor model/KV precision.
It uses the existing16-slot staging allocation, with exact admitted staging and
window-arena bytes logged on opt-in, and adds no resident weight bank.
Request-footer counters separate submitted/completed copy payload and blobs,
immutable-view groups, GPU miss groups and CPU expert jobs/entries. These are
requested/retired logical bytes, never measured physical PCIe traffic. Partial
submissions count only accepted copies, including rejected draft work.

Before any model launch, build with
`python3 tools/sycl/build_engine.py rebuild-arena-copy`, commit its build result,
then run `python3 tools/sycl/check_verifier_retirement.py`. The frozen component
gate requires17 native-variable-stride provider/planner/pool cases covering grouping, share/capacity,
disabled/direct modes, file/shared fallback, missing callbacks and malformed view refusal;10
existing submission cases; and8 verifier cases, including all16 largest-model
blobs (2662400B each) from ordinary RAM, payload/outer guards and copy counters.
The v5 gate additionally requires16 grouped-output cases from the existing
`sycl_mmvq_test --arena-grouped` fixture. It covers GU18/21/22/23 crossed with
Down20/42 at H2560/FF640 and windows1/4 (groups10/16, entries10/40): all seven
actual model pairs plus supported IQ4_XS/Q2_0. An ordinary host vector is copied
on the transfer queue into a guarded device payload, retired before compute,
and mixed with a separately uploaded resident expert. Check both entry tiles,
invalid-token zeroing, empty groups, finite/nonzero references, copied bytes
and outer guards. Grouped output must equal separate GPU projections using the
same unchanged Q8_1 contract. This checks transfer/grouping, not generated-text
identity or an independently implemented projection oracle; common-mode native
projection errors and model/task quality remain separate. Two distinct weight
blobs repeat across group metadata; the original verifier case separately covers
16 distinct largest-size copy payloads. This is a Main-authored extension of
this fork's existing fixture, with no XeStrata source or proposal input.
Require normal owned closure, unchanged source/runtime/boot and no fresh fault;
retain the source396 validated fallback during this comparison. No component
speed threshold is imposed: this is a capability/safety prerequisite, not a
kernel-optimization claim. A separate committed model check must establish
actual GPU expert dispatch before32K+ thinking-enabled strong/weak generation
comparisons. Default/model/physical262144/session/RESTORE qualification and70
tok/s remain pending. This candidate targets measured CPU-pool capacity
sensitivity; it does not assume CPU arithmetic rather than RAM is saturated.
The initial component attempt at`189e5163d` stopped in the new host fixture with
SIGILL, before any model launch, with no fresh GPU fault. It incorrectly chose
the canonical Q2_0 CPU path: its quantizer is compiled with AVX512 flags and the
actual binary contains unconditional ZMM instructions, while this CPU reports
no AVX512F/VNNI/VBMI. Preserve that failure at`6352c2c3f`. The corrected fixture
uses the actual native IQ3_XXS/IQ2_S GU and IQ4_NL Down formats on two differently
sized layers, retaining all grouping/fallback/refusal assertions. The revised
17+10+8 gate uses a fresh v4 root; the native engine code is unchanged by this
fixture/controller correction. This is not canonical-pack support or a GPU fix.
At`3ef95537a398`, the rebuilt native provider17, submission10 and verifier8 cases
all passed, including the16 largest-size copies and byte/outer-guard checks.
All owners closed normally; source/runtime/boot stayed unchanged and no fresh
GPU fault appeared. Engine bytes remained those of`189e5163d`; only the fixture
and controller were corrected. Independent Orca producer/consumer review and
native Luna-max review found no verified source blocker; grouped arithmetic/output
qualification remains separate from the35 component cases.
The next bounded model gate uses `python3 tools/sycl/arena_copy.py prepare`,
then a committed generated plan before `run`. It pins the current engine and
all component bytes, recovers the original build inputs, hashes all model inputs
before preparation and checks their identities before/after execution. Keep
pool5,131072 context/resident INT8 KV and the previous8192 prefill settings;
set the new opt-in before exec andpcie-frac0.25. One owned process serves two
thinking-template questions with at most16 output tokens each. Both must show
nonzero immutable-view GPU miss groups, equal submitted/completed blob and byte
counts, remaining CPU work, normal closure and no fresh fault. Use parameter
validation and stage progress, not the full model API trace. The driver.16 delta
is admitted only for this named diagnostic, without updating the shared runtime.
This check does not compare speed, numerical/task quality or physical262144/
RESTORE. Long-generation strong/weak comparisons require their own committed
order, repetitions and decision criteria after actual route admission.
The frozen check on controller source`9feb1dbb1` then passed both requests:
66/64 input tokens and16 generated tokens each. Actual mode0 GPU miss groups
were1309/1534, with2684518400/3143577600 logical payload bytes respectively;
all submitted blobs/bytes completed. CPU work remained positive in both requests.
The native owner exited normally with unchanged boot and no fresh GPU fault.
See the [compact route result and original frozen-plan reference](../bench/results/2026-10-15-pr6-recovery-audit/decode-arena-copy-current.json).
This is execution/copy-conservation evidence, not native-output or task-quality
qualification or a speed gain. Validate dispatched mixed-format expert outputs
before advancing a long-generation candidate. Preserve the source396 fallback
and default-off candidate for that imminent comparison.
At build source`218561fc0`, all51 v5 component cases passed, including the16
staged grouped-output cases. Engine and prior three fixture bytes remained
unchanged; the added fixture supplies the new arithmetic evidence. Owners closed
normally with unchanged boot and no fresh fault. Independent native Luna-max
review confirmed that the omitted generic MMQ/IQ-dequant environment flags do
not affect this grouped route; its relevant defaults match the model. This is
the fixed tested contract, not a claim of bit identity for every optimized path.

The controller source`a6e9e4762` passed all six offline evidence tests, including
copy-footer conservation and negative comparison cases; these do not execute a model.
The next `arena_copy.py prepare check` uses official `bench.sh` for one
parameter-validated, thinking-enabled regular-code request: at least32768 fresh
input tokens,1024 generated tokens including thinking and actual answer body,
with positive copied groups and exact copy conservation. Commit its generated
plan before `run`. This is a diagnostic, not clean speed. After it closes, the
same frozen source admits six independent clean processes via
`arena_copy.py prepare clean N` for N0..5, each followed by a committed plan and
`run`. Both arms use pool5/pcie-frac0.25 and the same binary, configuration and
input; only `STRATA_SYCL_VERIFY_ARENA_COPY` changes in order0,1,1,0,0,1.
Preserve qualified queue/safety settings, remove observers for clean speed and
require strong MTP acceptance>=0.60. Generated text, output length and draft
counts may differ; retain the actual values. Pairs0/1,3/2,4/5 signal faster only
when all three candidate/base ratios exceed1 and their geomean exceeds1.02;
signal slower only for the corresponding inverse condition, otherwise
inconclusive. Stop on any failed or mismatched process without an automatic
retry. Record completed logical bytes per emitted token and remaining CPU work;
neither establishes wire traffic or saturation. This screen cannot adopt a
candidate: weak-MTP, agent-session, broader task quality, long-prefill
nonregression and physical262144/session/RESTORE checks remain pending.

The frozen six-process ordinary-Arena copy screen then closed normally on the
same source/binary, with33,211 fresh input tokens, complete copy conservation,
unchanged boot/model/runtime and no fresh GPU fault. The matched clean pairs were:

| Pair | Copy off, tok/s | Copy on, tok/s | Candidate/base |
| --- | ---: | ---: | ---: |
| 0 | 14.1 | 11.5 | 0.815603 |
| 1 | 14.0 | 11.5 | 0.821429 |
| 2 | 10.3 | 11.2 | 1.087379 |

Their geometric mean is0.899794. The unchanged all-three-pairs decision rule
returns **inconclusive**; it does not support adoption. Preserve the lower final
control sample without excluding it or calling it noise; its cause is unknown.
The work and acceptance remain different between arms as recorded in the
[individual samples](../bench/results/2026-10-15-pr6-recovery-audit/decode-arena-generation-current.json).
Ordinary-Arena copy remains default-off. The next separate scheduling experiment
must qualify its exact source and event/owner behavior before model execution;
the completed copy screen does not qualify that new implementation.

The frozen long route diagnostic then passed33,211 fresh input tokens and3,177
thinking-plus-answer tokens, with four successful official warmups, normal native
and owner exits, unchanged boot and no fresh fault. MTP accepted2253/2775 drafts
(81.19%); answer/thinking bodies contain5571/3730 characters. All314831 requested
expert blobs completed, totaling642680908800 logical bytes, with1010725 remaining
CPU unique misses and1410855 CPU entries. This is202291756 requested bytes per
emitted token. At unchanged traffic,70tok/s would request14.16GB/s; a hypothetical
6GB/s payload link would permit29.66tok/s if every byte crosses it. These are
conditional requested-payload calculations, not wire measurements or saturation.
The11.4tok/s diagnostic reading includes validation/progress output and is excluded
from clean speed. Continue the unchanged six-process screen; no adoption follows.

The Main-authored early-PCIe prototype started from`52b9734b9` in an isolated
child during that now-closed screen. Its original commits are merged into
`feature/sycl` for qualification. It copies no XeStrata or other external
implementation. `STRATA_SYCL_VERIFY_PCIE_EARLY=1` requires the native ordinary-Arena
copy route, mode0 and existing shared/hit overlap. The unchanged copied-expert
projection is captured separately. After DMA submission, a freshly recorded copy
event gates its launch on the same compute queue, behind shared/resident work
using the same scratch. CPU result staging and the merge remain after CPU return;
checked host-source retirement and failure latching remain required. No future
event or host-task wait is captured. Unrecorded copy events and plan/count
mismatches must refuse submission. `STRATA_SYCL_VERIFY_PCIE_EARLY_AUDIT=1` adds
ready/pending queries only for diagnostics; clean timing leaves it off. Its
counters prove submissions/readiness observations, not hardware saturation or
quantified overlap. Default behavior remains the existing schedule.

Before build/run, commit all source, fixtures and updated controller/recipe.
Following screen closure and source review, require the existing51 gates
plus the new real-queue early-enqueue fixture: ordinary host copies, resident-to-
copied scratch ordering, payload/output/outer guards, empty input, maximum copy
extent, fresh-event replay, and refusal of mismatched or unrecorded inputs.
The separate `sycl_verifier_retirement_test --early` entry has six cases:
small,16 largest blobs, empty, fresh-event replay, mismatched plan and unrecorded
event. They use production publication/fetch/launch/retirement with guarded device
payloads and a resident-produced dependency byte. Retain all16 native projection
cases: synthetic graph ordering does not replace their arithmetic checks.

Two closed test mechanisms remain failed. Source`246415bfc` passed all51 existing
cases, then its host-task gate delayed arrival at the first UR memcpy until the
two-second deadline. Source`b41fecbb8` again passed51, then its fixed32x64MiB finite
DMA was complete before the first pending observation. Neither failure admitted
model execution or a resize/retry. Both owners closed normally without fresh GPU
faults. The [current57-case result](../bench/results/2026-10-15-pr6-recovery-audit/verifier-copy-safety-current.json)
and its original Git references preserve these distinct outcomes. They do not
establish production deadlock or physical transfer bandwidth.

The current isolated native-event method uses `tools/sycl/check_event_gate.py`
and the manual `sycl_host_event_gate_test` target. Three fresh processes check
import only (no dependent command/kernel), then copy/kernel dependencies, then
command-graph replay. The selected context/device owns one regular host-visible
event, imported with keep ownership and HOST scopes. A host signaller is armed
before import; its two-second fallback counts as failure. Import and dependent
submission each need to return within100ms. The pending/output/progress window
is50ms. Cleanup signals and joins before draining consumers, dropping the SYCL
wrapper, and destroying the native event then its pool. Loader/header inputs
are pinned test-only dependencies; production linkage and safety settings stay
unchanged. Source`1dd3eef34` passed native import in367us with50,100us pending,
then failed the copy case's resident-progress assertion; graph was not run.
The [native-event result](../bench/results/2026-10-15-pr6-recovery-audit/native-event-gate-current.json)
preserves normal closure, unchanged boot and no fresh GPU fault.

The next v2 run is a diagnostic with the same enqueue order and thresholds. It
records the resident marker, whether the gate is still held, and host time in
the resident event wait. This separates marker failure from deadline expiry;
it cannot qualify a replacement gate even if it passes. Its source`7cb121367`
again failed: dependent submits returned in578us; the resident wait took1,999,496us,
the resulting marker was60 (the correct0x3c), and the gate was already released.
Normal closure/boot and no fresh GPU fault were confirmed. This does not establish
when the kernel actually ran; the remaining question is host wait semantics versus
queue/engine progress. The completed source review traces the version-aligned
Intel LLVMv7.1.1 UR-v2 wait to the target native event's host synchronize; mapping
the installed runtime DSO to that exact source commit remains unverified.
The v3 diagnostic keeps the same enqueue order and safety settings. It
materializes the resident native event and queries it at most20 times,5ms apart,
with a200ms observation limit. If it remains unready, release the gate before
the SYCL wait; if already ready, time that wait while the gate is held. Record
native readiness, query/handle/wait times, gate state and the post-wait marker.
Native-handle retrieval can flush/materialize an event, so this observation is
not unchanged-work timing or proof of the original kernel completion time.
No host marker is read before checked completion, and no host atomic capability
is assumed. This diagnostic cannot qualify the replacement gate or model route,
even if it completes successfully. Source`172b9f1e9` compiled, then the one
`copy-query` child failed the bounded native-query observation before producing
a readiness sample: its200ms limit was exceeded or the held gate expired.
The combined check does not identify whether a resident native query or the
gate-status check consumed that interval. No raw status/time sample exists to
resolve this yet. The child closed, boot was unchanged and no fresh GPU fault
was recorded. A new observation needs separate call timings; this failed run
must not be relabelled as a device fault or a qualified gate.
The v4 observation adds flushed begin/end timestamps around the existing
gate-held checks and resident native queries, plus a window summary printed
before the failure check. Enqueue order, native call count,20x5ms observation,
200ms window,2s fallback and safety settings stay unchanged. Keep per-call
counts/maxima and the last window even on failure. These text observers can
change timing and provide no throughput or gate qualification.
Source`174911063` built and the bounded child again failed, with normal closure,
unchanged boot and no fresh GPU fault. Two calls started and returned: the
gate-held check took5us and returned true; the first raw resident
`zeEventQueryStatus` took1,998,861us and returned SUCCESS. The whole window
was1,998,885us. No second query or post-window gate check was reached. This
locates the latency inside the raw resident query, rather than the outer
observation loop; it does not establish when the device kernel completed.
`Runtime::make_queue(bool)` selects profiling, not in-order behavior: every
queue it returns is in-order. This fixture explicitly profiles copy/compute
and leaves the probe unprofiled. Compatible NEO26.31.39395.14 source's
`queryCounterBasedEventStatus` calls timestamp-completion synchronization for a
profiled counter-based event. Exact correspondence to installed26.31.39395.16
remains unverified, so this is a testable source hypothesis, not attribution
of the installed DSO or model latency. The next diagnostic changes only the
compute queue to `make_queue(false)` while preserving the copy/probe queues,
context/device, enqueue order, checks,2s fallback, tracing and safety settings.
Queue profiling is a new explicit diagnostic variable; no production queue or
model route is qualified by this comparison.
That source`51864e379` contrast built and completed with verified queue
properties, normal closure, unchanged boot and no fresh GPU fault. All20 raw
resident queries returned NOT_READY in at most3us, across101,429us while the
gate remained held. Manual release preceded a52us event wait and correct
marker/payload. All41 observed calls started and returned. This is evidence of
profiling-dependent native-query latency in the small fixture; no resident
completion was observed during the held interval. It does not qualify the
replacement gate, production queues, model performance or physical context.
The next bounded diagnostic restores the original profiled compute queue and
sets `EnableImplicitConvertionToCounterBasedEvents=0` before native exec. The
original NEO26.31.39395.14 registry documents the key and the actual pinned
installed DSO contains its exact name; neither proves effective consumption or
the installed DSO's exact source mapping. Compare with the committed profiled
v4 control, rather than attributing the difference from unprofiled v5 solely to
the key. A new `copy-query-profiled` mode preserves enqueue order, observers,
checks,2s fallback and all runtime safety settings; require actual1/1/0 profiling
and in-order queues. Keep the same20x5ms/200ms window and record unsuccessful
status/latency observations. This single fresh v6 process remains diagnostic,
with no model, replacement-gate or production configuration qualification.
Source`44253b63b` built and the new diagnostic failed its unchanged200ms
window: the gate-held check took4us and the first resident native query took
1,999,034us, returning SUCCESS. Actual profiling was1/1/0 and all queues were
in-order. Both calls returned, ownership closed and no fresh fault occurred.
The override did not avoid the wait; installed key consumption and explicit UR
counter-event selection remain unverified. Review the original UR event creation
and native gate scheduling before another configuration or model experiment.
The completed source review corrected confusion between v4/v6 timings. Neither
receipt records the actual host signal timestamp, so a late SUCCESS does not
establish post-fallback ordering or the earlier kernel completion time. The
profiling-off NOT_READY samples also do not identify a scheduling cause. A new
committed no-gate/held-gate diagnostic with joined query-thread and actual signal
timestamps is a possible next discriminator; production query cost and the
complete57-case/model gates remain separate unresolved questions.
The next production diagnostic uses `STRATA_SYCL_QUERY_PROFILE=1` on the
existing default route. It times the complete compatibility query wrapper at
resident launch/pool-return and early-PCIe launch/pool-return, without adding
queries or changing their status handling, arithmetic, queue properties or
safety environment. Counters reset after prefill and report four constant-size
rows after decode; disabled calls read no clock. Wrapper wall time includes
runtime checks, EventState locking and any completed-event wait, plus effects
of clock readings. There is no event registry-map lookup in this wrapper.
These four points exclude direct event queries elsewhere in decode; their sum
is not total request query time, device service time, CPU arithmetic or proof
of resource saturation. The new
`check_verifier_retirement.py --query-profile` result requires the normal51
cases plus an isolated real completed-event observer check. It does not modify
the failed early57-case outcome or admit early/arena-copy execution. Next gates
are a committed, bounded two-request default-route model check, followed by a
separate32K+ thinking-enabled diagnostic and independent observer-overhead
qualification. No throughput improvement or new model validation is established
by adding the observer.
The observer engine at source`ce8dd0e94` then passed all51 existing normal
cases and the completed-event observer case. The six offline evidence/runtime
admission tests passed at`8d4e53937`. Two earlier preparation refusals (unknown
diagnostic mode, then receipt package labels compared with identity-only
metadata) occurred before native launch and remain recorded in Git. The new
controller admits only the exact same-build component-qualified runtime hashes;
it does not relabel the mode or update the shared runtime baseline.
The bounded normal-route model check subsequently completed both16-token
thinking-template requests, with no fresh GPU fault and all owners closed.
Resident launch queries were240/288 calls, all pending, with1.065699/0.973088ms
wrapper intervals; their matching pool-return queries were all ready, with
1.240609/1.513645ms. The second request's combined2.486733ms is about0.21% of
its1185ms rounded decode time. The first request includes graph capture and is
not a steady decode comparison. The four-point observer saw no2s query delay
on these requests; short safety checks do not establish long-generation cost,
observer overhead, a limiting hardware resource or a speed improvement.
Use the separately committed official32K+ thinking-enabled regular-code
diagnostic next, with only this narrow observer and the established validation/
safety settings. Query intervals remain diagnostic until independent overhead
qualification, and clean speed remains separate.
That official diagnostic completed at controller source`06750114d`, native
source`ce8dd0e94` and engine SHA256`51e97cbbd9d550d2c0d892bc2537faba5dfa07d6be25a202108865b0bd1d8307`.
It processed33211 fresh inputs and2931 generated tokens, with2783 reasoning
and5578 answer characters and2105/2481 accepted drafts. Four warmups and the
timed request matched the official journal; all owners closed, boot unchanged
and no fresh fault. The timed request made39744 queries at each resident point:
launch150.074225ms (20ready/39724pending, max3.391505ms), pool-return354.337961ms
(39713ready/31pending, max15.038424ms). Total504.412186ms is0.24208% of the
208368ms rounded decode interval; both inactive PCIe categories remained zero.
These four queries were not dominant in this observed production execution.
The208368.1ms/14.1tok/s and651.3prefill tok/s include validation/progress and the
query observer; they are diagnostic, not a new clean speed or overhead estimate.
The run recorded1205698 unique CPU expert jobs,1556213 CPU entries and32107
cache hits out of1588320 lookups (2.0% rounded). These are logical work counts,
not measured DRAM or PCIe bytes, and do not distinguish CPU arithmetic from RAM.

The next placement question corrects an earlier research assumption: the
`prefill.reset()` before standalone token generation is outside `--serve`.
Serve's `refill` returns lent cache slots without resetting the persistent
owned `sp` workspace. The official benchmark uses serve, so the standalone
reset does not establish spare decode VRAM. Review the existing borrowing path
and actual variable-size cache/MTP/graph owners before a larger-cache trial;
static sizes and the2% hit rate alone establish neither memory admission nor a
speed gain. Keep context/session/RESTORE state and cached graph readers intact.
The closed Orca source review and native Luna-max review found a concrete
safety prerequisite before borrowing: lend/refill/adapt discarded residency
upload errors, so failed publication could leave device slot ownership stale.
No such copy failure was observed in the no-borrow diagnostic, and this is not
an established cause of the earlier hangs. The current source change checks
all post-initialization publications, including standalone and small-prompt
reservations, and propagates failure before dependent prompt/verify work.
The new host-only fixture injects first/middle/last-device failures for lend
and refill, checks exact snapshots and stops fanout; it does not inject GPU
faults. Newly authored helper and tests use no external implementation.
The rebuild at source`126eda334` reused the current-build cache while
preserving the named qualified fallback. Engine SHA256 is
`5b10303034666b8e25b8c625511534a1e04ffddd0419e7221af90cb20b3d7708`.
Normal51+query-observer1+residency-publication9 cases passed in the fresh v2
component root, with unchanged boot, all owners closed and no fresh GPU fault.
The six offline evidence/runtime-admission tests also passed. The injected
failures test the shared host publication helper and fake table snapshots;
production call-site returns were source-reviewed, not model-fault-injected.
Compilation and these component passes do not exercise actual borrowing.
The next source check shares the exact sized/uniform tail calculation with ten
host cases (boundaries, no-fit/overflow,128-slot floor and request rounding).
A default-off single-GPU `STRATA_SYCL_CACHE_LOAN_AUDIT=1` checks the device
residency table after lend/refill and compares every refilled and unlent-prefix
resident expert against its immutable RAM bytes. It records actual loan bytes,
chunk, offset and free VRAM without retaining tensor dumps. The helper, audit,
fixture and controller are Main-authored from this fork's Strata path, with no
external implementation or XeStrata source.
The rebuild at native source`1dc52b068` reused the current cache and preserved
the qualified fallback. Engine SHA256 is
`2fce2d79fc5f6eebf8829b65d0de863d99d697a0c4c013ad9c25d28cae9fc917`.
All normal51+observer1+publication9+tail10 cases passed in the fresh v3 root;
ten offline evidence/runtime tests passed. All owned processes closed, no
fresh driver fault occurred and boot/runtime remained unchanged. These are
component/admission prerequisites, not a borrower model result.
`tools/sycl/cache_borrow.py prepare` then freezes the1280-slot hypothesis,
8194/257-token thinking-template inputs and qualified runtime/safety settings;
commit its plan before `run`. It requires positive loans at two distinct
rounded chunks/offsets, exact GPU maps and full resident payload agreement,
same-slot refill, G1/T4 decode and configured MTP expert detach/restore. The
readback audit is diagnostic work, not clean speed or a quality benchmark.
The v1 check allocated1280 slots/2639104000bytes (about2.458GiB), and lent
1115 slots from165 for chunk8192:2320512000bytes available versus2319958528
required. The24576-entry device map matched the host, with477712384bytes free.
It then refused before prefill math: `layer_major.inc` requires owned scratch
for the mode10 mixed accumulator. No generated token, refill/payload check or
decode qualification resulted. The supervisor terminated/reaped the exact
owned child on ERR; its original `ownership_closed=false`/incomplete validation
remain recorded, with separate confirmed process closure, no fresh driver fault
and unchanged boot. This was a configuration guard refusal, not a GPU hang.
The controller now observes a native ERR through a bounded natural exit,
instead of raising inside Owner and immediately signaling the process. Eleven
offline tests, including a real CPU-only child that emits ERR and exits1,
passed at controller source`81662a525`; no second request or cleanup signal
was sent and the child/session closed. This does not revise the v1 outcome.
Main missed this guard in the first preflight. Do not disable the fast mixed
scratch/staged-bank path merely to obtain an admission pass: inspect its direct
allocation assumptions and introduce checked borrower bounds/lifetime before a
new component/model plan. The original v1 root is closed and cannot be retried.
The completed Orca source audit and native Luna-max review additionally found
unchecked Alloc arithmetic and nontransactional relayout: GEMM is rebound and
carve changes pointers/T before all allocations are known to fit. The next
implementation needs a sealed checked layout and parent-loan generation with
reader retirement; on failure preserve the old layout or poison it and keep
the parent unavailable. Registered USM containment alone does not authorize
a loan. Preserve the owned fast path and staged-bank rules, and qualify host,
device, error and relayout behavior before another borrower model check. These
are source findings, not an observed GPU fault or an implemented borrower route.
The first implementation step checks Alloc's element/guard/alignment/cursor
arithmetic and sizing slack, validates signed32 routing/page shape bounds, and
stops all later takes after a failed allocation. Carve returns before making
aliases from a failed prefix or treating a null borrowed region as owned.
A shared layout-state guard marks init/rebinding in progress; any return or
exception before commit poisons the instance. Public run, internal chunk run,
draft-KV and chunk reporting cannot use that incomplete layout; reset/init is
required. This is a poison-on-failure contract, not pointer rollback. Fifteen
host cases exercise the actual arithmetic/state helpers, including injected
host phase exceptions; real Prefill/GEMM/queue fault injection and checked
parent ownership, graph retirement and loan/refill admission remain pending.
The new helper and tests are Main-authored from this fork's allocation path.
At native source`6a553b54b`, the engine and seven fixtures rebuilt successfully
(engine SHA256`b6e870895ab0e3399626169f7da02e09d011e7c72110406afcc342884e65876e`).
All86 prerequisite cases and11 offline tests passed in a fresh v4 root; every
owner closed, boot remained unchanged and no fresh driver fault occurred.
The owned-mode math was not changed, and the new engine has not run a model.
The completed Orca Luna audit and native Luna-max review retained this owned
route step. They found no supported-route defect, but malformed QSA metadata
can overflow earlier sizing expressions; add shared primary/range/geometry
preflight without rejecting valid split ordinals. Also make reset/destruction
retirement checked: the existing release ignores stream-sync errors. Next add
a small actual Prefill init/relayout failure fixture before parent-loan
admission; helper-state exceptions alone do not exercise those paths.
CPU-math/RAM/GPU resource attribution remains unknown; physical262144/session/
RESTORE, model quality and clean speed remain separate gates.
The next committed gate uses a shared QSA metadata/product preflight and checked
Prefill retirement before freeing owners. Its new model-free SYCL fixture calls
actual Prefill init/relayout/reset with a1024-cell session and256/128-row scratch.
Freeze normal71+host22+device17 cases in a fresh v5 root: smoke capacity refusal
first, then exact/moved/shrunk layouts, failed init/rebind poison, reset recovery,
QSA/USM/context refusal and complete parent/guard readback. The final separate
child changes a fully drained external queue's context; reset must exit1 before
owner release. This tests host-domain synchronization rejection, not a device
fault. Use the shared immutable safety/tracing environment with alias1,MMQ0,
compactFP16,BF16,QSA1,GDN-chunk640,vLLM backend set before exec. Each child has
180s wall/600s CPU/4GiB RSS and the job64MiB raw/artifact budget; stop on the first
failure without resizing or retry. Pin the source/controller before rebuilding
the current comparison cache; keep the396d0c91f301 fallback until the next model
qualification selects a generation. These gates do not enable borrowed mixed
scratch: parent-loan generation and all graph/MTP/cache-reader retirement,
refill/publication and a new complete model plan remain required. Run the11
offline controller tests separately after components. No new speed sample is
claimed by this plan.
The v5 build at96e6d9e81 completed (engine SHA256
`968200f3d60abb265279eae8890041b89827b85dfd93017c095c00e91ebad5be`).
Normal71+host22+device-smoke1 passed; the full child passed13 subcases and
then failed its foreign-queue assertion. The fixture created that queue with
default out-of-order properties, so queue_for rejects it before the intended
context check; the combined assertion misleadingly prints "foreign queue
admitted". The gate is failed/incomplete, retirement was not executed and no
model is admitted. All owners closed, no fresh GPU fault occurred and the11
offline tests passed. Keep this outcome; correct both foreign queue constructions
and separate the assertions in a fresh committed plan after source review.
Native Luna-max review corrected the report's overly broad coverage claim and
approved the fixture-only correction. The new v6 plan keeps all110 cases,
order, environments, budgets and stopping rules above. Both foreign queues must
be explicitly in order and have a different context; verify the original
same-domain stream completes before substitution. Separate actual init refusal
from its runtime-domain reason, and require the reset child to print both setup
checks before the expected context-error exit1. Preserve failed v5 through Git;
neither this correction nor source review admits parent loans or model work.
At sourcea51989adc, the corrected fixture rebuilt and all110 cases passed in
the fresh v6 root. Both init context refusal and reset's actual context-error
exit1 were reached; the latter stopped before owner release. All processes
closed normally, boot remained unchanged, no fresh GPU fault occurred and the11
offline tests passed. Engine bytes remain968200f3d60a from96e6d9e81; the changed
fixture has its own newly built identity. Keep the shared QSA/checked teardown
step, with borrowed mode10 still rejected. Next implement checked parent-loan
generation, sealed interval/layout and complete cache-reader retirement, keeping
loan bookkeeping until successful refill/publication. No new model or speed
result follows from this gate, and physical262144/session/RESTORE remain open.
The next Main-authored source step represents the parent cache, exact loan
interval, nonrepeating generation, decode host-reader admission and masked/use/
refill/publication phases. Only ExpertCache mints a loan capability; USM
registration alone cannot mint one. Prefill binds it to its checked carve's
base/capacity/used extent and layout generation, and rejects stale/inactive
capabilities at run/chunk/draft entry. Checked borrower retirement and parent
close keep owners live on uncertain completion. This is an implementation
step, still unbuilt/unqualified: generator/verifier integration, actual global
reader retirement, refill/publication conservation and host/device/error gates
remain required before borrowed mode10 can run. The original owned fast guards
remain in force until those call sites supply the complete contract. All new
ownership code is original project work, with no XeStrata implementation input.
The independent Orca audit and native Luna-max review confirmed a fill-admission
race and missing pending-loan checks in that first source step. Main replaced
the Boolean fill check with a scoped, single-writer reservation held through
the final enqueue. Prepare, close and phase transitions refuse active writers;
refill writers are restricted to the loan interval. The ticket's pending query
retains poisoned/refilling/in-transition loans without dereferencing a cache.
Relayout rejects them before mutation, and release/reset fails before freeing
owners. Uniform products, residency dimensions and sized-slot alignment/sums
are checked; sized allocation uses real slot indices and keeps exact default
payload lengths. Source`181043472` passed the37-case host-only fixture under
GCC13.3 with warnings as errors:22 scratch/layout,10 loan-protocol and5 scoped
writer cases. Both compiler/test processes closed normally. Exact source and
compiler hashes, commands and budgets are retained in the
[archived host result](https://github.com/yqYo1/Strata/blob/5f1c06452cd2c13c0d64bf67ca8a870a52b76065/bench/results/2026-10-15-pr6-recovery-audit/prefill-cache-loan-host-current.json).
This does not build or qualify the changed SYCL engine or actual cache tickets.
The prospective generator/verifier connection is default-off under
`STRATA_SYCL_CACHE_SCRATCH_LEASE=1`, restricted to the single-cache serve route.
It closes decode admission, retires verifier/session/MTP readers before use and
keeps the exact lent rows/slots through refill completion and checked residency
publication. Short borrowed chunks retain full dense weights. This connection
still needs independent complete-source review and actual ticket/device/error
tests. Existing owned-only mixed-mode guards continue to reject borrowed mode10;
no model execution, new speed or full-context qualification is claimed.
The next fresh v7 component plan commits154 checks before rebuilding:
normal75 (including14 tail/preflight cases),host37,existing device17,actual
cache-loan20 and five fail-stop children.
The new model-free fixture uses actual ExpertCache tickets and Prefill layouts,
a1024-cell session,256/128 rows,128 prefix slots of256B and1MiB tail slots
(at most513MiB device parent). It checks exact resident payload/map restoration,
a moved smaller second generation, an empty resident-tail cycle, cache reopen
epoch refusal and prepared/used/refilling/poisoned reset plus pending parent
close. GPU work is bounded memset/copy/readback; no invalid address or hung
kernel is injected. The existing small traced check runs first. Each child keeps
the qualified immutable environment and180s/600CPU-s/4GiB-RSS/64MiB output
budgets; any failure closes the gate without retry. Preserve the396d0c91f301
fallback and rebuild only the closed comparison cache. Complete source review
and generator/model qualification remain separate from these prospective tests.
The completed source audit/native curation confirmed a conditional short-tail
setup gap, without evidence that a model request triggered it. Main now selects
an occupied interval and preallocates the exact masked rows before preparing
the parent. A short request widens its tail only toward the validated lower
bound; if no borrowable resident exists, rejection leaves the parent unchanged.
Four host cases cover exact/widened/prefix-only/invalid maps. The device fixture
also checks that an empty-tail preflight leaves decode admission open; it then
exercises the lower-level capability on that empty tail independently of the
generator positive-mask policy. The review mistakenly carried the old57-case
early-PCIe gate into its next-step text; use this154-case cache-loan gate instead.
The failed early-PCIe route stays disabled and separate.
The first native build at`e9a3854c1` stopped before GPU/test execution:
Clang's warnings-as-errors rejected two existing PfPart aggregate initializers
after adding the lease field without an explicit default member initializer.
The failed receipt and closed ownership remain recorded. Set that field's
default to`{}` and commit the source before a corrected build of the same
prospective v7 gate; no test outcome or qualification is reused from this failure.
The corrected source`0a385b388` built successfully (engine SHA256
`60f09c471ff928c9e223bdf71ac249b96778ef233f0908fe0b39b618ab5e2434`).
All154 v7 checks and11 offline tests passed. The provider's single summary
marker represents17 cases, so the138 emitted case markers are not the case
total. All owned processes closed, boot was unchanged and no fresh GPU fault
occurred. Actual cache overwrite/same-slot refill/map restoration, moved and
reopened tickets, the empty resident-tail capability and five pending-owner
fail-stop paths passed. Keep the source-reviewed preflight and ownership step.
Next connect the checked parent capability to all three existing mixed-scratch
owner guards without changing the fast staged-bank path, then qualify the full
generator/model cycle. These component results do not qualify model math,
performance, task quality,physical262144/session/RESTORE or global graph execution.
The next source step connects the same active parent capability to all three
mixed-scratch owner checks. Bind and execution now also check that the complete
R,region and embedding spans fit the sealed used extent and are pairwise
disjoint; parent capacity alone is insufficient. Owned allocations retain their
existing direct-owner check. FFN math, staging, phase/drain and subview checks
stay unchanged. Trace records distinguish direct owners from parent loans and
associate each borrowed FFN with its loan/layout generation; successful mask
and refill publication record the actual decode admission state.
The v8 normal75+host39+device17+loan20+fail-stop5 gate (156 cases) passed on
build source`87ce707df`, using the same qualified settings, budgets and no-retry
rule. All owners closed, boot was unchanged and no fresh GPU fault occurred.
The two added host cases cover sealed-used bounds, overflow and view overlap.
The396 fallback remains protected. The earlier
standalone GCC mode/result is retired to commit5f1c06452; its same-build consumer
has superseded it. The changed engine SHA256 is
`02a61db21c50f5824f3af748ec6e89fbe05db04c7cacc0454c6b7c435d0e5bc6`.
Component passes alone did not establish model qualification; v2's outcome is below.
The independent Orca audit and native Luna-max curation found no source-supported
defect in those three guards. This is bounded source acceptance: planner geometry
is outside the ten reviewed files, and execution/math remain unqualified. The
fresh v2 model controller requires that committed review, the complete same-build
156-case result and the explicit checked-loan flag. It checks all48 borrowed FFN
entries/retirements against one sealed generation, then actual decode admission
after each successful refill. Its two thinking inputs8194/257 each request16
generated tokens;8193 processed rows use only the primary mixed bank, so the full
tail remains unqualified. The closed v1 refusal and its original failed outcome
remain recoverable from the recorded Git reference; its separate process closure
does not turn it into a pass. Source`b5161eb1b` passed all15 offline controller
tests, including missing/stale/reordered evidence and natural ERR-child closure.
The same-build proof recovered9831 original inputs and verified the engine and
all eight fixtures before the committed v2 model plan ran. That model check
failed: the first loan was masked and mode10 setup reached layer0, then run_impl
refused a stale/inactive owner. The source path replaces `m.cache` with the
private per-layer execution cache before run_impl; the new span helper also
uses that mutable pointer as the original loan's parent identity. This wrongly
rejects the valid parent ticket. No borrowed FFN, refill or generated token was
completed. Native exit1 and ownership closed normally; boot was unchanged and
no fresh GPU fault occurred. Preserve this failed result. The earlier source
audit missed the dynamic identity coupling. The original source correction
keeps the parent supplied to init separate from the execution cache. All lease
checks, activation and retirement use that parent; binding also requires the
ordinary parent cache to be active. The actual per-layer cache substitution
immediately rechecks the lease before inference, with a trace identifying the
distinct execution cache. The v3 model gate requires this actual
substitution and all48 borrowed FFN entries/retirements, two complete refill/
decode cycles and the original payload/map checks. Its offline parser rejects
missing, duplicated or false substitution evidence. Use a fresh v9 156-case
component plan, complete corrected-source review and a committed v3 model plan;
never retry closed v2 or bypass the owner check. Source`3832efe2a` passed all16
offline evidence tests. Build`bb71e78db` passed all156 fresh v9 cases, with
owners closed, unchanged boot and no fresh GPU fault. Its source audit and
native curation support the parent correction,
but found a short-residency evidence gap. Controller-only source`57dfa2204`
now requires all six ordered dense phases inside their respective loans:48
long-path activations before Full restoration, and Full residency with zero
activations on the short256-row path. Its17 offline tests passed. No native
input/fixture changed; same-build admission recovered9831 inputs and verified
the engine/eight fixtures. A separate native review accepted this controller
delta and all14 final pins, including12 unchanged inputs. The fresh v3 plan was
committed before its single model execution.
The committed v3 check passed: actual private-cache substitution preserved the
parent, all48 borrowed FFNs retired, both full1280-slot payloads and24576-entry
maps matched after same-slot refill, and both requests generated16 tokens.
The first1115-slot loan covered2320512000B; the smaller97-slot loan covered
241280000B at chunk256. Decode admission followed0/1/0/1. The short path observed
Full dense state with zero layer activations; the long path restored Full after48.
Exit0, ownership closure, unchanged boot and no fresh GPU fault were confirmed.
Keep the parent correction. Free VRAM at the four lend/refill observations was
477712384/151470080/147935232/138100736B: the larger decode cache leaves little
headroom, so capacity and longer generation still need measurement. Next compare
the checked larger cache against the owned small cache using official32K+ input
and long generation, with prospective repetition/order and clean processes.
The first input processed8193 rows and used only the primary mixed bank;
tail routing, clean speed, task quality,physical262144/session/RESTORE remain
unqualified. The default-off route stays unadopted for general use. Closed v9/v3 raw, JIT,
successful compiler output and consumed source-audit transactions were retired
after committing the evidence: 121 files, 39114382 bytes. The compact
model result records the path/hash/reason manifest. Current engine/eight fixtures,
the396 fallback and14 separate legacy provenance holds remain.
The official32K+ capacity admission is now maintained in
`tools/sycl/cache_capacity.py`: four official warmups/calibrations and one
thinking-enabled1024+ output request, with all five exact1280-slot refill/map
cycles and48 actual long borrowed FFNs required. Its21 offline tests passed
at3f4752335; native/controller review verified18 pins, and Main recovered9831
build inputs and verified the engine/eight fixtures and runtime. The source
report halved residual storage in its arithmetic; the correct BF16 extent
is20480B/row, as used by the controller. The512MiB admission headroom remains
unchanged. The committed official check passed
all four warmups and a33211-input/2932-output thinking request:48 actual FFNs,
five exact1280-slot payload/map refills and normal native exit0, with no fresh
fault. Its680140800B residual stayed fully on GPU in mode10. The observed
long-allocation free budget was4185026560B versus2307657440B fixed including
the512MiB allowance. The discrete pre-release low-water was14786560B; the long
refill observation was102322176B. Neither is continuous peak memory.
Diagnostic638.5 prefill/11.6 decode tok/s and85.03% draft acceptance are kept
as one observed request, without a clean gain or regression claim. Logical
decode cache hit rate was10.2%; remaining1120890 CPU miss jobs do not measure
RAM/PCIe traffic. Corrected117488-row arithmetic is21915776B above the observed
33K free budget.1256 slots would save63897600B while still lending enough for
chunk8192, but its117K admission and speed are unmeasured. The independent
clean owned128/borrowed1280 comparison is defined in `tools/sycl/cache_compare.py`.
Its31 offline tests and independent source review passed at controller source
`cba7b0a7c`; the owned32K parameter-validated check also closed normally with
48 actual direct-owner FFNs,2931 generated tokens and no fresh fault. These
checks admitted the frozen six-process comparison. All six subsequently closed
normally, with no fresh fault, matched generation/MTP regimes and unchanged
full hashes for all15 model assets. Each process handled33211 fresh input
tokens and2931/2932 thinking-plus-answer output tokens. The individual samples are:

| Pair | Owned128 prefill / decode tok/s | Borrowed1280 prefill / decode tok/s |
| --- | ---: | ---: |
| 0 | 653.1 /14.1 | 647.6 /14.6 |
| 1 | 644.9 /14.0 | 647.2 /14.6 |
| 2 | 648.5 /14.2 | 647.3 /14.6 |

All three exact-clock decode ratios favor borrowing:1.036040,1.040718,1.034376,
geomean1.037041 (**+3.7041%**). This misses the prospective>5% faster screen;
preserve its inconclusive/small-effect outcome rather than relaxing the gate.
Prefill geomean0.997771 (**-0.2229%**) passes the separate nonregression gate,
without a conclusive speed direction. Decode sample SD is0.05930tok/s owned
and0.01546tok/s borrowed. Cache hits are2.0%/10.2%, with1205698/1120890 CPU
miss jobs respectively; those counts do not establish physical RAM/PCIe traffic
or saturation. Keep the default-off route for the planned1256-slot117K capacity
follow-up, and calibrate mixed-format CPU work against RAM-read throughput
before selecting another decode kernel change. Default adoption remains pending.
See
[the current comparison](../bench/results/2026-10-15-pr6-recovery-audit/cache-comparison-current.json).
Broader quality, full mixed
tail and physical262144/session/RESTORE remain unqualified. Closed successful
raw/JIT and consumed source-audit transactions are retired after their compact
evidence is committed. The owned check and first clean pair retired217 files,
38525713 bytes; the comparison's research-index entry records the path/hash/reason
manifest and successful post-cleanup source/build admission. The remaining four
clean roots then retired280 files,51307063 bytes, with the second manifest in
the same index entry. All seven comparison/check roots are closed and empty;
their individual samples and decisions remain in Git. All9831 native
inputs,131 benchmark inputs, the engine/eight fixtures and396 fallback remained
verified; cleanup executed no GPU work. Current engine/eight fixtures remain
for mixed-format CPU calibration and the1256-slot117K check. The396 engine/five
fixtures remain the historical117K/long-generation reference for that explicit
old-owned/current-borrowed long-input comparison on the current runtime. Review
and retire the extra generation when that comparison closes; this32K cache-layout
screen does not qualify the current engine's117K borrowed mixed tail. The14
legacy provenance holds remain separate.
The same borrowed-cache request records1424035 CPU entries over1120890
unique CPU expert jobs: mean NT=1.27045027. Since each job has a positive
integer token count, at least817745 jobs (72.9550%) have NT1:
each NT2+ job consumes at least one of the303145 excess entries.
`expert_pool_dispatch_multi` increments entries per CPU-routed row and misses
by the number of distinct jobs; the request snapshots exclude prefill. This
bound is on logical job count, not CPU time, physical traffic or the unknown
per-format/NT histogram. Prioritize NT1 applicability when selecting a CPU
method; high draft acceptance alone does not establish multi-token expert reuse.
The CPU-only discriminator `tools/sycl/cpu_roofline.py` now passed all115 cells
and575 samples at source `c24bfb62f44bf797319557e9e7c1b3264acc6d52`, with the
production CPU sources/pinned ggml and official Intel LLVM/Zen3 flags. All56
real-weight numerical cases passed, with maximum relative-L1 errors6.52976e-8
GU and2.14853e-7 Down, and exact pool/serial outputs. Down42 uses an independent
scalar reference with production FP32 ActQ scales; its2.72877e-4 difference
from ggml's FP16 activation scales is a separate contract diagnostic. Finite
outputs and external canaries passed; private pool scratch is not instrumented.
Every sample lasted at least1.300002841s; maximum clock-query/warm-pass ratio
was9.93725e-5, below the prospective0.001 gate. Earlier495 timing samples and
the failed timing-floor outcome remain at `cf098f72e`, unchanged; the replacement
recollected the entire matrix rather than accepting partial original success.

On Ryzen5 5600X (6 cores/12 threads,32MiB L3), the coordinated256MiB read scan
achieved median32.40/39.35/30.96GB/s on1/6/12 pinned threads. It includes barriers
and checksum and measures requested reads; it is not a physical DRAM counter or
an upper bound for every access pattern. The six-compute-participant NT1 expert
results are below. Each call handles six distinct experts. Hot uses8.6–15.3MiB;
stream uses at least96MiB without revisiting a blob inside a pass.

| GU / Down | Stream logical GB/s | Hot logical GB/s | Stream us/expert |
| --- | ---: | ---: | ---: |
| IQ3_XXS / IQ4_NL | 26.40 | 28.29 | 82.43 |
| IQ3_XXS / Q2_0 | 22.14 | 22.77 | 77.47 |
| IQ3_S / IQ4_NL | 22.95 | 25.12 | 101.49 |
| IQ3_S / Q2_0 | 20.39 | 20.25 | 91.68 |
| IQ2_S / IQ4_NL | 26.84 | 28.69 | 73.43 |
| IQ2_S / Q2_0 | 21.14 | 22.19 | 71.45 |
| IQ4_XS / IQ4_NL | 40.66 | 59.67 | 65.48 |

IQ2/IQ3 hot/stream differences are about0–10%, while IQ4_XS gains about47%.
This supports investigating NT1 low-bit GU/decode work in the former families;
it does not establish whole-model CPU/RAM saturation. Blob bytes differ by
format, so compare latency as well as byte rate. The GU pool phase occupies
about70–81% of NT1 IQ2/IQ3 stream wall time, including its scheduling/waits.
Logical rates exclude extra table/scratch traffic. Actual per-format/NT job
frequencies and model critical-path CPU time remain unknown. Keep the pure-kernel,
six-job-call and model questions separate; no model speed or70tok/s follows.
See [the individual samples and pinned plan](../bench/results/2026-10-15-pr6-recovery-audit/cpu-roofline-current.json).
The1256-slot117K route now passed the official bounded admission at controller
source`f27e07e07`, using the unchanged`bb71e78db` engine. Its16 offline tests and
exact source review preceded the run. The74-token warmup and fresh117489-token
request completed both exact cache payload/map cycles, all96 FFN windows and
retirements,48 secondary-bank windows, dense Full restoration and actual decode.
All117488 BF16 residual rows occupied2406154240B on GPU. The actual parent loan
started at slot132 and lent1124 slots:2321664000B versus2319958528B required.
Live admission reported4298412032B free versus4206942336B fixed including512MiB
allowance. The mixed tail starts at1059131904 inside the1180698624B arena,
after its active plan prefix. Discrete lend/refill minimum free was128106496B;
this is not a continuous peak. Native exit0, owned-session closure, unchanged
full hashes for15 model assets and no matched fresh fault passed. The diagnostic
request took127263.2ms (923.2tok/s), with24 thought tokens and no answer body.
It does not qualify clean speed, sustained decode, quality or physical262144/
session/RESTORE. See [the compact admission and original plan](../bench/results/2026-10-15-pr6-recovery-audit/cache-long-current.json).

Next close the retained old-owned/current-borrowed long-input comparison before
changing engine inputs. Both engines/five and eight fixtures were rechecked
against9813/9831 recoverable original inputs. The
[comparison controller](../tools/sycl/cache_long_compare.py) now binds the exact
admitted suffix (first132/lent1124) and persists a terminal failure with unknown
fault evidence when the final journal probe fails. These controller-only changes
at`f29c46bda` passed12 offline tests; they are not a new engine/GPU qualification.
The qualified diagnostic preload now establishes same-process public config and
first-init evidence in the unchanged retained396 binary, before its`--help` body.
The330B host trace, wrong-provider stop125 and retained native165B trace passed;
all owned processes closed without a fresh fault. Config enable/init/release return0,
UNINITIALIZED is37 and the checked null-platform dispatch returns46. Its dedicated
validation logger is silent. One diagnostic loader reference remains until process
exit, so ordinary runtime teardown is not qualified by this helper. The wrapper
injects it only in the native child, and EAGER is set before exec to skip the
retained main's late environment write. The original core-test fixture was restored
to its exact recorded hash; new tool regressions live outside the native snapshot.
The earlier C11 build and numeric-enum parser failure remain failures in the
[research index](../bench/results/2026-10-15-pr6-recovery-audit/research-index.json).
At`df172f7e3`, all14 comparison tests pass, including same-PID/pre-model activation,
clean absence, exact loan binding and failed-journal terminal evidence. The corrected
source audit and native Luna-max review accepted the controller without another
source fix. The old-owned117K request then passed at the committed`fc118efc6` plan:
all96 direct-owner retirements, same-nativePID3599067 activation before model work,
24 thought tokens, native exit0 and complete owned closure, with a successful
journal query and no matched fresh fault. Its126038.0ms /932.2tok/s is first-use
validation evidence, not clean speed. All six clean processes subsequently
closed in the frozen old/current/current/old/old/current order. Each had one
short warmup and four fresh117489-token requests; three subsequent requests
form each process median. Native exits, owned closure, actual placement,
observer absence and journal probes passed, with no matched fresh fault. All15
model assets passed complete SHA256 checks before collection and after the
last process. Both native binaries and all138 benchmark inputs stayed fixed.

| Independent pair | Old-owned128 median, tok/s | Current-borrowed1256 median, tok/s | Current/old |
| --- | ---: | ---: | ---: |
| 0, old first | 1005.8 | 999.9 | 0.994195 |
| 1, current first | 1004.2 | 1000.1 | 0.995864 |
| 2, old first | 1005.1 | 1000.3 | 0.995245 |

All three ratios favor old-owned128; their geometric mean is0.995100913
(**current0.4899% slower**). Neither prospective2% faster/slower screen passes;
the nonregression gate passes. Do not dismiss the consistent direction as
measured noise, or adopt the borrowed route as a prefill speedup. First-use
values931.6/925.9/926.8/930.7/924.9/927.6 remain separate. Old source`396d0c91f`
meets the repeated117K-input1000 speed condition. Current source`bb71e78db`
has medians999.9/1000.1/1000.3 and misses its all-three-process1000 milestone.
These clean readings belong to`bb71e78db`, not the later`3b4998f326` status
instrumentation binary. The latter has instrumented117K execution evidence and
the33K MTP comparison below; its clean117K rate remains unmeasured.
The [compact result](../bench/results/2026-10-15-pr6-recovery-audit/cache-long-comparison-current.json)
retains all24 individual samples; its unchanged predeclared analyzer reproduces
the same decision after redundant result fields are removed. This is route-level
short-output prefill evidence: both engine revision and placement differ. All
24 outputs end after thinking, with no answer. Quality, sustained decode, agent
session, physical262144/session/RESTORE and default adoption remain pending.

Keep the old-owned128 measurements as the fastest recorded117K reference. Its
executable is now retired after the closed MTP comparison and exact source/recipe
recovery; retain its source receipt through the next candidate decision. Current3b
is the bounded execution fallback, without a clean117K speed or adoption claim.
The named current-borrowed1256 resource attribution is now closed. Main recovered the extra build's9831 exact
original inputs and recipe, verified its engine/eight fixtures and both source
commits' default reachability, then retired197 files/162893807B including consumed
research dispatch transactions. One qualified engine generation remains;14 legacy
provenance holds are separate. The fallback's original inputs and required bytes
passed again after deletion. Retired-candidate model admission stops before
reading deleted artifacts; its7 offline query tests pass. See the existing build
and comparison results for the path/hash/reason manifest. Source inspection confirms
expert payload refill reads the original host Arena and queues H2D copies; it
does not save the production expert payload D2H. D2H/memcmp is loan-audit-only.
The outer refill precedes prompt_ms but follows the internal NativeDense host
restore interval.

The unchanged current build then processed one parameter-validated117489-input
request:127198ms,24 thought tokens, native/collector exit0, complete owned closure
and no matched fresh fault. All15 model hashes and141 benchmark inputs match
before/after. The first controller failed because it used the independent-buffer
`direct_owners=1` parser for a borrowed lease. Preserve that failed receipt.
A separate closed-log analysis at`eaeba9e6c` uses the existing borrowed-owner
checks, including96 entry/loan/retirement tuples, stable nonoverlapping spans,
sealed generations and ready publication. Its8 controller and12 admission
regressions pass, as do the analysis's profile/client/ownership checks. It runs
no new native process and does not relabel the original controller success.
See [the compact attribution](../bench/results/2026-10-15-pr6-recovery-audit/prefill-resources-current.json).

The outer1124-slot expert refill is388.3ms, **0.3053%** of the observed prompt
wall time; it is not the dominant interval in this diagnostic. NativeDense's
nonexclusive entry/run/restore clocks are388.342/125598/819.103ms. The coarse
queue intervals are QSA44428.9ms, MoE37869.3ms and GDN25180.9ms; QSA attention
alone is30763.8ms and selection7688.98ms. These intervals include submission
gaps and are not additive GPU-only service. The125.472s CPU-accounting interval
contains119.335s main-thread CPU, of which78.872s is inside GPU waits and40.463s
outside. The MoE status region accounts for10.871s outside-wait wall; grouping
is0.245s. Status includes the copy/check/wrapper path, not pure transfer or CPU
math. Parameter validation and profiling remain enabled, with no matched clean
observer-overhead estimate.

All96 planned windows conserve to37353 completed jobs and56394240 routed rows.
7485 large fused jobs account for49290450 rows;29868 smaller jobs for7103790.
Associated CCS busy cycles are91.83% during the observed pre-output client
window; this is activity, not arithmetic efficiency. Next inspect the actual
QSA shapes and the complete status-copy/check call path, and choose matched
GPU memory/arithmetic references. Physical PCIe/GDDR traffic and compatible
sustainable ceilings remain unknown. Neither these observations nor requested
bytes establish a limiting hardware resource, clean speedup or model adoption.

The next status-wrapper diagnostic is independently authored here under the
repository license. It adds a nested partition of the existing MoE status region:
tag, two runtime prechecks, two queue/context lookups, copy enqueue, event observer,
checked drain, sticky-error postcheck, observer retirement and the status branch.
Each phase records calling-thread CPU/wall and its existing explicit-wait subset;
the remainder is kept unclassified. Existing per-window drains, asynchronous-error
delivery, status destination ownership and the shared-expert execution order stay
in place. No new SYCL command, allocation or process-environment mutation is added.
Disabled CPU profiling leaves this observer inactive. This is attribution code,
not an inference optimization.

First commit and run the ten CPU fixture cases and malformed-profile rejection
gates using `check_host_profile.py`'s new finite v2 root. Then build exactly the
engine and eight existing safety fixtures with `build_engine.py build-status-profile`,
preserving the selected396 fallback. Run the existing156 bounded safety checks
at the new `--status-profile` root before preparing a model plan. The subsequent
prospective discriminator is two separate117K diagnostic processes with parameter
validation and all safety settings fixed: UR tracing layer absent/present, warning
logging only, with the same native status partition. Full API history stays off.
Freeze exact generated inputs before either run; stop on any failed gate. Require
96 complete status scopes and exact phase counts/partition conservation, checked
borrowed owners, normal closure and no fresh fault. This distinguishes tracing-layer
dispatch overhead, not full trace-output cost. Any speed claim needs a separate
clean, repeated comparison. Do not coalesce drains from the old10.9s aggregate.

The status-only engine at`3b4998f326` then passed all156 same-build safety checks
and both frozen117K processes (controller`ad7191936`, plan`2402e8765`). Native
exits/owned closure,96 status scopes, exact phase counts and CPU/wall/work/owner
conservation passed, with no matched fresh fault. All141 benchmark inputs stayed
fixed;15 model assets passed full SHA256 before preparation and after each native
closure. Warning logging was identical; only requested UR tracing-layer presence
differed. These are first-use, parameter-validated attribution samples.

| Status observation | Trace layer absent | Trace layer requested |
| --- | ---: | ---: |
| Whole prompt, ms | 142083.1 | 126889.7 |
| Status scope wall, s | 10.899999 | 10.900710 |
| Copy enqueue wall, s | 10.886858 | 10.886816 |
| Copy enqueue CPU, s | 10.824033 | 10.799420 |
| Prechecks + queue/context CPU, s | 0.000379 | 0.000392 |
| Unclassified main-thread wall, s | 30.264224 | 15.227134 |
| Unclassified main-thread CPU, s | 10.204606 | 10.201809 |

The status interval is concentrated inside `queue.memcpy`, before the separately
tracked checked drain. Its internal runtime/device waiting is unresolved; calling
it CPU arithmetic or pure4-byte PCIe service would be incorrect. The small precheck
and context totals do not justify optimizing those paths. The whole-prompt spread
is retained, including the slower sample. Its large unclassified wall difference
with nearly unchanged CPU needs a separate source/measurement discriminator; the
two observations do not establish tracing overhead, I/O, saturation or a speed
regression against the clean subsequent-request baseline. The status and original
Host regions are parallel views and must not be added.

The [compact status result](../bench/results/2026-10-15-pr6-recovery-audit/prefill-status-current.json)
keeps both samples and nine-phase partitions; the original generated environment,
workloads and model identities are recoverable from the pinned plan. Native
Luna-max curation supports one finite copy/queue discriminator: ordinary C++
versus same-context USM-host destination, each idle versus verified-pending on
the same in-order queue. Bound queue work and attempts; preserve checked drains,
async errors and destination lifetime. A separately confirmed unclassified setup
call allocates blocking staging with cudaHostAlloc; its measured contribution
is unknown. Verify actual provider/view/staging flags before adding a narrow
allocation timer. Neither source finding is an adopted speedup. The old fastest
fallback and status-only candidate remain two generations pending that imminent
artifact-selection check; no third engine build is admitted. Physical262144,
session/RESTORE, long-generation quality and70tok/s remain pending.

The finite production-source copy fixture then completed all four processes
with native exit0, exact guard/payload/phase checks, same-child parameter
validation, fixed inputs, closed owners and no matched fresh fault. The compiled
fixture is source13c30d0c0; controller2f4e25238 and plan161a70f4d correct
the required loader-provider environment before the first native attempt,
without rebuilding or relabeling the executable.

| Component cell | Enqueue wall median, us | Drain wall median, us | Pending before copy |
| --- | ---: | ---: | ---: |
| Ordinary host, idle | 99.558 | 14.558 | Not requested |
| USM host, pending requested | 46.427 | 61.636 | 0/12 |
| Ordinary host, pending requested | 82.827 | 17.583 | 0/12 |
| USM host, idle | 54.273 | 60.960 | Not requested |

**The pending comparison is ineligible**, despite successful execution. All
pre-copy queries found completion after eight128MiB fills. Full tracing and
profiling remain enabled; these microsecond values do not qualify clean speed
or explain the model's10.9s enqueue aggregate. The first producer trace has UR
global-timestamp queries and event releases between fills, but lacks their
individual durations. Next keep all eight events alive through checked drain
and measure producer/query CPU and wall separately, with the same fixed work
and a new committed plan. Do not enlarge or retry the closed workload. The
[compact component result](../bench/results/2026-10-15-pr6-recovery-audit/status-copy-current.json)
keeps all48 useful enqueue/drain samples and the failed eligibility gate.

The separately committed retained-event v2 (source7b8f3f5ab, plan59af47a05,
buildd9650897f) also passes all four bounded native processes and guards,
source/owner/validation checks and no-fresh-fault probes. Retaining all8 events
removes the releases between fills in the first traced producer window, but
both pending cells still observe0/12. The comparison remains **ineligible**.
For USM/ordinary pending cells, median producer wall is0.364618/0.475528ms
and pre-copy getter wall2.258631/1.408892ms (thread CPU2.258766/1.376972ms).
The getter is a material component interval; these measurements do not prove
that it waits for GPU completion or identify the cost within the runtime stack.

Read-only inspection of the exact qualified libsycl and UR-v2 ELF bytes finds
the normal getter forwarding to urEventGetInfo and a status path calling
zeEventQueryStatus. Static calls do not attribute the observed latency among
SYCL, layers, loader, adapter and driver. Native Luna-max curation of both
producer/profile reports accepts the v2 execution checks and preserves the
failed pending eligibility. The next prospective diagnostic compares
`Runtime::make_queue(true/false)` with UR tracing absent/present in separate
processes, keeping validation and safety settings fixed. Use finite events
after checked completion and timed status-query batches with a committed timer
floor, repetitions and ownership/output budgets. This isolates queue-property
and tracing effects at the call boundary, not runtime-layer cost or model speed.
Production queue defaults and elapsed-time consumers remain unchanged. Do not
enlarge the closed fill workload or add a stalled gate.
The v1 executable/raw root was retired after recovering143 fixture/controller
inputs and9836 original native inputs; its compact failed gate remains in Git.

The completed-event v3 fixture uses the unchanged production Runtime/compat
sources and an explicitly owned factory queue. Two traced smoke processes
(profiled/plain) precede three fixed four-cell cohorts: profiled/plain crossed
with tracing present/absent, validation and safety fixed. Each measured process
has four batches of4096 queries after a checked queue drain, retaining the
event and guarded12-byte device envelope through use. Require complete status,
exact payload/guards, provider mappings, first-init validation and normal owned
closure. Each timing batch must reach20us wall and100 times the median128
clock-pair CPU/wall overhead; a shorter batch stays ineligible without retry.
Record actual UR call counts because completed-event queries may use a cached
path. The finite45s process/64MiB raw-text budgets and comparison order are
frozen before execution. This is host API attribution, not a production queue
change or proof of the earlier fresh-event/model enqueue mechanism.

Source69f388655, plan73a659f19 and buildd1233a6ba then pass the two traced
smoke checks and all12 fixed processes:48/48 timing batches are eligible,
with complete payload/guards, exact providers/validation, normal owned exits,
unchanged source and no fresh fault. Independent process medians are:

| Queue / observer | Completed-query wall medians, us/query |
| --- | ---: |
| Profiled / full trace | 11.748614,11.980574,11.894657 |
| Plain / full trace | 11.981615,12.162145,11.801724 |
| Plain / validation only | 0.024876,0.022764,0.022788 |
| Profiled / validation only | 0.019462,0.025150,0.018291 |

Each traced process records16388 UR event-info calls; profiling adds9 global
timestamp calls for4 fills and5 copies, not one per status query. Quiet counts
are unobserved, not proof of bypassed UR dispatch. The completed-query cost
does not explain v2's1.4..2.3ms fresh-getter interval or the model's10.9s enqueue
aggregate. Profiling does not give a consistent all-pair latency direction;
no production queue default or model route is changed. Next isolate the finite
producer/copy/checked-drain path without a pre-copy status query, with a fresh
committed plan and fixed work. Keep the old pending failures intact. Native
Luna-max curation confirms the source contract; its working-set collection is
retrieval-only, with mutable upstream pins and an open guard PR kept distinct
from deployed behavior. The compact result preserves all48 CPU/wall samples.
Both closed standalone roots and consumed audit/dispatch output are now retired:
74 files/24408964B, after independently recovering each fixture's143 inputs,
the9836 native inputs and its exact original compiler recipe from Git. The
pushed retained feature branch preserves the original measured commits while
default integration is blocked. Two still-consumed model engine generations
remain;14 older provenance holds are separate. See the compact result's
path/hash/reason manifest and recovery proof, rather than retaining raw output.

The v4 diagnostic measures the finite producer/copy/drain path without an
explicit pre-copy event query. Four traced64KiB smoke cells precede three fixed
four-cell validation-only cohorts: owned profiled/plain queue crossed with
ordinary/USM-host destination. Keep eight128MiB fill events through the existing
checked drain, separating producer,4-byte copy enqueue and drain CPU/wall with
shared timing boundaries. One separate warmup plus six steady samples per
process yields72 comparison samples. Whole and phase eligibility requires20us
wall and100 times the128 clock-pair median CPU/wall cost. After drain, profiled
events provide checked positive ordered service/summed-span aggregates; GPU
clock eligibility remains distinct from host execution and timing eligibility.
For each host kind compare plain/profiled whole-wall process medians; for each
queue kind compare USM/ordinary. A mechanism screen requires all72 whole samples
eligible, latency geomean<=0.95 and all three paired ratios<1. Moving time from
enqueue into drain alone is no throughput gain. A passing screen permits a
production-consumer audit/next candidate, not model adoption or a third engine.
Keep the old pending-ineligible outcomes and use no adaptive work enlargement,
stalled gate or production queue-default change. Requested fill bytes are not
physical GDDR traffic. Source/recipe/controller inputs and process order were
committed before build/run under the existing finite budgets.

Source`5f31f8882`, plan`bb1dab6d9` and build`5301a62fc` then close all four
smokes and12 measurement processes normally, with72 eligible steady samples,
12 separate warmups, qualified profiled GPU intervals and no fresh fault.
None of the four predeclared mechanism screens passes:

| Whole-wall comparison | Three paired ratios | Geometric mean |
| --- | --- | ---: |
| Plain/profiled, ordinary host | 0.684332,1.007210,1.032128 | 0.892703 |
| Plain/profiled, USM host | 1.072523,1.039981,0.974924 | 1.028334 |
| USM/ordinary, plain queue | 1.036191,1.008676,0.959860 | 1.001075 |
| USM/ordinary, profiled queue | 0.661150,0.976891,1.016180 | 0.869038 |

The first ordinary/profiled process has a3.243ms whole median; later process
medians range2.006..2.300ms. Ordinary-host enqueue medians are1.943..3.162ms,
followed by sub-floor2.4..4.5us drains. USM enqueue medians are sub-floor
10.5..11.3us, followed by1.917..2.216ms drains. Thus the API boundary changes
where waiting is recorded without a consistent whole-path improvement. The
profiled fill-service medians1.947..3.114ms support queued GPU work as relevant
to this fixture's elapsed time; they do not quantify the model's10.9s interval
or establish physical bandwidth, arithmetic saturation or a model speedup.
Keep every paired ratio and the below-floor phase flags. Adopt no queue or
destination change. End this status-copy microbenchmark series and move to
actual production GPU-service/submission/dependency coverage, checking installed
profiler/counter capability and missing command-profile tags first. Clean model
speed, physical262144/session/RESTORE and sustained decode qualification are
unchanged. The compact result retains all72 steady samples and12 warmups.
After recovering143 fixture/controller inputs,9836 native inputs and the exact
original compiler recipe from pushed retained history, Main retired60 external
files/741980B, including this unused build and consumed research output. Its
unused fixture/controller sources are also removed from the active tree and
recoverable at`5f31f8882`. Both model engines remain unchanged; the two consumed
model generations and14 legacy provenance holds remain separate.

Installed-tool discovery now finds VTune2026.6 build633341 and PTI SDK1.2.0;
their presence does not establish usable profiling. The older VTune2026.4 and
perf refusals remain in the previous committed capability result. The controller
at`430aa601f` and generated plan at`dae871acc` pin source, collector, target and
environment before collection. They check CPU software samples with a three-
second OpenSSL workload and GPU collector launch with`/usr/bin/true`. The latter
has no GPU workload and cannot qualify GPU timing or counters even if it exits0.
Neither probe changes permissions or driver settings. Independent read-only
research audits the installed PTI View contract and actual production command-
profile coverage alongside the serialized checks.

Both VTune2026.6 probes then close normally but fail their capability gates:
CPU collection exits1 under ptrace_scope1; GPU-hotspots exits1 because this
Ryzen microarchitecture is unsupported. No model/target GPU work, permission
change or fresh fault is observed. Matching PTI source is now recovered from
official commit`fdc44f65ccfc4c75f951a8f01b1ef9326223aaaf`; five public headers
match the installed1.2 bytes. Its`xpti_adapter.cc` initializer writes XPTI and
UR environment variables when no foreign subscriber is detected, including
replacing`UR_ENABLE_LAYERS` with tracing alone. The ITT initializer can also
write`INTEL_LIBITTNOTIFY64`. This source finding is not a measured loader result:
Main has not loaded PTI into the model. Do not add a late PTI dlopen that changes
the qualified environment. The next bounded candidate is the official Unitrace
Level-Zero collector, with unrelated integrations disabled, all configuration
fixed before target exec and the validation preload retained.

The official Unitrace2.4.1 collector/CLI at`4b6d74294` is now built with L0 on
and XPTI/ITT/OpenCL/OMP/MPI/Perfetto off. Source`09d650542` and plan`5bc84cf8b`
pin331 official source files,457 compile dependencies and43 generated/header
inputs. The earlier GCC program-lookup preparation failure remains in Git;
no model or GPU workload ran. Direct preload can attach to named session
control but cannot create it. The launcher-only overlay at`8fba6f3aa` therefore
preserves explicit Sysman-off, reads the validation preload only for its target
and prepends the collector for the intended reverse constructor order. Its
build is closed; the collector itself is unchanged. Independent review requires
a unique session, cleanup after proven target closure on failure and direct
constructor/retention evidence; its claimed early Sysman override is corrected
because the guarded setter covers that call too. CPU-probe source`1880b13d3`
and plan`8113b69c6` then pass in native PID376891: validation init/first-UR-init
proof precedes collector init and canary body, while collector fini follows
canary completion. The exact session is removed, owners close normally and no
fresh fault occurs. No GPU command or model runs. The compact result retains
the seven ordered loader/validation lines. After original18 controller inputs
and9836 native inputs are recovered,51 normal build/loader and consumed research
files/173699B are retired. Current collector/header inputs and the methods
collection remain consumed by the next GPU qualification. Loader success cannot
qualify temporal gating, GPU timing, graph/queue coverage or32K+ production
attribution. Both model engines, clean speed evidence and full-context/quality
obligations are unchanged.

GPU coverage now has two closed attempts. The first fails UTC containment and
host-flow gates despite recording all606 submission kernels. The prospective
raw-clock attempt at`e54d9a168`, plan`3fbc13928`, records606 submission and62
native-graph kernels, with normal numerical checks, owned closure and no fresh
fault. Its strict graph host-sink gate still fails; preserve both failed runs.
The separate closed-data analysis at`63bd98618` conserves every kernel, copy,
fill, barrier and requested byte after expanding captured graphs by actual
replays. All61 graph append APIs account for their183 unsunk non-barrier nodes.
This establishes fixed-fixture service-record coverage, not complete graph
completion edges, a production critical path or a speed improvement.

The Main-authored recording-window wrapper/controller at`626990208`, with
check plan`b4ce2f00c`, now passes two fresh processes. Both execute all606 kernels
and numerical guards; paused-throughout records no GPU work, while external
resume at ready/pause after drain records606 kernels,53 copies/55844 requested
bytes, one1028-byte fill and20 barriers. All records have unique IDs, host
sources, positive kernel service and containment within the controlled raw
window. The selected device IDs/UUID and trace PCI lane identify B570; caller
and target share the observed time namespace. Paired-host samples lie inside
all raw-clock brackets, with best widths23.765..24.446us per phase; the worst
single bracket is126.217us. The device reports19.2MHz and64-bit global/kernel
timestamps. These observations do not establish nanosecond accuracy or complete
graph completion edges. Native and control commands close normally, exact
sessions are removed and no fresh fault occurs. The initial missing-header
compile failure remains in Git at`aef9e43f6`; it executed no GPU work.

The production controller at`f9f3d12c2`, with plan`982592026`, now closes one
official`bench.sh` request through this control path. The updated target wrapper
first passes a CPU-only constructor/validation/retention canary; nine benchmark
host tests pass too. Recording remains paused through loading and warmup. The
long request reads39263 fresh tokens, reuses none and generates24 thinking
tokens without an answer. Its instrumented prompt rate is672.4tok/s; this is
neither a clean speed comparison nor sustained decode or quality evidence.
Native PID561204 exits0, owned processes close, the exact session is removed,
all15 model hashes match before/after and no fresh driver fault occurs.

The266967750-byte timeline contains206882 GPU records, including172646 kernels,
and493490 host API records. Every recorded GPU operation has a unique ID and
native host source, the B570 PCI lane and containment within the controlled
whole-request RAW interval with the declared1ms allowance. This passes the
collection gate only. Closed-data analysis at`d4a993fb8` now binds the official
journal/sample, verifies all48 BF16 GPU-residual layers/five chunks and ordered
restoration, filters host events by native PID and request extent and finds no
GPU event outside the exact controlled window. Every GPU record has a host API
parent, but direct append counts differ because graph replay expands commands;
complete capture/dependencies and hardware saturation remain unqualified.

Before first stream output, the observed58.703s interval contains45.690s of
GPU service, including8.126s of copies, with only12us of kernel/copy overlap.
Observed GPU-empty time is13.013s. API wall intervals intersect5.332s of that
empty time; follow-up`79f9e821c` attributes3.977s to`zeModuleCreate`, rather
than proving CPU arithmetic or submission is the steady bottleneck. Kernel
name geometry is workgroup count then group size, as official
`ze_collector.h:GetZeKernelCommandName` shows. The first QSA source report's
global-work-item interpretation is invalid. All120 prefix records reconcile
as60 attention passes and60 preparation passes across five actual shapes.

The same follow-up identifies36 exact Q80Block`prompt_product` records taking
4.174s. The existing MTP`draft_kv` path uses these Q8_1-activation integer dot
products; its opt-in`STRATA_MTP_BATCH_F16=1` instead dequantizes Q8_0 weights
and uses FP16 GEMM. The bounded diagnostic at controller`dbea9dd0e`, with
plan`d21dd2d30`, now passes the existing route after a successful small batched
draft-KV warmup and actual native-environment checks. It changes MTP KV
numerics, not the verifier arithmetic for a fixed candidate window; draft
candidates, acceptance and generated text can still change.
GEMM scratch precedes the separate temporary region; source bounds and same-queue
drains remain intact in the reviewed path. The warmup processed73 draft-KV cells
for74 read tokens. Its gate checked positive batching but omitted explicit cell
conservation; the recorded73/73 cells do match. Add that check prospectively
before another run rather than changing this diagnostic's original criteria.

The long request reads39266 fresh tokens, with39265 batched residual rows, and
generates24 thinking tokens without an answer. Instrumented PP is756.8tok/s;
the five draft-KV batch timers total307.2ms. Native exit0, owned closure,
original source/fixture/runtime/model checks and no fresh GPU fault all pass.
This uses parameter validation, trace and draft timing, whose entry synchronize
is an observer. Input and observers differ from the earlier672.4tok/s profile,
so the two readings do not establish a speedup. No clean speed, quality,
acceptance or adoption claim comes from this diagnostic. Next compare FP16 off/on
with identical inputs and observers in repeated independent processes, then
check sustained generation and MTP acceptance separately from prefill.
The [compact diagnostic result](../bench/results/2026-10-15-pr6-recovery-audit/prefill-mtp-current.json)
preserves the actual samples and gates. Collector loss limits, the loader symbol-lookup
diagnostic and observer overhead remain separate. Both model engines and clean
speed samples are unchanged; physical262144/session/RESTORE, task quality and
sustained decode remain pending.
The production attribution and pilot consumers have closed. Original21 window
inputs and its build plan,8698 compiler inputs,149 production-controller inputs
and the current9836 native inputs/eight fixtures were checked before retirement.
The four private closed window/model/analysis source files are retired to pushed
`ed9368b3c`; maintained model qualification does not import them. The serialized
cleanup removed the complete production timeline, standalone window
executable/compiler manifest and normal FP16-pilot output. Its path/hash/reason
manifest records156 files/294666727B. Post-cleanup native source/fixture
admission passes. Installed
profiling tools, both consumed model generations and14 older provenance holds
are excluded. Cleanup does not establish new validation or hardware attribution.

The matched MTP comparison froze six independent processes in off/on order
`0,1,1,0,0,1`, using official thinking-enabled regular-code long generation.
Each process had four official warmups and one fresh33210-token long request.
Parameter validation, trace and draft timing matched in both arms; the timer's
entry synchronization is an observer. These are diagnostic first-long-request
samples, not clean speed or subsequent-long performance.

| Process | FP16 MTP | PP, tok/s | TG, tok/s | Generated | Admission |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | off | 643.9 | 10.1 | 2913 | passed |
| 1 | on | 690.6 | 8.6 | 3790 | passed |
| 2 | on | 685.0 | 9.4 | 3790 | passed |
| 3 | off | 632.6 | 9.4 | 2913 | failed |

The fourth native process exits0, with a complete journal, owned closure and no
fresh fault. Its final qualification fails because Main edited unrelated
research dispatch metadata while`ur_validation.preload_qualification` required
the entire research index to be committed. Preserve that failure and all four
samples; processes4/5 are not run and the six-process performance decision is
ineligible. Do not retry this root or retrospectively qualify it.
Source`4d4792007` narrows the reader to its consumed preload-component subtree;
seven host tests pass, including unrelated edits and changed/missing/type-altered
qualification fields. Original9836 native inputs, eight fixtures, engine and
preload qualification pass again without new GPU execution. The original cohort
remains failed and cannot supply the new comparison. Its full
source/plans/evidence remain reachable at`b62f673a7`; the maintained result keeps
the four useful individual samples and original rejection. The closed raw root
is retired after147 original controller inputs and pushed ancestry were checked:
282 files/51589605B, with a path/hash/reason manifest in the compact result.
Both consumed native generations and required fixtures remain.

The first pair's7.26% PP increase is an observation, not adoption. Its30.1%
generation-length difference exceeds the declared5% decode band. The two FP16
processes match output/reasoning hashes,2725/3192 accepted drafts and logical
expert counters despite different decode times. Keep slower samples without
calling them noise. These TG readings are not comparable with older14.6tok/s
clean samples. Weak-MTP, agent-session, quality, physical262144/session/RESTORE
and separate clean evidence remain pending.

Controller`dc8805e94` prepared a minimal-marker comparison with six new
independent processes in the same off/on/on/off/off/on order and the same official
thinking-enabled>=32K regular-code workload. Trace, draft timing, UR layers
and the private validation preload are absent; the qualified runtime safety
settings remain. Existing`DRAFT_PREFILL` success markers report batched work
with exact8192/tail cell coverage per request, including all four warmups. Source
review finds no added GPU wait in that marker and excludes the heavy restore
verification branch through prompt-cache0/conversation-cache-mib0. This is
minimal-marker timing, not proof of zero observer cost or an inner kernel.
The prior diagnostic samples cannot enter this cohort. Nine synthetic host tests
passed, but the first native attempt fails its warmup gate: Main incorrectly
required mode0. The actual four markers are mode2, with73/265/911/1068 cells for
74/266/912/1069 read tokens. Source`QsaState::kv_mode` describes placement,
not storage precision:2 is the MTP ring;0 is fully resident. The first long gate
is not released; cohorts1..5 are not run. Owner cleanup terminates the attempt,
with an anchored empty-session scan, direct-child reap and no original identities
remaining. Its original observation/ownership failure fields remain unchanged;
no natural native exit or speed sample is claimed. No fresh GPU fault is found.
The original failed plan/result is recoverable at`b29ece914`. Current controller
`28bb3ea73` uses a fresh v2 root and requires mode2, rejecting a fully-resident
fallback0 as a placement change. Ten host tests include all four real warmup
marker/footer pairs; replay against the actual closed stderr also passes.
Markers establish nominal callback coverage, not internal needed rows or kernel
batch size. The fresh six-process plan is committed at`f038a0f34`; no old samples
enter it. All six independent processes now pass with33208 fresh input tokens
and the same request hash in both arms:

| Process | FP16 MTP | PP, tok/s | TG, tok/s | Generated | Accepted/offered | TTFT, s |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| 0 | off | 642.3 | 14.5 | 3900 | 2787/3342 | 51.816 |
| 1 | on | 704.0 | 11.8 | 3318 | 2347/2916 | 47.289 |
| 2 | on | 694.6 | 14.2 | 3318 | 2347/2916 | 47.929 |
| 3 | off | 634.9 | 9.2 | 3900 | 2787/3342 | 52.484 |
| 4 | off | 637.2 | 10.9 | 3900 | 2787/3342 | 52.321 |
| 5 | on | 699.8 | 12.9 | 3318 | 2347/2916 | 47.596 |

Each process has an answer and thinking, four warmups, five ring callback spans,
the all-GPU BF16/mode10 prompt route, a valid journal, native exit0, closed owners
and no fresh fault. Original controller/native/fixture inputs and all15 model
assets pass the final checks. These are first-long-request minimal-marker
samples, not subsequent-long steady state or proof of zero observer cost.

The three raw-clock PP ratios are1.096054,1.093921,1.098182; geomean1.096051
(**9.6051%**). The frozen prefill faster screen passes. Medians are637.2Q8 and
699.8FP16, with sample SD3.7873/4.7089tok/s. This33208-input result does not reach
1000tok/s. TG medians are10.9/12.9, ranges9.2..14.5/11.8..14.2. Its paired ratios
0.816328,1.541097,1.181395 have mixed directions; generation ratios0.850769 all
violate the prospective5% band, and decode nonregression fails. The1.141204 TG
geomean therefore does not establish faster decode. Acceptance falls83.3932%
to80.4870%, still above the declared60% strong-MTP criterion. Same-arm answer,
thinking, acceptance and logical-work counts match despite large timing spread;
its cause remains unresolved. Preserve every reading and keep FP16 unadopted.
Broader task quality/weak-MTP/agent/physical262144/session/RESTORE and70TG remain
pending; Q8 MTP stays the baseline for the next independent prefill candidate.

All six target-prefill completion markers conserve19397 expert uploads and
39629260800 requested H2D bytes, zero permanent hits and identical GU/Down job
totals. These are logical bytes, not measured PCIe traffic. This input has one
FFN accumulation window, so packed capacity alone cannot save cross-window
reloads here. The next reviewed candidate concerns a private layer-local layout
within the existing bank allocation, with demand-preserving copy coalescing as
a separate mechanism. Qualify its stride/slot/consumer and checked layer-drain
contract before execution; test cross-window reuse on multi-window long input.
No new allocation, implementation or throughput gain is established yet.
The closed raw root is retired after all147 controller inputs, six native/journal
receipts and generated configuration bytes were checked:414files/76154934B.
The unused396 executable (22079288B) is also retired after9813 original source
inputs and its exact recipe were recovered. Its source receipt remains for
possible reference reconstruction through the next packed-bank comparison.
Current3b and its eight required fixtures pass post-cleanup source admission;
this is not new GPU validation. The14 older provenance holds remain separate.
Controller`77396f090` now binds fallback retention to the same-build status
observations and closed six-process MTP result, rather than older engines'
top-level long-input/generation records. Eight host tests pass, including
rejection of an older source/binary at the same output path, incomplete owners,
faults and insufficient input/output. All9836 original inputs and the recipe
were recovered before copying the current engine/eight fixtures to external
`known-good-3b4998f32610`; every artifact hash matches. This is the same retained
generation, not a new native build or GPU validation. It provided the fallback
for the packed-bank candidate. After the clean cohort below, its remaining
immediate consumer is a matched long-generation reference check with thinking
and at least1K generated tokens before the next native build; retire the extra
after selecting the validated generation. Clean117K speed,task quality and physical262144/session/RESTORE
remain unqualified for3b.

The next opt-in candidate,`STRATA_SYCL_PREFILL_EXPERT_BANK_PACKED=1`, uses a
private per-layer view of the existing allocation. It does not change the
general decode cache layout or allocate another bank. The upload destination
and dequantization consumer use the same checked private view. At physical512
slots it explicitly admits512 permanent experts without a scratch slot;
smaller views retain the existing single scratch policy. Rebinding follows
the checked layer drain. Source and destination strides remain independent,
including the owned maximum-stride slab fallback. Host tests cover all seven
model formats and the unchanged partial-residual/short-tail/262144 planner
fallback. Small traced GPU byte-reader tests precede actual-size transition,
padding/tail, immutable-source, private-map and unwind checks. CPU source
`cf8ae547a` passes all seven format plans, two layers of full512-expert scans
per format and19 new rejection checks, retaining the original18 ticket checks.
Native build`44eac4cc0` then compiles the engine and nine fixtures successfully;
its first GPU attempt stops before native execution: the older bank controller
does not consume the already-qualified Oct19 runtime asset delta. Preserve
that failed admission in the compact result. The revised controller verifies
the exact assets against the passed3b component result without changing the
shared baseline hashes, using a fresh v2 root. Native Luna-max report review
accepts the private layout with targeted followups: all-layer stride preflight,
full-length copy coverage and rejected-enqueue nonpublication/checked unwind.
Main adds those gates; seven runtime-admission host tests pass and amended
native build`e0623d4f7` compiles the engine/nine fixtures. The v2 traced small GPU
check passes two changing-layout cases (43/17 uploads/hits and10/0 on unwind),
full-copy/guard/source/private-map checks, and one rejected-enqueue checked
unwind without publication. Owners close with native exit0, unchanged boot and
no fresh driver fault. Actual-size validation then passes: a512/384/256/512
transition uses2434 uploads/1662 hits and8192 queued readers; the single
1022361600-byte bank with1510400/2662400/2176000-byte strides uses1709 uploads,
1363 hits and6144 readers. Exact-full-copy, all padding/unused-tail words,
source immutability, private map, two-layer unwind and11 rejection checks per
case pass. The synthetic failed-enqueue case confirms no mapping publication
and checked retirement; it does not inject a real provider failure. Native
exit0, closed ownership, unchanged assets/boot and no fresh fault hold. This
qualifies the upload/layout component, not model arithmetic or throughput.
The shared safety suite then passes156 checks on build`ac904ae09`. This is a
controller-only rebuild: the engine and all nine fixture hashes, native inputs
and recipe match`e0623d4f7`. The new117K controller uses two independent off/on
processes with Q8 MTP, checking the actual private layouts, both accumulation
windows, request journals, immutable inputs, decode/cache/dense restoration
and completed logical upload totals. Its diagnostic output cannot qualify
clean throughput. The first baseline process completes its117489-input/24-output
request (diagnostic920.7 PP), native exit0 and no fresh driver fault, but the
new controller then calls a nonexistent`generation_engine.native_end` helper.
The original comparison remains failed/incomplete; the packed-on arm is not
launched. Main replaces that call with an explicit start/exit receipt and
binary/argument/boot/probe/owned-session identity check, adds an offline
closed-benchmark collection regression, and requires a fresh v2 off/on plan.
Five host regression tests and offline replay of the captured baseline pass;
the failed v1 status remains in`fdbcc5e6b` and its74 normal-output files
(13351468B) are retired after149 original controller inputs are recovered.
The corrected frozen v2 plan then passes its baseline process:117489 input,
24 thinking tokens, diagnostic921.3tok/s,19631 uploads and40102860800 requested
H2D bytes with13594 permanent hits. It verifies96 borrowed retirements,two
exact payload/map cycles,dense/cache restoration,native exit0,closed ownership,
full model hashes before/after and no fresh fault. Its96-window census conserves
37353 jobs and56394240 routed rows. The packed-on process also passes all gates,
including48 actual private layouts with the predicted seven-format layer census.
Both arms conserve33225 logical expert lookups and identical37353 jobs,
56394240 routed rows and GU/Down fusion totals. Uploads fall19631 to18937;
requested H2D bytes fall40102860800 to38673152000, saving1429708800B (3.5651%).
Both native processes exit0 with closed owners, unchanged full model hashes and
no fresh fault. Their diagnostic PP is921.3/920.3tok/s; this is not speed-gain
evidence. At the achieved6GB/s microbenchmark reference, the saved bytes alone
represent0.2383s of copy service, not a hard ceiling or measured critical-path
gain. The work screen passes, but neither clean speed nor model adoption is
qualified. Use a separate matched six-process clean comparison, retaining
first-long separately from three subsequent requests per process.
`tools/sycl/packed_bank_compare.py` prepares that independent comparison:
off/on/on/off/off/on, four fresh117489-input requests per process, Q8 MTP and
identical native binary/cache placement. The prospective faster screen requires
all three subsequent-median pair ratios>1 and geomean>1.02; the slower screen
requires all<1 and geomean<1/1.02. Otherwise it is inconclusive. Nonregression
requires geomean>=0.98 and every pair>=0.95. The1000 milestone separately
requires all three candidate process medians>=1000tok/s. Stop after any input,
request,route,work,owner,fault or budget failure; never retry a root. Each pair
must conserve input bodies and completed MoE work and reduce requested expert
H2D bytes. Native startup keeps the qualified safety settings; parameter
validation,readbacks,profilers and trace observers are absent only for timing.
Full model digests bracket the cohort; inventory,inode,size,mtime and ctime are
checked before/after each process. Commit generated configuration and the full
frozen plan before running. These short outputs cannot qualify decode or model
adoption; numerical/task and physical262144/session/RESTORE remain separate.
Five offline tests pass for the clean environment,real shared route reader,
request bounds,work/owner refusal and separate speed/target decisions. Same-build
qualification recovers9836 original native inputs and verifies the current
engine/nine fixtures; this is controller validation,not a new GPU execution.
After recovering149 controller and9836 native inputs, Main retires the closed
paired diagnostic's141 normal-output/JIT/configuration files (26289776B).
All19 current/fallback native artifacts remain hash-matched. Only40839B of
conserved aggregate MoE job sizes remain outside Git for the named next
matched-GEMM ceiling fixture,with a131072B budget and review at clean-cohort
closure; remove it if that fixture is not selected. Full layers/traces are not
retained. Clean admission consumes the compact paired result and retained
engine/fixtures,not the deleted diagnostic root.
All six clean processes are closed. Each passes the native/request/work,
observer-absence,owner,runtime/model-identity and fresh-fault gates:

| Process | Packed | First-long tok/s | Median of three subsequent requests |
| --- | --- | ---: | ---: |
| 0 | Off | 836.176 | 999.889 |
| 1 | On | 929.917 | 1002.056 |
| 2 | On | 928.963 | 1002.106 |
| 3 | Off | 926.748 | 1000.209 |
| 4 | Off | 927.347 | 999.525 |
| 5 | On | 927.958 | 1002.149 |

All three pairs conserve the four input bodies and logical MoE work; requested
expert H2D bytes fall in every paired request. Subsequent-median ratios are
1.0021672,1.0018969 and1.0026255; geomean1.0022298 (+0.2230%). Nonregression
passes, but the prospective2% faster screen does not: the speed decision remains
inconclusive/small effect. All three candidate medians exceed1000tok/s, meeting
the prefill throughput milestone for subsequent fresh117K requests with Q8 MTP.
Keep first-long separately; its gap remains unexplained. Full model digests
match across the cohort, all native exits are0 and no fresh fault is observed.
The24-token outputs do not qualify sustained decode,task quality,weak-MTP/agent
sessions,physical262144 or session/RESTORE. Model adoption remains unqualified;
this result does not change the packed-bank opt-in default.
This targets the observed expert-upload service, not proven physical PCIe
saturation. Two-window117K work confirms fewer requested copies; actual physical
traffic and critical-path copy service remain unknown. After recovering151
controller and9836 native inputs, Main retires the six-process cohort and
closed research transactions:425files/76769421B. All19 current/fallback native
artifacts remain hash-matched. The compact result keeps all24 PP clocks and
the same reducer decision; unused parser aliases,short decode rates and normal
probe details remain recoverable from`da23ad440` rather than the active file.
The next maintained`tools/sycl/generation_compare.py` diagnostic uses the retained3b
and currente062 engines without rebuilding. Each process runs both official
32K-input long-generation cases with thinking, a16384 output allowance and
four official warmups. It prospectively requires actual answer/output bounds,
MTP batched-ring work, GPU BF16/mode10 placement, dense/cache restoration,
source/runtime/model binding, native closure and no fresh fault. One sample per
case/arm is a pilot, not a speed/nonregression or strong/weak classification.
The first preparation at`5e5bf9f7a` stopped before creating a run root or launching
a native process: its new reader selected the outer historical MTP diagnostic
instead of the closed nested comparison. The corrected reader uses the original
comparison and adds an archive-backed regression check; no original evidence
or admission criterion is changed.
At`95cea638a`,36 host checks pass across the generation reader, MTP marker,
packed source/layout, new collection path, retained-build and resource gates.
The new path exercises the actual shared native-end reader and rejects fresh
faults; the archive-backed check selects the original nested MTP comparison.
These are host checks, not new GPU validation.
Both arms of the [frozen generation reference](../bench/results/2026-10-15-pr6-recovery-audit/generation-reference-current.json)
are closed. Each process runs both workloads with thinking enabled, fresh input
and an actual answer body:

| Engine | Workload | Read tokens | Generated tokens | PP tok/s | Decode tok/s | Accepted/offered |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Retained3b | regular-code | 33214 | 3611 | 635.9 | 12.3 | 2578/3102 |
| Currente062 | regular-code | 33214 | 3611 | 635.2 | 14.4 | 2578/3102 |
| Retained3b | varied-prose | 33288 | 8638 | 761.5 | 9.3 | 4660/11931 |
| Currente062 | varied-prose | 33288 | 8638 | 745.1 | 9.3 | 4660/11931 |

All four requests pass the native/draft/GPU-route/cache-restoration gates.
Both native exits are0, owners are closed, full hashes of all15 model files
match before/after the cohort and no fresh fault is observed. The observed
83.11%/39.06% acceptance falls in the requested bands in both arms. One sample
per case/arm, with diagnostic observers, does not qualify MTP classification,
a speed/nonregression decision, task quality, agent sessions, physical262144,
RESTORE or model adoption. Preserve the lower prose PP and the single-sample
code decode difference without attributing either to a proven implementation gain.

Post-closure replay at`974c45c67` binds existing trace windows and native decode
footers to the actual requests; no new GPU execution follows from this analysis.
Both engines perform the same logical decode work. Code has1370846 CPU unique
expert jobs for3611 emitted tokens (379.63/token); prose has4605780 for8638
(533.20/token). At least70.75%/49.03% of those jobs have one input row, from
`max(0,2*jobs-entries)`. Prose requires1.842 target rows/emitted token versus
1.146 for code. These counts include the initial one-row verifier and speculative
verification, exclude separate MTP work, and are not physical DRAM/PCIe traffic
or proof of CPU/RAM saturation. Zero requested target-expert copy bytes under
pcie0 does not imply zero activation/routing or other PCIe transfers.

Select the currente062 engine/nine fixtures for the next bounded resource
attribution diagnostic. After committing the samples and recovering the exact
9836 old native inputs and original recipe/compiler/environment from`751c81802`,
Main retires the3b engine/eight fixtures, source receipt, successful cohort raw
logs/JIT and completed scope-review transaction:152files/114331335B. The deletion
manifest is pinned in the compact generation result; original outcomes remain
unchanged. Only the selected engine/nine fixtures remain for this comparison.
The14 legacy provenance holds remain separate, pending before any new build.
The source-only resource diagnostic runs existing decode stage timing and passive
CPU/DRM counters on one current regular-code request, with no native rebuild.
The native and collector exit0, owners close and no fresh fault is observed,
but original controller collection fails its rounded-clock reader. The source
prints DONE totals to0.1ms and timing averages to0.01ms/window; a one-window
warmup's126.0ms/126.05ms pair exceeds the reader's incorrect0.00501ms bound.
Preserve this failed outcome; fix the host reader against both documented print
resolutions and replay the existing capture, including model post-hashes, before
qualifying attribution. No GPU rerun is required for that parser correction.

Its41 initial host checks pass. The earlier real admission stopped before native
launch because the historical build inventory includes the updated Python
resource-gate test. Admit only that named committed host-only test, record both
hashes and its revision, and keep every native input/artifact check exact; do
not refresh the original build receipt. At`9c323c437`, nine follow-up host checks
pass; actual admission recovers9836 native inputs and matches all10 current
artifacts. Both original generation arms replay their official/request/route/
work/native-closure gates;151 controller and9836 retained native inputs recover.
At`5a2a22dd5`,20 host checks pass, including the actual one-window quantization
pair and refusal beyond either print interval. Post-closure replay preserves the
original failed controller outcome at`8d3083970`; its separate analysis passes
the official journal, all four warmups, route/work/native/collector/owner gates,
152 original controller inputs,9836 native inputs and all15 model post-hashes.
It executes no new GPU work. The actual long request generates3611 tokens at
14.5tok/s with2578/3102 accepted; its diagnostic PP is631.4tok/s.

Each of1035 target windows takes240.10ms:112.77ms inside the target CPU pool,
75.02ms waiting for GPU reach,33.17ms MTP draft and19.14ms other outer work.
These are host dependency scopes, not additive hardware service costs. The
contained after-first-output interval has1294.72 CPU seconds over246.79135 wall
seconds (5.2462 cores) and CCS busy fraction0.564849; before first output it has
49.03 CPU seconds over52.13603 wall seconds (0.9404 cores), CCS0.825179.
Scheduled CPU includes polling/stalls and CCS busy does not establish arithmetic
or VRAM saturation. Default pool workers spin for20ms before sleeping; high
scheduled CPU alone cannot be labeled useful expert math. The source's current
SYCL mixed route separates graph segments instead of spinning on host flags.
Passive read calls total0.66846s after first output, not a causal overhead bound.

CPU arithmetic versus RAM bandwidth remains unresolved. The direct dispatcher
census below now supplies the actual quantization-pair/NT mix for selecting a
bounded CPU comparison. Existing unframed routing output is insufficient;
use actual job NT rather than callback row count or TLS as request identity.
Current pcie0 requests no target-expert weight uploads; other bus traffic is
still unknown. No clean speed, hardware-saturation or model-adoption claim
follows from this diagnostic.
After recovering152 original controller and9836 native inputs, Main retires the
closed resource capture/JIT and consumed review transaction:73files/15463239B.
The compact result preserves both the original controller failure and the
successful corrected post-closure analysis. The current engine/nine fixtures
still match. The following counters improve attribution, not throughput.
The selected implementation is now `CpuWorkCensus`, attached explicitly to the
serialized request after prompt refill. `STRATA_CPU_WORK_CENSUS=1` is fixed
before exec. It counts actual final `ExpertJobMulti.nt` after pool completion
and before the job map is reset, with no new GPU readback, event, wait or math.
The bounded snapshot records exact resident membership and runtime format;
every callback checks unchanged slot mappings. Request/window/layer ordering,
job-map/NT counts and existing decode footer deltas must conserve. Missing
callbacks, unknown kinds, changed residency/format, wrong-thread use and uint64
overflow invalidate the diagnostic. Zero-CPU windows still require callbacks.
This first diagnostic supports the serialized native single-GPU static-cache
row-split route; adaptation, peer/remote and split-device routes are rejected.
It does not infer a request from TLS or reconstruct unframed routing records.

Before rebuilding, the original current engine and nine fixtures are retained
at the then-committed `retained_validation_reference` in the build result.
All9836 original source inputs/recipe and ten copied artifacts were verified;
seven host retention checks pass at`8a3ad20e3`. The immediate consumer is this
census build and its component/model qualification, not indefinite extra storage.
Build with `python3 -B tools/sycl/build_engine.py rebuild-cpu-census`; its bounded
CPU-only contract runs before engine compilation. Then use the new finite
`check_verifier_retirement.py --cpu-census` component gate (the existing suite,
15 census host cases and14 actual dispatcher cases), the dedicated packed-bank
gate and a small traced model check before any32K thinking-generation census.
At`63af639ac`, the build and15 CPU-only cases pass. The subsequent18 component
groups pass, including14 real dispatcher census cases and all existing queue,
arithmetic, scratch/cache restoration and expected fail-stop contracts; no fresh
fault is observed. Actual source admission recovers9838 inputs and matches the
engine and ten fixtures. Six reader checks at`3120b27d2` pass, including static
profile/reverse-map admission, 64-bit membership word order, NT/format/clock
conservation and refusal of incomplete or mismatched records.

The controller at`386ef54e8` and frozen plan at`b2c78fa48` subsequently passed
fresh packed-bank checks, all four official warmups and both thinking-enabled
long-generation cases. Every prior request's counters passed before the next
was released. All9838 native inputs, eleven artifacts, controller inputs and
fifteen model assets passed final admission; native exit was0, all owned
processes closed and no fresh GPU fault occurred. The
[current exact work result](../bench/results/2026-10-15-pr6-recovery-audit/cpu-work-census-current.json)
preserves the two samples and seven format bins per case:

| Workload | Fresh input | Generated | Diagnostic PP, tok/s | TG, tok/s | MTP accepted | NT1 CPU jobs | CPU-pool wall share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| regular-code | 33214 | 3611 | 634.3 | 14.5 | 2578/3102 | 77.24% | 46.98% |
| varied-prose | 33288 | 8638 | 746.4 | 9.3 | 4660/11931 | 67.40% | 44.38% |

These include parameter validation, trace and counters; they are not clean
speed or a new throughput gain. CPU pool time includes scheduling/waits.
GPU-reach waits take about75ms/window and the separate MTP draft about33ms;
those are host dependency intervals, not device arithmetic utilization. There
are no requested target-expert PCIe uploads on this pcie0 route; other traffic
is unknown. Physical262144, broader quality, agent-session and RESTORE
qualification remain pending.

The old CPU calibration projected through the exact format/NT counts gives
112.66..119.42s pool wall for code (observed117.24s), and399.27..420.42s for
prose (observed410.52s). Hot/stream figures are projections from six-job calls,
not matched production batches, physical RAM bandwidth or a hardware ceiling.
CPU kernel sources are unchanged between the calibration and census builds.
IQ2_S contributes42.01%/43.06% of NT1 jobs and39.91%/40.27% of observed pool
wall in code/prose. This updates the pre-census IQ3_S-first priority.
The exact IQ2_S NT1 subset component at source`ef58ccb55` passes1048576 signed
integer cases,1280 padded synthetic row dots and96 real experts' GU/SwiGLU bits.
Three independent processes and108 setup-inclusive hot/stream samples reject
both candidates: scalar int16-table latency is4.66153 times the current direct
helper, and AVX2 int32 gather5.24602 times. All six process/access medians
regress. The original samples,prospective gate and outcome are recoverable at
`8d186fc98:bench/results/2026-10-15-pr6-recovery-audit/cpu-iq2-current.json`.
No production-pool,engine or model trial follows this negative result.
The subsequent direct-fusion component at source`ad4f29ac9` shares gate/up
activation loads and loop control while preserving integer lanes and FP32
FMA/reduction order. Both variants pass the same numerical gates and108 samples
in three independent processes. Rolled-loop latency geomean is0.778435 and
fully expanded0.728357 versus the existing direct helper; all six process/access
medians improve. These are single-core GU/SwiGLU component measurements, not
model throughput, production-pool gains or hardware saturation.
The single-core result is recoverable at`69caeabd9`; its integer sweep uses
part0 and repeated sign bytes,with full synthetic rows and real experts covering
broader layouts. The checked producer is now connected to
the CPU build and dispatcher under default-off`STRATA_IQ2_PAIR=1`, after the
existing multi-row dispatch. It only replaces IQ2_S NT1 on the supported Zen3
target; other types,NT2+ and Down keep their existing paths. Byte spans,
alignment and nonoverlapping storage are checked before producer writes.
`STRATA_IQ2_PAIR_AUDIT=1` adds optional per-thread dispatch/work counters and
actual affinity evidence; its environment is fixed before exec. Thread-local
registration identifies threads,never requests. No inference engine has been
rebuilt or model route qualified by this source connection.
The [current full-pool result](../bench/results/2026-10-15-pr6-recovery-audit/cpu-iq2-current.json)
uses source`47de83c92` and committed plan`50a445328`, with both22/20 and22/42,
five workers plus the host,
28-job NT counts22/5/1/0 and24-job counts16/5/2/1. Repeat uses24/28 distinct
blobs; stream uses168 without reuse per pass. Neither repeat set is claimed to
fit L3. All288 samples in six independent processes pass work conservation,
actual NT1 pair dispatch,six-core participation and minimum timing duration.
Sixty producer-contract cases per check process and all full outputs pass;
all eight9,461,760B captures have SHA256
`6877ca0e7cac9f29cd8881a87d4f17836077e316b9fb763ba463d553b9ca2ad3`.
Full-pool latency ratios are0.911505 for IQ2_S/IQ4_NL and0.927087 for
IQ2_S/Q2_0:8.85% and7.29% less time. Both pass their separate prospective
3% gates,including all per-cell and independent-process nonregression checks.
The failed first compile is preserved at`62d068c73`; the corrected source and
fresh plan do not relabel or retry that attempt. This permits an inference
rebuild,not model adoption. The previous comparison at`18836f101` used
`STRATA_IQ_SINGLE_EXACT=1` while the actual model census uses0. Its result
remains valid within that scope. This fresh comparison fixes0 in both arms and
resolves that advancement hold without rewriting the old result.
No model execution has used the new producer yet. The`6c7f7f36d` candidate
passes18 same-build safety groups with9845 recovered inputs and no fresh GPU
fault. Its native byte origin is`b0b39c0e1`; the later controller-only build
preserves engine and fixture hashes. Optional drained,request-bound IQ2 audit
snapshots are ready for the observed two-workload model pilot; seven host reader
tests reject framing,identity,dispatch and counter failures.
Physical CPU/RAM saturation and whole-model speed gains remain unmeasured.
Closed CPU component builds,tensor captures and consumed review reports are
retired after original source/recipe recovery; individual samples and deletion
evidence remain in the linked compact results and Git history. The validated
`63af639ac` engine/ten fixtures are retained,with a hash-checked fallback for
the imminent IQ2 pair inference rebuild. All9838 original inputs and the recipe
were recovered. Eleven controller tests preserve same-build identity,owner,
fault,thinking/output and route gates; this does not qualify new inference code.

The14 old worktree proposal/metadata paths are also reviewed before rebuilding.
Available unique partial failed-GDN source/recipes are archived at`228b85f13`
and removed from the active tree; missing original headers/fixture/v2 source
remain an explicit recovery failure, with no GPU/model performance claimed.
Other unused proposal snapshots, metadata and duplicate failed-build reports
are retired with path/hash/reason evidence in the current build result. The
separate six legacy product/source-recovery exceptions retain their original
receipts/unique evidence; they are not qualified comparison generations.
Generic native-end and draft-work readers now live in`generation_engine.py`.
The closed packed model/clean launchers and their retired-run-only tests are
removed; source and original reducer remain recoverable at`6213f0325d807b5eaf319bbf8096c9337f5402bd`.
The shared packed source/layout evidence and its negative tests remain for this
immediate consumer. No new native validation or throughput gain follows from
this controller change;14 legacy provenance holds remain pending before any build.

Main also applies the reviewed geometry formula to the retained117K histogram:
554.378TFLOP useful GU/Down work versus564.569TFLOP with source tile padding
(+1.8383%). This is not an instruction count or hardware ceiling; the exact
histogram remains only for the named production-shaped GEMM fixture.
After committing the component results, Main recovers22 original component
inputs and the CPU compile recipe, then retires15 closed output/CPU-fixture
files (433524B), including the successful full API trace. The compact result
retains its trace-presence/hash evidence and the deletion manifest. Current
candidate engine/nine fixtures and fallback3b/eight fixtures all match their
hashes after cleanup;14 older provenance holds remain separate.

Source inspection binds the native expert footer to deltas taken immediately
before the target verifier loop, after prompt prefill. For Q8/FP16 respectively,
CPU unique jobs are1483997/1281981, routed CPU entries1908954/1656916 and resident
hits230406/209804. Entries plus hits equal2139360/1866720 lookups, exactly
4457/3889 target rows times48 layers times10 experts. The row count also matches
1+4*(offered/3), for this spec4/suffix-draft0 configuration. These include the
initial one-row target window, exclude separate MTP computation and are logical
work counts, not physical RAM traffic or saturation. The former process0 field
at`5d3493678` accidentally selected its first warmup footer; the current result
corrects its scope without changing the original speed/admission/fault outcomes.
After original147 controller inputs, generated plans, pushed ancestry and live
dependencies were checked, the closed rejected root's74files/13318721B and13
consumed research transactions/23444B were retired with a path/hash/reason
manifest. Failed status, the real regression fixture/replay, reviewed method
reports, required native fixtures and both consumed engines remain intact.

A source follow-up rejects the suspected late prompt-bound optimization:
`generate.cpp` already sets the current prompt length before reading. With the
configured default32768-token MTP window and64-row guard, a fresh P-token prompt
needs at most`min(P-1,32831)` draft-KV rows. `draft_kv` already skips earlier rows
and whole chunks. The old33210-token diagnostic's first-needed378/needed32831
therefore match the current request. This is source accounting, not measured
inner work or a hardware ceiling. Do not scale that bounded MTP cost linearly
with ever-longer inputs or claim another skip optimization.

Logical cache-hit ratios are9.8/10.6%, with zero GPU miss-copy counters under
`pcie-frac0`. The actual48-layer native layout has seven format pairs and
1510400..2662400B per expert. Combining that range with unique CPU-job counters
gives logical provider-footprint bounds, not physical traffic or saturation.
Per-layer/format job counts and complete CPU/GPU costs are still missing.
The next work-mix discriminator can inspect existing`--dump-routing` callbacks
rather than add a native build. Multirow records contain layer, IDs and
placeholder weights; the footer counts callbacks, not row records. Bind request
boundaries/cache membership and reconcile reconstructed layer/group, lookup and
unique-CPU-job totals before using them. No such routing collection has run.
The current196632B profile is present and pinned (SHA`8f59b4aa...8975baf`);
source admission reproduces1256 expected resident pairs,26 per layer0..46 and34
in layer47. The first admitted model process fills1256/1256 and verifies slot0.
The canonical slot-order membership hash and source pins are in the research
index's`current_1256_route_workmix_contract`. This is expected static membership,
not full GPU readback or measured physical traffic. The report's missing-profile
premise used the retired pilot path; use`comparison.plans[0]` for the current
configuration, rather than the result's historical top-level pilot config.
The source contract adds current-route detail but overlaps the existing CPU-mix
protocol; consolidate those entries after consumption instead of retaining two
parallel general audits.

The closed capability output and consumed decision-support report/transaction
bodies are retired:43 external files/112835B, with a path/hash/reason manifest
in the compact capability result. All four original controller inputs,676
collector assets and the generated plan were checked before deletion. The
private capability CLI modes and unused imports are removed; existing attention
collection modes remain. Original source and plan stay reachable from pushed
`feature/sycl`. Both model engines are unchanged, with two current consumers and
14 separate legacy provenance holds. This cleanup changes no failed capability
gate and establishes neither a production observer nor a speed improvement.

For the separate earlier early-PCIe candidate, only a later qualified mechanism permits a
fresh complete51+6-case plan. Preserve all pending-completion,
untouched-output, payload/guard and owner-lifetime requirements under that plan.
Model execution remains blocked until the complete57-case gate passes.
After all57 passes, use `tools/sycl/early_pcie.py prepare`, commit the generated
plan, then `run`: two thinking-template requests with16 generated tokens each,
G1/T4 capture reached, positive copied/early groups, per-request group/copy and
ready/pending conservation. Preparation refuses an incomplete component gate.
This uses parameter validation and stage progress; full API tracing remains in
the component fixture. No clean speed or task quality comes from this small run.
Run the separate parameter-validated long model check before clean speed.
The model check needs32K+ fresh input,1024+ thinking-plus-answer tokens, positive
actual early launches/groups matching completed copies, normal ownership and no
fresh fault. Record audit readiness separately. A later committed six-process
copy-enabled comparison uses `tools/sycl/arena_copy.py prepare check
--early-scheduling` followed by its committed generated plan, then the six clean
cohorts. It changes only early0/1; readiness queries are enabled in the diagnostic
and disabled in clean timing. Every measured footer is bound to its actual input
and generated work, and early groups must equal completed copied blobs. A faster
screen requires paired ratios>1 and geomean
>1.02 required for a faster screen. Weak-MTP/agent-session, task quality,
prefill nonregression and physical262144/session/RESTORE remain adoption gates.
Compilation and the51 existing passes do not qualify early model execution,
performance gain or adoption.

A conditional complete-blob copy budget also constrains that candidate.
With at most2 resident experts per layer, top10 routing and at most4 emitted
tokens/window, even identical routes and perfect acceptance require at least
`(10-2)*98227200/4=196454400` requested miss bytes per emitted token if all misses
go to GPU. At the observed upstream encoding-only7.877GB/s budget this corresponds
to40.095tok/s; illustrative6GB/s gives30.541tok/s. This optimistic scenario is
below70tok/s, so full miss offload alone under this placement cannot supply
the target. CPU/GPU sharing, larger resident reuse and wider effective windows
remain distinct possibilities. The [calculation](../bench/results/2026-10-15-pr6-recovery-audit/client-resources-current.json)
records its assumptions; it is neither measured wire traffic nor a hybrid
throughput ceiling, and unchanged-work CPU scaling is not established.
Actual unprivileged perf `cycles` and `task-clock` command probes at`dea283917`
both exited1 with the access-limited diagnostic under`perf_event_paranoid=4`.
This preserves the earlier VTune ptrace refusal as a separate result; no host
permission was changed and model hardware counters remain unavailable.
After the source and results became default-reachable through PR139, Main
recovered and verified the newer build's9816 original source/dependency blobs,
recipe and final artifact hashes before retirement. The closed extra build,
resource/perf raw roots and consumed audit payloads were deleted:444 files,
151385935 bytes, with a path/hash/reason manifest recorded before deletion.
The selected old engine, its five original fixtures and source receipt remain;
its9813-input admission passed again after deletion. The fresh-build fallback
check now accepts the original hash-only fixture receipts while verifying the
relocated sizes/hashes and exact fixture selection. Its updated regression test
passed. Original execution, quality limits and failed performance gates remain
unchanged; this closure adds no throughput gain.
Recovered spans must fit the selected request interval. CPU ticks include
polling, and a shared DRM client's counters are not exclusive process usage.
Keep model quality,agent-session,physical262144/session/RESTORE and matched
no-MTP attribution as separate pending checks.

The maintained model controller now provides `prepare|run qsa-long-input-check`
and `prepare|run qsa-long-input [0|1|2]`. Its prospective screen holds131072
context/resident INT8 KV,8192 chunks,GU-large,ordered-exp QSA and GDN columns4
fixed; Down fusion is explicitly off. Minimum actual input is98304 and the
client's constructed122880-token budget plus24 output and256 guard requires
123160 context. The validation request must close before three independent
clean four-request processes. Keep first use separate from each process's
three subsequent samples; the target is a subsequent median>=1000tok/s in each
process. This measures longer-input amortization rather than a same-input
kernel improvement. Capacity/admission and actual route/token checks remain
required. The first parameter-validated117489-token request closed with native
exit0, thinking observed,24 generated tokens and no fresh fault. Its750.2tok/s
first-use reading is diagnostic only: the frozen route gate failed. At admission
only4257304576B were free; fixed reservations including512MiB headroom needed
4154841216B and the mixed bank needed419430400B, a316967040B shortfall.
Mode10 safely fell back to mode2, with98304 GPU residual rows and392888320B
host residual storage; the accumulator/GU fusion counters were absent. The
[failed screen](https://github.com/yqYo1/Strata/blob/2e83c15f6d4cc62ddaa65a99719dfb76bb3344bd/bench/results/2026-10-15-pr6-recovery-audit/prefill-long-input-current.json)
keeps its original failure and individual sample. Do not call this a matched
route comparison or a clean speed regression. Before a new configuration
probe, audit/test existing output-head release and restore, or explicitly
compare residual-prefix/window placement; preserve safety headroom and guards.
Selective Down timing remains the next pending component comparison.

The next capacity discriminator uses `qsa-long-input-balanced-check`, followed
by `qsa-long-input-balanced [0|1|2]` only after it passes. Its only placement
change is the existing `STRATA_PREFILL_LAYER_MAJOR_R_GPU=81920`; output heads
remain resident and the512MiB safety allowance stays fixed. Relative to the
failed automatic98304-row prefix, this releases335544320B for the mixed bank,
18577280B more than the observed admission shortfall. That is an admission
calculation, not measured free memory or speed. More residual rows move through
RAM, so their extra transfer cost must be included. The new check must prove
actual mode10,81920 GPU rows,GU-large and Down0; it links the original failed
screen in Git. Prioritize this measured VRAM cliff before spending another
clean timing cohort on the still-unqualified selective Down candidate.

The balanced validation request closed normally at117489 actual read tokens,
with mode10 and81920 GPU rows confirmed. Its first-use parameter-validated
reading is756.8tok/s, compared with the original diagnostic750.2; one sample
per configuration does not establish a speed gain. The mixed bank fits with
18581376B beyond the unchanged512MiB allowance, and the host residual is
728432640B. GU counters show5240 fused jobs/35581471 rows and16172 separate
jobs; Down remains separate for21412 jobs. No fresh fault occurred.
The clean four-request process then closed normally, with117489 fresh read
tokens per request, thinking on, mode10 and81920 GPU rows throughout. First
use was755.3tok/s; subsequent samples were802.1,802.1 and802.0 (median802.1,
sample SD0.0577). Ownership closed, boot unchanged and no fresh fault occurred.
The prospective futility rule (median<950 and all3 subsequent samples<975)
stops this configuration without the other two independent processes. The
route/correctness screen passed; the1000tok/s target did not. Keep all samples
in the [current result](../bench/results/2026-10-15-pr6-recovery-audit/prefill-long-input-current.json).
This is a different input size from the earlier78K~903tok/s reference and does
not establish a same-input regression or sustained decode performance.

The current-library output-head fixture passed two exact521472000-byte
detach/restore cycles, with all packed weights and projection logits unchanged
and different GPU addresses in both cycles. No fresh GPU fault occurred. This
isolated API result does not qualify the combined production transaction.
The subsequent same-engine model check was rejected before long-prefill work:
NativeDensePhase explicitly excludes head release, and the verifier's two
leases also reject nesting. Worse, error teardown ended in CPU SIGSEGV inside
the installed Level Zero driver; no new GPU fault was recorded. No head detach
occurred. Preserve this failed result and its diagnostic limits, and do not
repeat the configuration or remove the exclusion guard. Disabling existing
dense banking to bypass it is not the selected optimization.

The selected capacity path is bounded two-bank mixed storage using idle m.R
and the unused shared-arena tail. Source review establishes335544320 and
121566720 disjoint bytes, enough for the419430400-byte bank; it does not prove
runtime ownership or release any allocation yet. Implement checked bank
selection for preparation, arbitrary expert-major gathers and shared consumers,
with phase/owner validation before a traced component/model check. Keep PLE,
short-input fallback and final callbacks intact. Separately audit the newly
observed early-error teardown fault. Selective Down remains off and its clean
component timing is still pending.

`STRATA_SYCL_PREFILL_MIXED_SCRATCH=1` is the new default-off implementation of
this capacity option, initially restricted to mode10, native16, owned chunk8192
scratch and the existing single queue. It checks direct Prefill allocation
owners and registered device spans, retains every mixed row through shared
projection/HC completion, and uses the existing final checked drain before any
callback may reuse m.R or the arena. Preparation and shared consumers use the
same chunk/bank plan; a new GPU gather validates every row before loading either
bank and sets sticky status bit4 for invalid rows. No numerical conversion or
accumulation order changes. This is Main-authored Strata code, based on the
existing local planner/route interfaces and the reviewed source lifetime map;
no XeStrata implementation is used.

Source034bddff8393 passed the CPU capacity plan and all five GPU gather cases,
including banks crossed within one job,65536/65537 boundaries, partial windows,
invalid maps, rejected spans, guards and immutable inputs. The same-build GU
ownership, traced smoke,18-case validation and three-process timing also passed;
the fused/separate wall-time ratio geomean was0.95335. This is component evidence.

The first parameter-validated117489-token borrowed-scratch model request then
closed normally, with mode10,98304 GPU residual rows, zero extra mixed allocation,
thinking observed and no fresh GPU fault. All96 entry/retirement windows matched
the48-layer81920+16384 schedule and expected extents. Its794.2tok/s first-use
reading includes validation/logging and is not a clean speed comparison.
The clean cohort0 then closed with795.0tok/s on first use and845.9,846.1,845.8
on its three subsequent requests (median845.9, sample SD0.1528). All four actual
117489-token request hashes match the81920-row screen at commit0dc815171;
the new placement gains5.46% over its802.1tok/s subsequent median.
Each placement has one independent process; this is not a repeated-process
gain or an isolated kernel change. No fresh fault occurred. The prospective
futility rule stops this configuration without cohorts1/2;1000tok/s is not met.
Keep this opt-in capacity reference. Next measure remaining exact VRAM owners
and the changed route/resource costs. The target-cache startup0.26GiB is rounded
and cannot justify a release budget; head/dense composition remains excluded.
Short prefill, broader numerical/task quality, full physical262144 context and
RESTORE remain unqualified; this check does not establish default adoption or
sustained decode performance.

The selected next prototype reduces the temporary compressed-expert GPU bank
from512 slots (1363148800B) to384 (1022361600B), while retaining the existing
immutable512-expert RAM upload slab. Cache383 experts on first demand and use
one scratch slot for remaining experts on the same in-order queue. This saves
340787200B and admits all117488 residual rows in the pinned total-budget example,
without releasing heads or reducing the512MiB allowance. Extra weight uploads
between windows and their CPU/PCIe cost remain to be measured.
The committed CPU admission/ticket planner passes exact-budget, explicit-prefix,
stale-ticket and three short-tail fallback cases. It selects the original512-slot
path when full GPU residency is unavailable or a final1..8192-row window would
take normal FFN. Its262144-row case is capacity arithmetic, not physical-context
qualification. The query-only check reports10132807680B as the B570
max_mem_alloc_size, above the2406154240B residual. It submits no allocations or
kernels, so actual allocation success remains untested. The planner now accepts
an explicit per-allocation cap and must preserve it independently of total budget;
CPU cap-boundary/fallback checks passed at sourcecb74810ab, including stricter
hypothetical2GiB/1.5GiB devices, the exact residual-size boundary, invalid caps
and explicit full-prefix rejection. These remain CPU planning checks.
The default-off STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS=384 route now connects
that plan to the engine. Production and the new synthetic GPU fixture share
the same checked upload helper. Its private map is published after accepted
enqueue, with source/cache/status owners retained through the existing checked
lease. The original path still publishes all512 experts after its initial drain.
Staged requested H2D bytes/hits are counted separately: its uploads occur inside
the MoE dependency path, not the initial bulk-upload interval.
The corrected source1222e27d0 compiles successfully, including the production
engine and shared-helper GPU fixture. The initial attempt rejected the const
cache view at compile time, before any execution; the implementation now owns
an explicit mutable upload target rather than casting away const. The source034
validated engine/fixtures are retained externally for this immediate comparison.
The traced three-case GPU smoke and four-case validation passed with no fresh
fault. They cover383/384/512 slots, two-layer resets, scratch-only capacity1,
actual2662400B stride, all512 host experts, two queued readers per selection,
immutable source/slot guards, seven admission rejections per case and caller
unwind after queued work. Three old math/gather fixtures rebuild to identical
SHA256 bytes, so their existing checks remain component evidence; this does not
claim a fresh run or qualify the changed model orchestration.
The parameter-validated117489-token model request then passed, with all117488
engine rows on GPU, zero host residual,96 exact entry/retirement window pairs
and normal process closure. Requested expert H2D payload was40102860800B across
19631 uploads, with13594 permanent hits; unused experts are not bulk-uploaded.
The final layer used267 permanent slots. No fresh fault occurred. Its874.6tok/s
first-use reading includes validation/logging and is diagnostic.
All three clean independent processes then closed on the same frozen engine,
configuration and four request hashes, with117489 fresh read tokens per request:

| Process | First use, tok/s | Subsequent individual samples, tok/s | Subsequent median | Sample SD |
| --- | ---: | --- | ---: | ---: |
| 0 | 890.8 | 957.8,953.4,955.8 | 955.8 | 2.2030 |
| 1 | 888.2 | 956.6,957.9,957.9 | 957.9 | 0.7506 |
| 2 | 891.9 | 956.9,955.4,954.0 | 955.4 | 1.4503 |

All12 requests closed normally, with actual full-GPU residual placement,
normal native exits and no fresh fault. The median of the three process medians
is955.8; the pooled median of the nine subsequent samples is956.6.
The matched98304-row baseline has one process with median845.9, so the12.9921%
ratio to955.8 is still a screen against one baseline process, not a paired
three-process gain or isolated kernel improvement. The1000tok/s target remains
unmet. The individual process evidence is preserved at commits1b3b92d93,
1f302e6e4 and7f6d7c0fe; the compact current result records their parity checks.

The65536-minimum configuration screen used the same official benchmark
extension, retaining131072 resident INT8 context,
chunk8192,384 expert slots,borrowed mixed scratch,GU-large and Down0.
The fixed generator produced78451 actual read tokens and78450 engine rows.
The validation passed full GPU residency and all48 single-window owner/extent
pairs, with1606656000B residuals, zero extra mixed allocation,18484 weight
uploads/dequantizations and21514 jobs. Requested expert H2D was37750041600B;
there were no permanent hits because every expert is visited only once per layer.
Its833.8tok/s first-use reading includes validation and is diagnostic.
The clean process then measured835.0 on first use and930.9,921.6,930.4
subsequently (median930.4, sample SD5.2310), with normal closure and no fresh
fault. The prospective median<950/all samples<975 futility rule stops this
configuration without the other two processes. The1000tok/s target is unmet.
This different input size does not establish a same-input kernel regression.

The117K route's observer check passed, and the separate first-use/warm pair
completed normally without a fresh fault. Its original controller result remains
failed: a global owner count incorrectly compared192 windows against96.
The corrected offline analyzer independently validates96 windows per request,
complete CPU/coarse/work counts and four negative parser cases; it does not
rerun the model or overwrite the original failure. First-use/warm source copying
took5.15749/5.12719wall seconds (5.13619/5.06660CPU); warm frame wall was122.49556s.
Warm host grouping took0.25367CPU seconds. The951.1tok/s warm reading is profiled,
not a clean speed sample. Main keeps only aggregate attribution and checked
work totals; the full layer arrays and exact job-size histogram are unnecessary.

The next candidate is an opt-in immutable layer view from the already resident
Arena source, eliminating the512-expert staging copy. This may avoid a second
slab and worker without changing arithmetic or PCIe payload. It requires explicit
bounded source lifetime through checked queue retirement; transient/custom
providers keep synchronous copying. Ordinary host-pointer DMA may itself stage
or behave differently, so no speedup or1000tok/s claim follows from byte removal.
CPU-only next-layer double-buffering remains an alternative if direct borrowing
fails its correctness/performance comparison.

The prospective implementation is default-off under
`STRATA_SYCL_PREFILL_EXPERT_SOURCE_VIEW=1`. It applies only to the existing
all-GPU staged expert bank. Arena publishes bounded immutable host views from
its loaded-layout snapshot; the caller keeps the source open until checked
queue retirement. Empty provider capability or an unsupported bank route retains
the synchronous path. RAM admission counts an upload slab only when needed.
The upload helper separates host blob stride from the fixed GPU-cache stride;
each transfer still copies exactly that layer's blob bytes. No worker, GPU math,
cache policy, decode layout or requested PCIe weight payload changes.
Before any model run, require independent source-contract review,14 CPU view
extent checks plus existing planner/ticket checks, then traced small and full
GPU byte/guard/queue-unwind checks for both strides. Clean same-binary117K
off/on model comparisons follow only after these pass. The current engine and
same-build fixtures remain the validated fallback through qualification.

The independent source audit and native Luna-max review found the private-Arena
caller/stride/queue contract sound, while flagging externally mutable shared
file backing. Main disables resident views for every nonempty shared-arena path;
those sources keep the original synchronous copy. The next bounded gate also
uses the real Arena provider to check blob identity, later source calls, invalid
layers, close/reopen, File fallback and Linux shared exclusion, before synthetic
GPU upload/guard/unwind cases. Initial compilation and the14 CPU view extent
checks passed. The real provider checks and all six traced small/six full GPU
cases subsequently passed at build source`a57611b9c`, without a fresh fault or
boot change. These include fixed/compact source strides and checked unwind.
An earlier build stopped before execution because the provider target was not
registered in the separate SYCL CMake path; that failed receipt is preserved.
The official117K check subsequently passed actual48-layer view admission,
zero owned upload slab, unchanged384-slot/residual placement,96 owner windows
and complete work/CPU accounting. Requested expert H2D remained40102860800B.
The legacy source-copy region now prepares references in0.000038133wall /
0.000035870CPU seconds; actual bulk CPU copying is absent. Upload submission
still took4.94585wall /4.88016CPU seconds, and explicit wait CPU was78.90636s.
The125.91473s observer frame and923.2tok/s first-use reading are profiled;
they are not clean repeated performance. All owners closed normally without
a fresh fault. The first clean pair then closed normally on the same binary,
110 benchmark-input hashes and four fresh request hashes, with identical
requested expert H2D bytes for every matched request:

| Process | First use, tok/s | Subsequent individual samples, tok/s | Subsequent median | Sample SD |
| --- | ---: | --- | ---: | ---: |
| A0, synchronous copy | 891.6 | 957.4,952.7,951.6 | 952.7 | 3.0806 |
| B0, immutable view | 829.0 | 995.5,996.9,997.2 | 996.9 | 0.9074 |
| B1, immutable view | 908.7 | 995.8,995.9,996.1 | 995.9 | 0.1528 |
| A1, synchronous copy | 883.4 | 951.8,950.7,954.7 | 951.8 | 2.0664 |
| A2, synchronous copy | 886.2 | 956.4,956.0,957.6 | 956.4 | 0.8327 |
| B2, immutable view | 920.5 | 998.5,998.3,998.3 | 998.3 | 0.1155 |

All six processes subsequently closed in the frozen A0/B0/B1/A1/A2/B2 order,
with normal native exits, no fresh faults and identical requested expert payloads
for each matched request. The three paired subsequent-median ratios are
1.046394,1.046333,1.043810; their geometric mean is1.045512 (**4.5512%**).
All three pairs favor the view and exceed the unchanged prospective aggregate
criterion. These are three independent process-pair ratios, not nine independent
request ratios. First-use ratios0.929789,1.028639,1.038705 remain separate; the
initial829.0 regression is retained and its cause remains unproven.
The three view-process medians996.9,995.9,998.3 still miss1000tok/s.
Retain the view as an opt-in tuning path; default adoption, broader quality,
physical262144,session/RESTORE and sustained-generation qualification remain pending.

The committed offline analyzer recovered all110 benchmark inputs per process,
9813 original engine/dependency inputs and the original build recipe from Git.
Six fault-evidence and six adversarial statistics regressions passed; they do
not execute the GPU or qualify the pending Down component. The compact
[current comparison](../bench/results/2026-10-15-pr6-recovery-audit/prefill-long-input-current.json)
keeps all24 speed samples, spread, original source/receipt pointers and limits.
Do not equate disappearance of the source-copy category with a gain in other
workloads or requested payload equality with measured physical PCIe traffic.

Keep the compiled
fine command sampler off: it lacks a staged-expert-copy tag and would reject
those untagged events. Coarse event intervals include host submission gaps and
do not establish physical PCIe traffic or arithmetic saturation. The reviewed
double-slab source contract also requires audited providers, exclusive source
calls and join/error/RAM-admission qualification; no implementation is adopted.
The old47 intermediate residual stores plus47 inputs requested36931502080B
of host transfers; the new route replaces those with GPU-local copies. Staged
expert payload at117K is39.95..40.10GB/request versus the old bulk50.29GB. This does
not measure wire traffic or assign the speed gain to transfers alone: the
remaining FFN window also changes from16384 plus host chunks to35568 GPU rows.
Physical262144, production failure/session/RESTORE and default adoption remain
unqualified; this bounded screen does not establish sustained decode speed.

A read-only PCI topology check on2026-10-19 found16GT/s x4 at the card's
03:00.0 upstream bridge and its AMD ancestors. The endpoint and internal04:01.0
bridge instead report2.5GT/s x1, including as their maximum; interpreted as a
payload limit that contradicts the earlier approximately6GB/s H2D measurement.
The [retained topology readings](../bench/results/2026-10-15-pr6-recovery-audit/qsa-hardware-profile-current.json)
preserve the discrepancy. The observed upstream encoding bound is
16e9*4*128/130/8=7.877GB/s, using the [Linux encoding conversion](https://github.com/torvalds/linux/blob/master/drivers/pci/pci.h);
the [B570 product specification](https://www.intel.com/content/www/us/en/products/sku/241676/intel-arc-b570-graphics/specifications.html)
allows4.0x8. At that upstream bound,40GB would take5.08s if every requested byte
crosses it. This is a conditional payload estimate, excluding protocol overhead
and shared traffic; it is not measured wire bandwidth, additive model time or
evidence of PCIe saturation. No recovery or device configuration was changed.


## Cached-Q/KVvec4 parallel FP32 sum candidate

The component-only experiment at source`bba952e33` retained the ordered-exp path's cached Q,
vector K/V staging, maximum, ordinary `sycl::exp2`, alpha, BF16 probability
operand and FP32 FMA normalizer update. Each SIMD16 lane adds probabilities
for cells `lane` and `lane+16`; `reduce_over_group` then replaces the32 ordered
indexed exchanges. It adds no model dispatch or default selection. The
[SYCL group algorithm contract](https://github.khronos.org/SYCL_Reference/iface/group-algorithms-library.html)
does not fix a floating-point reduction order. The pinned
[SGLang QSA source](https://github.com/sgl-project/sglang/blob/877be2974aa151a984e52822c028276c2b1e9303/python/sglang/srt/layers/attention/qsa/sparse_attn.py)
uses an FP32 sum without specifying that order. No external implementation
code is copied. Existing ordered-exp bit/state requirements remain unchanged.

Before execution, the new fixture fixes exact maximum/alpha checks, nonnegative
finite sums/state, zero empty-prefix and positive nonempty-prefix normalizers,
a conservative gamma31 bound for any32-term association and a recurrence bound
for the single FP32 FMA update. Finite synthetic outputs must remain within
one BF16 ULP plus2e-6 of the ordered reference and pass the independent tensor
oracle. These are component-fixture limits, not a general task-quality tolerance.
Both probe variants must reproduce their own normal output and reject invalid
state spans; guards, diagnostics, input immutability, partial/missing/duplicate
rows, padded heads and physical262144 addressing remain required.

After small traced and full parameter-validated checks, use three clean processes
with six rotated rounds for cap33 gathered and cap2051 gathered/clustered:
108 timing samples. Advance only if candidate/base latency geomean<=0.95,
no shape>1.05 and both cap2051 patterns improve in each process. Complete
wrapper wall time includes admission, prefix preparation and checked drain;
it does not prove arithmetic saturation or physical VRAM traffic. The proposed
mechanism reduces subgroup dependency/exchange work, with unchanged PCIe
payload. Source`bba952e33` compiled successfully. The initial traced small check passed
both two-case numerical fixtures and the small timing-fixture sanity check,
including state-span rejection checks, with normal closure and no fresh fault.
Both full11-case numerical fixtures subsequently passed, including the
physical262144 addressing cases, with normal ownership closure and no fresh
fault. All three clean timing processes then completed without a fresh fault.
Candidate/base latency ratios were0.995263 for cap33 gathered,0.995479 for
cap2051 gathered and0.990206 for cap2051 clustered. All cap2051 process medians
improved, but the geomean0.993646 misses the prospective0.95 gate.
**This candidate does not advance to a model trial.** The full108 samples remain
in [the current component result](../bench/results/2026-10-15-pr6-recovery-audit/qsa-parallel-sum-current.json).
This is evidence that this isolated reduction change has a small effect in
these shapes, not a physical-bandwidth or occupancy measurement. Physical-context
model qualification remains separate from the component checks.
The a576 validated engine/fixtures remain the fallback while the new native
build used the existing comparison cache. At that pinned source, commands were
`build_engine.py rebuild-parallel-sum`, followed by
`check_attention.py smoke|component|perf parallel-sum`, committing each phase
before the next. Native fixtures and all controlling inputs must be committed
before the build. The exact current result is written to
`bench/results/2026-10-15-pr6-recovery-audit/qsa-parallel-sum-current.json`.
The rejected prototype exports, conditional tests and build/measurement modes
are retired from the active tree; restore source`bba952e33` for a concrete new
comparison. Keep the independent fault-observation robustness fix. Source and
recipe recovery preceded retirement of closed raw logs and unused fixtures;
the known-good a576 fallback is retained for the next Down comparison.
