# Native 16-bit long prefill

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
No lighter-accounting rewrite is selected from these values. Next compare the
retained old/new engines without observers on the same runtime, and collect
bounded thread/resource attribution before selecting an inference optimization.
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
asset hashes remain unchanged. Keep the current8c5d799f engine and570b65dd
fallback only for the next same-runtime observer-off old/new engine comparison;
retire the extra after that decision, before another native build. The new
single-workload selector does not replace the required strong/weak pair for adoption.
Use `generation_engine.py prepare|run engine-compare [0..5]`, committing each
prepared plan and closed result. The frozen order is old/new/new/old/old/new,
with the unchanged official regular-code request and four warmups. Both arms
remove CPU/wall/validation observers before native exec and preserve the safety
settings. Original fallback inputs/recipe and both runtime checks bind the
selected binaries. The diagnostic nonregression gate is new/old per-token
latency geomean<=1.05 and maximum pair<=1.10 after request/work reconciliation;
report individual samples and spread even if this gate fails. No inference
optimization, runtime-baseline adoption or resource saturation follows from
this single-workload comparison alone.
Independent source audit and native Luna-max review found no technical reuse
blocker. The automatic off-arm consumer lacks an explicit four-warmup phase
assertion: process0/1's original journals contain all four successful warmups,
but all six journals must pass that check before interpreting the comparison.
Retain them for that closure check; `passed` alone does not satisfy it. Residual
null-counter work is the intended old/new test subject, not proof of a defect.
The audit worker is completed and its terminal is closed; the frozen research
index will be reconciled after the comparison, without changing its input hash.
The first old/new pair closed at13.1/11.8tok/s (original results`c1e33bdf0` and
`b6e70064d`). Both read33211 fresh tokens, generated2931 including thinking and
accepted2105 of2481 drafts, with normal closure and no fresh GPU faults. The
new/old per-token latency ratio1.115806577 exceeds the unchanged1.10 per-pair
limit. Preserve this slower sample and finish the two remaining planned pairs;
one pair does not establish the cause or qualify an engine choice.
The next independent new-engine process closed at14.4tok/s (`4de041ebb`),
with the same33211/2931 token work and2105/2481 drafts, successful warmups and
normal fault-free closure. Its adjacent old-engine partner remains pending;
the11.8-to14.4 spread already shows why the original repetitions are needed.
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
at`3625c5f7b` passed all8 synthetic CPU-only tests with
`python3 -B tests/core/test_client_resources.py`. It rechecks PID/start/SID,
deduplicates descriptors, preserves raw cycle counters and recovery spans,
rejects impossible deltas and reads scheduled CPU ticks. The CLI has a900s/
32MiB bound and opens no GPU device. No live collection has been qualified;
its next consumer is the bounded resource diagnostic after this frozen cohort.
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
