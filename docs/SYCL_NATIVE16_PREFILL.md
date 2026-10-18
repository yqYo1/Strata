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
