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
the executed prefix33 and row-publication kernels to128GRF/SIMD16, static SLM32B
plus28224B of dynamic-local arguments, and no emitted scratch descriptor.
These are compiler/launch properties, not hardware utilization or bandwidth.
The unlaunched compiled score32 variant has1920B spill metadata and2048B scratch;
it must not be attributed to the executed production prefix33.

The next independently authored `qsa_bf16_prefix_qcache_batch` candidate loads
the same16 BF16 Q fragments per QK subgroup once, preserves their DPAS order,
and reuses their initial shared arena for K/V after a uniform workgroup barrier.
It requests20036 local bytes, including a nonzero four-byte unused accessor;
emitted GRF/spill cost and actual speed remain unmeasured until its gates run.
The unchanged production prefix is the comparison arm. The candidate remains
fixture-only; no model dispatch or numerical-contract change is implied.

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
