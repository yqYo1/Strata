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
- QSA and GDN internal projection/attention tensors still use the existing
  FP32 kernels. Their replacement is a separate implementation step.
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
and [paired model receipt](../bench/results/2026-10-15-pr6-recovery-audit/own-native16-model-smoke-v2.json).
An earlier new allocation-registry destruction-order error was reproduced and
fixed separately; the fixed model processes exit normally.

The first official `bench.sh` comparison on the B570 used the same binary and
78,453-token request, thinking on, chunk 8192 and 32,768 GPU residual rows.
FP32 measured 346.5 tok/s; BF16 measured 123.8 tok/s (one process per arm).
The BF16 GPU residual fell from 1,280 to 640 MiB, but this implementation's
throughput fell 64.3%. It is rejected for default/performance adoption pending
operator diagnosis and improvement. See the
[individual-sample comparison](../bench/results/2026-10-15-pr6-recovery-audit/own-native16-first-long-comparison-v1.json).
This is a first screening pair, not a repeated variance study or evidence that
the datatype alone caused the slowdown.

Natural-input quality, improved long-input speed, physical 262144-context and
RESTORE qualification remain required. Keep the default FP32 route.

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
