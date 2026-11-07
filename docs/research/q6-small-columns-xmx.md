# Q6_K small-column head: source and dispatch reference

**Scope and status.** This is a source/dispatch reference for the resident Q6_K
head `[K=2560,N=248320]` at `C=1` and `C=4`; it is not performance evidence.
`C=1` is ineligible for the pinned GGML Q6 XMX path. `C=4` is conditionally
source-eligible. Intel's device/AOT tables support the B570-to-BMG G21 family
mapping, but local runtime compatibility and Strata dispatch remain unproved.
`schedulable_now=false`: the separate component gate and a prospective
numerical, lifetime and residency plan remain open. Source availability does not qualify a component
or model route.

## Control and candidate

The fixed local control is native source commit
`05a9b45c0a3a266779d4757a855b45a3bb42a62e`:
`src/kernels/sycl/native_mmvq.cpp` SHA-256
`e11821dc10a71670236d45f9bdb1387b185118389e000d42385abbe9ce7197e4` and
`include/strata/kernels/native_mmvq.hpp` SHA-256
`1798072d98d3b15ee8fbdeb067d81df5c45dfed6378907748bb9eb09cc198e13`.
It keeps Q6_K in 210-byte row-major blocks, quantizes activations to Q8_1,
and returns FP32. The production control uses DP4A; its calls enqueue on the
caller queue and use caller-owned scratch, with no allocation or wait inside
the kernel. Its exact multi-column layout preserves the C1 result when
`native_mmvq_multi_exact()` is enabled. Keep this fixed source as the semantic
and dispatch control, not a mutable proposal.

A Q6_K block stores 256 weights in 210 bytes: 128 low-bit bytes, 64 high-bit
bytes, 16 signed scales (one per 16 weights), and a 16-bit super-scale. The
target's logical packed weight payload is `521,472,000 B` (521.472 MB or about
497.2 MiB); full FP16 expansion is `1,271,398,400 B`. These are storage sizes,
not measured traffic, peak residency, or a hardware ceiling.

## Official compressed-Q6 int8-DPAS source

The source is llama.cpp commit
[`34ca79122b44d43eda7efb203b82ec91db94c55c`](https://github.com/ggml-org/llama.cpp/tree/34ca79122b44d43eda7efb203b82ec91db94c55c),
MIT licensed ([LICENSE](https://github.com/ggml-org/llama.cpp/blob/34ca79122b44d43eda7efb203b82ec91db94c55c/LICENSE));
the Q6 XMX code is part of PR [#29864](https://github.com/ggml-org/llama.cpp/pull/29864),
whose reported test device was Arc Pro B70, not B570 or Strata. This is an
immutable implementation reference. A prior pinned revision,
[`3cf03257f219afbe7334045ff7c6a06ac68c627d`](https://github.com/ggml-org/llama.cpp/tree/3cf03257f219afbe7334045ff7c6a06ac68c627d),
uses compressed Q6 MMVQ through DP4A or FP32 ESIMD; that older path is not the
DPAS specialization below.

Build-time `GGML_SYCL_MMVQ_HAS_XMX` requires Intel LLVM and the DPAS header
([`common.hpp`](https://github.com/ggml-org/llama.cpp/blob/34ca79122b44d43eda7efb203b82ec91db94c55c/ggml/src/ggml-sycl/common.hpp)).
Runtime enable requires `g_ggml_sycl_enable_xmx_esimd` and `gpu_arch` equal to
BMG G21, BMG G31, or LNL M. The MMVQ selector checks supported op/type and
dimensions, reordered weights, K divisible by 256, a 64-byte-aligned weight
pointer, a 64-byte-aligned Q8_1 base, and a 16-byte activation pitch
([`mmvq.cpp`](https://github.com/ggml-org/llama.cpp/blob/34ca79122b44d43eda7efb203b82ec91db94c55c/ggml/src/ggml-sycl/mmvq.cpp)).
Q6_K has a two-column minimum; `GGML_SYCL_XMX_MAX_COLS` defaults to 80.
Therefore C1 fails the Q6 count gate; C4 meets that gate only. Local runtime
enum mapping, compiler/runtime compatibility, successful reorder and observed
dispatch remain unproved; this upstream selector is not Strata's call path.

Intel's [Xe driver table](https://dgpu-docs.intel.com/overview/supported-hardware/xe-driver-gpus.html)
maps PCI ID `E20C` to B570, Xe2 and Battlemage. Its [AOT target table](https://www.intel.com/content/www/us/en/docs/fortran-compiler/developer-guide-reference/2026-1/ahead-of-time-compilation.html)
lists B570 under `bmg-g21`. Together these support family-level compatibility
with the pinned GGML branch. They do not establish the local runtime's enum,
installed compiler/headers, IGC/runtime/driver pairing, DPAS execution, Strata
dispatch, numerical correctness or speed. A later C4 screen must pin and verify
those local inputs and the executed route; C1 remains a negative selector
control and the existing DP4A path remains the numerical/dispatch control.

The Q6 XMX loader rearranges each block as `[ql:128][qh:64][signed
scales:16][d:FP16]`. It reconstructs signed int8 weights as `q-32`; each K=32
dot is split into two K=16 halves to apply Q6's signed per-16 scales, with the
other half masked for each DPAS operation. Q8_1 provides 32 signed activation
bytes and FP16 `(d,s)` per block; this symmetric Q6 dot uses `qs` and `d`, not
the stored sum `s`. In the epilogue, `scales_t::get()` forms the FP32 product
of the signed Q6 scale and its FP16 super-scale. Each int32 partial is
converted to FP32 and multiplied by that combined row scale; the two half
results are summed, then multiplied by Q8_1's FP16 `d`. Preserve this FP32
association/reduction order as a numerical gate
([`esimd.hpp` traits and epilogue](https://github.com/ggml-org/llama.cpp/blob/34ca79122b44d43eda7efb203b82ec91db94c55c/ggml/src/ggml-sycl/esimd.hpp)).

`reorder_qw_q6_k` copies the full source allocation to a same-sized temporary,
then rewrites the original device allocation in place; reordered bytes persist
there, with no separate persistent destination. A full-head reorder temporary
is `521,472,000 B` (521.472 MB/about 497.2 MiB); this is not measured peak
memory or total process footprint. `sycl_reorder_temp_buffer` can use host
memory under `GGML_SYCL_HOST_MEM_FALLBACK`, so device allocation failure does
not necessarily skip reorder; host staging changes transfer and residency
costs ([reorder](https://github.com/ggml-org/llama.cpp/blob/34ca79122b44d43eda7efb203b82ec91db94c55c/ggml/src/ggml-sycl/ggml-sycl.cpp#L4494-L4540),
[temporary/fallback](https://github.com/ggml-org/llama.cpp/blob/34ca79122b44d43eda7efb203b82ec91db94c55c/ggml/src/ggml-sycl/ggml-sycl.cpp#L3979-L4045)).
K=2560 and N=248320 are multiples of 256, so this logical shape has no K/N
block tail; alignment, row pitch and per-tile row clamping still apply. A
Strata implementation must separately prove temporary and weight ownership
through completion, queue/event dependencies, graph capture/replay lifetime,
actual admission and maximum single-allocation feasibility. Cold reorder/setup
and warm reuse must be measured separately.

## A16 reference and distinct hypothesis

SGLang `sgl-kernel-xpu` commit
[`22d8550db80f9366c7e263d9e52b01aa1ad57040`](https://github.com/sgl-project/sgl-kernel-xpu/tree/22d8550db80f9366c7e263d9e52b01aa1ad57040)
is the closest transient A16 example. Its `moe_grouped_mm_nt_xe20_w4a16`
kernel accepts packed W4 expert weights with grouped scales (32/64/128/256),
BF16 or FP16 activations, and a grouped MoE shape; it dequantizes operand
fragments for Xe20 DPAS and accumulates in FP32. It neither unpacks Q6_K's
planes/per-16 scales nor provides an LM-head route. The XPU docs scope this
quantized path to MoE experts; attention, router, embeddings and LM head remain
BF16 ([matching docs revision](https://github.com/sgl-project/sglang/blob/4c7c775cf4adcbd2c7419d13b51ec838a17e559e/docs/docs/hardware-platforms/xpu.mdx)).
The cited kernel file carries BSD-3-Clause and its repository root LICENSE is
MIT ([file notice](https://github.com/sgl-project/sgl-kernel-xpu/blob/22d8550db80f9366c7e263d9e52b01aa1ad57040/src/sycl/GroupGemmW4A16Xe20.cpp),
[root license](https://github.com/sgl-project/sgl-kernel-xpu/blob/22d8550db80f9366c7e263d9e52b01aa1ad57040/LICENSE)). No source was copied;
any reuse must preserve file notices and audit transitive provenance.

A Q6-to-A16 path would be a separately authored hypothesis: load packed Q6,
reconstruct codes/scales, convert weight and activation fragments to one
chosen format (BF16 or FP16), then DPAS with FP32 accumulation. It is not
implemented by the cited W4 kernel and does not retain the Q8_1 numerical
contract. Its plan must state scale application, both conversions and rounding,
reduction order, output type and task/logit acceptance; FP32 references are
diagnostic, not a universal precision threshold.

## Conditional next choices and gates

At most two candidates merit later consideration:

1. If the active clean model comparison closes and the component gate permits
   it, test an isolated C4 Q6×Q8_1 int8-DPAS production route, retaining the
   current DP4A route as control. Prove B570 build/runtime eligibility and
   observed dispatch first; include C1 as a negative dispatch control. Compare
   cold reorder/setup and warm reuse, resident admission and full production
   path. Check Q6 unpack/scales, Q8_1 `d`/unused `s`, FP32 association, tails,
   alignment, ownership, queue/graph lifetime and fault behavior.
2. Only if a distinct A16 question remains after that screen, choose one of
   Q6→BF16 or Q6→FP16 and write its numerical contract prospectively. Compare
   against the Q8_1/FP32-order control on identical C1/C4 inputs and validate
   task/logit effects, residency, conversion scratch, bounds and lifetime.

Before either component execution, close the declared model cohort/gates and
commit a phase-specific plan with order, repetitions, acceptance and stopping
rules; pin production caller, helpers, inputs, source, recipe/compiler/runtime
and executable; prove actual dispatch and memory admission; preserve the
validated fallback and decode behavior; and pass numerical, bounds, lifetime
and fault checks. Any later model route also needs the existing full 262144
physical-context, tail, prefill-to-decode, RESTORE and decode-preservation
gates. C4 component latency, source presence and B70 results cannot establish
speed, quality or adoption.
