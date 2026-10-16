# Optional vLLM Xe2 native16 backend (R821)

Source proposal only. Default OFF and not adopted. Main owns all compilation,
execution and qualification. This document reports no build, flag acceptance,
ISA, GPU, numerical, fault, performance, task-quality or physical-context result.
The original FP32 GEMMs, short-input/decode source and routing stay unchanged.
R818's independent typed-attention proposal is not included here.

## Build and select

Review the complete R821 patch against its pinned production source first.
Keep the already qualified official LLVM compiler and existing local ggml/MKL
paths. Add these cache settings to Main's isolated whole-engine build recipe:

```text
STRATA_ENABLE_SYCL=ON
STRATA_SYCL_VLLM_XE2_NATIVE16=ON
STRATA_SYCL_DEVICE_ARCH=bmg-g21-a0
STRATA_SYCL_AOT_DOUBLE_GRF=OFF
STRATA_PORTABLE=OFF
SYCL_TLA_SOURCE_DIR=/home/yayoi/ghq/github.com/intel/sycl-tla/.worktree/strata-native16-87f
```

`SYCL_TLA_SOURCE_DIR` must already exist and be absolute. It is the official
Intel sycl-tla source at `87f6850680a580654b9ea2c80dbc01aeb36ad231`.
The CMake module checks the exact bytes of all 207 conservative R810 include
closure headers, including dirty modifications. It does not run dependency
CMake, Git, FetchContent, a download, or an installation step. The inventory
is source metadata; no TLA header is vendored. Revision identity outside the
consumed header closure remains Main's source-provenance responsibility.

The adapter receives C++20, `CUTLASS_ENABLE_HEADERS_ONLY`,
`CUTLASS_ENABLE_SYCL`, `SYCL_INTEL_TARGET`, `-O3 -fsycl`,
`-fsycl-targets=spir64_gen`, `-fno-fast-math -ffp-contract=off`,
`-fno-sycl-instrument-device-code`, and `-ftemplate-backtrace-limit=0`.
Final device link requests the exact R810 translator extensions:

```text
-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate
-Xsycl-target-backend=spir64_gen "-device bmg-g21-a0 -internal_options -cl-intel-256-GRF-per-thread"
```

Kernel properties remain R810's subgroup16/GRF256. These flags and properties
are requests; compiler/ocloc acceptance and generated instructions are UNRUN.
The Runtime target supplies one canonical backend option string, avoiding
conflicting `-device`/GRF options. The opt-in global GRF256 link option can also
affect other SYCL images, including decode, even while their sources and route
are unchanged. Main must record that whole-engine build difference and assess
it; an isolated adapter-flag effect cannot be inferred from a benchmark.
With the option OFF, the existing AOT/JIT flags remain as before.

Set immutable child environment before native exec:

| Setting | Effect |
| --- | --- |
| `STRATA_SYCL_PREFILL_NATIVE16=bf16` | Existing native16 bulk prefill mode |
| `STRATA_SYCL_PREFILL_NATIVE16_BACKEND=vllm` | Both public `*_out16` calls use the actual R810 adapter |
| Backend unset or `own` | Existing joint-matrix GEMM16 route |
| `STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE=auto` or unset | Initial static rule: T<128 compact, otherwise wide |
| Tile `compact` | Explicit R810 16x64x32, 64-thread experiment |
| Tile `wide` | Explicit R810 128x128x32, 128-thread experiment |

The threshold is an untuned source choice, not a performance finding. Explicit
tile controls enable separate-process experiments without environment mutation.
Invalid names reject; explicit compact/wide with backend own also rejects.
Requesting vllm in an OFF build rejects during `Gemm::init_external` (or the
first out16 call). A compiled vllm request rejects non-Intel/non-Level-Zero/
non-G21 queues at initialization. The unchanged adapter checks the full device
capabilities before its first submission. No G31 image is compiled by this hook,
so the wrapper restricts the otherwise G21/G31-capable R810 source to G21.

One concise `[native16-route]` line is emitted per successful process config
initialization, including existing bulk mode `engine=0|bf16`, backend and tile.
For vllm it also identifies G21 AOT/R810/UNQUALIFIED. `engine=0 backend=vllm`
does not mean the whole model uses 16-bit GEMMs: only actual out16 calls route
there. To benchmark the existing bulk chain, use `engine=bf16`. Failed or never
reached initialization need not emit a line; there is no per-kernel trace.
Keep separate processes for each arm and retain the actual effective environment.

## Tensor and failure contract

The imported R810 `native16.hpp`, `native16.cpp` and `xe_gemm_bsd.hpp` are
byte-identical to the source package. `f16_out16` selects FP16 X and W;
`bf16_out16` selects BF16 X and W. `output_bf16` independently selects Y.
All four combinations reach the corresponding actual adapter template path.
X is compact `[T,K]`; W is compact `[N,K]`, used transposed mathematically;
Y has shape `[T,N]`, physical row pitch `ldy`, with `ldy=0` meaning N.
FP32 accumulation/reorder fragments remain local, followed by a direct typed
16-bit block store. No global FP32 output, scratch tensor, cast kernel,
CPU GEMM, oneDNN/oneMKL fallback, packing or transpose is introduced.

The vllm selection requires beta +/-0; nonzero, NaN or infinite beta rejects.
It requires T,N,K>0; N%64=0; K%32=0; ldy>=N and ldy%8=0; and all base pointers
64-byte aligned. The entire X and W rectangles and entire Y[T,ldy] rectangle
must lie in live registered Strata Device allocations in the supplied queue's
actual context/device. Y cannot overlap either input; read-only X/W aliasing
is allowed. Host USM, foreign raw USM, ordinary host pointers, overflow, unsupported
architecture or missing capabilities reject without GEMM submission. There is
no route retry, silent fallback, tail padding or reinterpretation as float.
The exact R810 guard also caps width/pitch/height at 0xFFFFFF, complete byte
rectangles and total work-items at INT_MAX, and requires Linux LP64.
Ragged T keeps the real logical surfaces; block2D tail behavior still needs Main's
hardware checks. Unsupported N/K tails are rejected rather than claimed safe.

The default own route retains its original beta, N/K%16, 2-byte alignment and
last-live-output-element extent contract. The vllm route's stronger constraints
are intentional and visible. Synchronous guards run before this GEMM enqueue;
a preceding producer/dequantization may already be outstanding on the queue.
On any enqueue/asynchronous exception callers must retain owners and drain or
fail-stop using existing checked cleanup. A timeout does not authorize frees.

## Existing Runtime and lifetime

The production allocation registry now retains each Allocation's original
`shared_ptr<Runtime>` and can return that existing owner after the same actual
allocation-extent/context/device checks. It creates no Runtime, provider,
ordinal selection or queue. Existing `validate_device_span` delegates to that
same validation and preserves all previous rejects. Its returned shared owner
adds reference-count work; quantify host overhead separately if relevant.
The adapter target links the one existing `strata_sycl_runtime`; it never compiles
a second Runtime source or R810's standalone fixture.

Each production Gemm lazily stores a state that retains that original Runtime
and constructs the unchanged adapter over the original stream's queue. It
bootstraps ownership from one registered X element; R810 then validates all full
spans before enqueue. There is no extra device allocation or scratch. The queue
must have been created by that Runtime, stay alive at its original address,
and retain its original async handler. Matching SYCL domains do not independently
prove queue origin; this remains the original engine's caller obligation.
The hook checks the borrowed queue address on reuse. Gemm calls/init/rebind and
allocation destruction remain serialized by the caller. A buffer owner is not
retained by retaining Runtime: every X/W/Y owner must outlive submitted work.
The existing in-order queue expresses producer/consumer dependencies. The
actual returned GEMM event uses the original `finish(stream,event)` contract,
so non-null streams remain asynchronous and null stream retains its existing
synchronous behavior. No default-provider call is added; the old null-stream
`queue_for` path itself still has its original default-runtime semantics.

## Long-prefill call-site inspection and limits

This is source inspection of production `9c1a2020`, not model-shape or allocation
runtime evidence. All sites keep the original row/column/stride values:

| Site | Actual source mapping | Static compatibility and remaining limits |
| --- | --- | --- |
| `bf16_proj16`, HC down/up | `T=rows`, N=`ne1>0?ne1:1`, K=`ne0`, compact Y | HC constants LR=320,D=10240 are N%64/K%32 compatible. N=1 fallback would reject. Resident BF16 weight base and actual extent are checked. |
| `native_proj16`, GDN/QSA output and shared experts | Per column slice r: `T=rows`, N=`min(scratch_elems/K,n-r)`, K=`ne0`, Y=`y+r`, ldy=`ne1` | No copied/packed layout. Existing scratch has 33,554,432 elements. A matrix that fits scratch uses r=0. Larger matrices can produce non-64 N, non-64-byte y+r, or a conservative T*ldy span beyond the standalone logical Y allocation; those reject. No generalized slice support is claimed. |
| Routed expert gate/up | X=`Xs+scratch_row*2560`, W=`dq_gu[q]`, Y=`GU_b16+scratch_row*1280`, T=ne,N=1280,K=2560 | All row offsets preserve 64-byte alignment if bases are aligned; T is ragged and must stay within real registered extents. |
| Routed expert down | X=`Hh+scratch_row*640`, W=`dq_d[q]`, Y=`Dm_b16+o0*2560`, T=ne,N=2560,K=640 | Offsets likewise preserve 64-byte alignment; true bounds are checked. |

`Alloc::take` rounds each region with 256-byte spacing, which preserves alignment
from an aligned arena base; it does not prove a user-supplied base is aligned.
The registered Allocation bounds remain authoritative. HC/sharing/routed
constants provide shape evidence only; `WeightRef::ne0/ne1` vary by actual model
and cannot be inferred from comments. Very large rectangles fail the R810
INT_MAX guard. No support claim is made for a physical 262144-row full matrix,
for grouped MoE, arbitrary variants, column-sliced matrices, or R818 sites.
The routed calls are individual dense GEMMs under the existing CPU grouping and
copy schedule. This hook does not port or remove that host grouping work.
Router/injection/scalar shared gate and existing QSA/GDN internals remain FP32;
changing this backend does not implement the independent attention proposal.

## Source and license acknowledgement

Intel's 2025 BSD-3 copyright, conditions and disclaimer are retained in full in
`src/prefill/sycl/vllm_xe2/xe_gemm_bsd.hpp`. Its actual tile body derives from
vllm-project/vllm-xpu-kernels `6b30c9b92cc525c32729df988b5f918e7c5661a5`,
`csrc/xpu/grouped_gemm/xe_2/gemm_xe2.hpp`, through the exact R810 adaptation.
That core remains BSD-3 and is not relicensed as MIT. The R810 adapter/header,
new engine hook/build glue/inventory/docs and new project changes follow Strata's
repository MIT LICENSE. No unmarked vLLM policy, grouped scheduler, Torch wrapper,
or XeStrata implementation/helper/test is included or used as a design basis.

External TLA's conservative include inventory contains 192 BSD-3-Clause files
and 15 Apache-2.0 WITH LLVM-exception files. Original external notices remain
unchanged. Source/binary redistribution must retain the relevant Intel/NVIDIA BSD
notices, conditions and disclaimers and applicable Apache/LLVM-exception notices;
the root BSD label alone does not cover every transitive header. The R810
`provenance.md` and `manifest.json` retain the per-header attribution evidence.
This acknowledgement does not replace the actual third-party terms.

Before integration/adoption Main must review the full patch and imported notices,
qualify R810 and the enabled whole-engine link/flags, compile every type/tile,
inspect actual device images, run guards/numerics/chains plus unchanged old
GEMM checks, then perform bounded real-model correctness and whole-dependency
performance measurements. Resolve any exact arithmetic/subnormal mismatch
without weakening the prior result. Quality and physical-context gates remain
independent. Keep this option OFF pending those decisions.
